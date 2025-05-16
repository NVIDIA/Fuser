// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/reorder_sharded_axis.h>

#include <device_lower/utils.h>
#include <fusion.h>
#include <host_ir/lower.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <transform_replay.h>
#include <scheduler/utils.h>

namespace nvfuser::preseg_passes {

namespace {

void reorderForGatherBasedComm(
    Expr* expr,
    CommunicationInfo communication_info) {
  TensorView* input = expr->inputs().at(0)->as<TensorView>();
  TensorView* output = expr->outputs().at(0)->as<TensorView>();

  IterDomain* sharded_id = communication_info.p_sharded_id;

  // For gather operations i.e. ID goes from sharded to unsharded
  // this will rematerialize a sharded axis.
  // ProcessGroup expects contiguous tensors.
  // Update input to push the rematerialized axis to the front -> collective
  // -> permute the rematerialized axis to the proper location
  // Example: [i0 DIDx(i1)] -> [i0 i1]
  // Rewritten to: [i0 DIDx(i1)] -> [DIDx(i1) i0] -> [i1 i0] -> [i0 i1]
  // Note: there are no reduction based collectives that
  // materializes an axis so expr is guaranteed to be a set.
  int64_t sharding_axis = input->domain()->rootPosOf(sharded_id);
  TensorView* input_permute = permute(input, {{sharding_axis, 0}});
  TensorView* output_permute = set(input_permute);
  TensorView* new_output = permute(output_permute, {{0, sharding_axis}});
  ir_utils::replaceValInAllExprInputsAndFusionOutputs(output, new_output);

  // Propagate shardings from input and manually apply sharding deletions.
  shardAllLike(input, {input_permute, output_permute, new_output});
  output_permute->axis(0)->parallelize(ParallelType::Serial);
  new_output->axis(sharding_axis)->parallelize(ParallelType::Serial);
  output_permute->setDeviceMesh(output->getDeviceMesh());
  new_output->setDeviceMesh(output->getDeviceMesh());

  // Set allocation domain for comm in/out
  input_permute->setAllocationDomain(input_permute->getLoopDomain(), true);
  output_permute->setAllocationDomain(output_permute->getLoopDomain(), true);
}

void reorderForScatterBasedComm(
    Expr* expr,
    CommunicationInfo communication_info) {
  TensorView* input = expr->inputs().at(0)->as<TensorView>();
  TensorView* output = expr->outputs().at(0)->as<TensorView>();

  IterDomain* sharded_id = communication_info.c_sharded_id;

  // For scatter operations i.e. ID goes from unsharded to sharded
  // Update input to push the scattered axis to the front -> collective ->
  // permute the sharded axis to the proper location.
  // Scatter example: [i0 i1] -> [i0 DIDx(i1)]
  // Rewritten to [i0 i1] -> [i1 i0] -> [DIDx(i1) i0] -> [i0 DIDx(i1)]
  // Reduce Scatter example: [i0 DIDx(i1) i2] -> [i0 r1 DIDx(i2)]
  // Rewritten to: [i0 DIDx(i1) i2] -> [i2 i0 DIDx(i1)] ->
  //                    [DIDx(i2) i0 r1] -> [i0 DIDx(i2)]
  // Note that reduction axis shifts from axis=1 to axis=2.

  int sharding_axis = static_cast<int>(output->domain()->rootPosOf(sharded_id));

  TensorView* input_permute = permute(input, {{sharding_axis, 0}});
  TensorView* output_permute = nullptr;
  // Calculate the number of reduction axes before the sharding axis.
  // After permuting the sharding axis to the front, the reduction axis
  // will be offset by this amount.
  auto reduction_axis = output->getReductionAxis();
  int num_reduction_axes_before_sharding_axis =
      (reduction_axis.has_value() &&
       sharding_axis > static_cast<int>(reduction_axis.value()))
      ? 1
      : 0;
  if (communication_info.type == CommunicationType::ReduceScatter) {
    auto num_reduction_dims =
        output->domain()->nDims() - output->domain()->noReductions().size();
    NVF_ERROR(
        num_reduction_dims == 1,
        "Cannot support reducing multiple reduction axes ",
        expr->toString())
    int reduction_axis_after_permute =
        static_cast<int>(reduction_axis.value()) +
        num_reduction_axes_before_sharding_axis;
    auto* red_expr = dynamic_cast<ReductionOp*>(expr);
    output_permute = reductionOp(
        red_expr->getReductionOpType(),
        {reduction_axis_after_permute},
        red_expr->init(),
        input_permute);
  } else {
    output_permute = set(input_permute);
  }
  int sharding_axis_after_permute =
      sharding_axis - num_reduction_axes_before_sharding_axis;
  // Note this is a no-op and is moving a device parallel axis back
  TensorView* new_output =
      permute(output_permute, {{0, sharding_axis_after_permute}});
  ir_utils::replaceValInAllExprInputsAndFusionOutputs(output, new_output);

  output_permute->axis(0)->parallelize(sharded_id->getParallelType());
  new_output->axis(sharding_axis_after_permute)
      ->parallelize(sharded_id->getParallelType());
  // `output_permute` and `new_output` have inherited mesh from `input`. We
  // need to change them to `output`'s mesh so communication is only
  // between `input_permute` and `output_permute`.
  output_permute->setDeviceMesh(output->getDeviceMesh());
  new_output->setDeviceMesh(output->getDeviceMesh());

  // Set allocation domain for comm in/out
  input_permute->setAllocationDomain(input_permute->getLoopDomain(), true);
  output_permute->setAllocationDomain(output_permute->getLoopDomain(), true);
}

// Returns the set of parallel types seen on the loop domain of the given tvs.
std::unordered_set<ParallelType> getParallelTypesToPropagate(
    std::vector<TensorView*> tvs) {
  // Get the set of parallel types seen on the loop domain of the given tvs.
  std::unordered_set<ParallelType> existing_parallel_types;
  for (auto tv : tvs) {
    for (auto id : tv->getLoopDomain()) {
      if (!id->isReduction() && id->isDeviceDim()) {
        existing_parallel_types.insert(id->getParallelType());
      }
    }
  }
  std::unordered_set<ParallelType> selected_parallel_types;
  for (ParallelType pt : kParallelTypeDIDs) {
    if (!existing_parallel_types.count(pt)) {
      selected_parallel_types.insert(pt);
    }
  }
  return selected_parallel_types;
}


// Reorder the DID axis with the given parallel types to the front.
// Returns the number of device dimensions that were reordered to the front.
// This allows us to limit propagation to only the relevant DID axis.
int64_t selectiveReorderDIDToFront(
    TensorView* tv,
    const std::unordered_set<ParallelType>& selected_parallel_types) {
  std::unordered_map<int64_t, int64_t> old2new;
  int64_t current_pos = 0;

  for (auto&& [pos, id] : enumerate(tv->getLoopDomain())) {
    if (id->isDeviceDim() &&
        selected_parallel_types.count(id->getParallelType())) {
      old2new[pos] = current_pos;
      current_pos++;
    }
  }

  tv->reorder(old2new);
  return current_pos;
}

using PropagationDirection = scheduler_utils::PropagateDirection;
void propagateDIDTransform(
    TensorView* ref,
    const std::vector<TensorView*>& tvs,
    int64_t did_pos,
    PropagationDirection direction) {
  TensorDomain* replayed_domain = nullptr;
  std::unordered_set<ParallelType> selected_parallel_types =
          getParallelTypesToPropagate(tvs);

  // This restricts the transform propagation to only the relevant DID axis.
  did_pos =
      selectiveReorderDIDToFront(ref, selected_parallel_types);
  for (TensorView* tv : tvs) {
    if (direction == PropagationDirection::kForward) {
      replayed_domain = TransformReplay::replayCasP(tv, ref, did_pos).first;
    } else {
      replayed_domain = TransformReplay::replayPasC(tv, ref, did_pos).first;
    }
    tv->setLoopDomain(replayed_domain->loop());
  }
}

int64_t mappedAxisInConsumer(TensorView* producer, TensorView* consumer, IterDomain* p_logical_id) {
  const auto pairwise_map = PairwiseLogicalDomainMap(producer, consumer).mapProducerToConsumer();
  auto c_it = pairwise_map.find(p_logical_id);
  NVF_ERROR(c_it != pairwise_map.end(), p_logical_id->toString(), " not mapped to any ID in ", consumer->toString());
  return posInDomain(consumer->getLogicalDomain(), c_it->second);
} 

void handleForLoopSplit(Expr* expr, CommunicationInfo communication_info) {
  TensorView* input = expr->inputs().at(0)->as<TensorView>();
  TensorView* output = expr->outputs().at(0)->as<TensorView>();

  IterDomain* p_sharded_id = communication_info.p_sharded_id;
  IterDomain* c_sharded_id = communication_info.c_sharded_id;

  TensorView* comm_input = set(input);
  
  int64_t p_sharded_idx = mappedAxisInConsumer(input, comm_input, p_sharded_id);

  comm_input->setAllocationDomain(
      TensorDomain::orderedAs(
          comm_input->getLogicalDomain(),
          std::unordered_map<int64_t, int64_t>{{p_sharded_idx, 0}}),
          true);

  if (expr->isA<LoadStoreOp>()) {
    IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, output, comm_input);
  } else {
    IrBuilder::create<ReductionOp>(
        expr->as<ReductionOp>()->getReductionOpType(),
        expr->as<ReductionOp>()->init(),
        output,
        comm_input
    );
  }
  
  TensorView* comm_output = set(output);
  
  int64_t output_sharded_idx = posInDomain(output->getLogicalDomain(), c_sharded_id);
  output->setAllocationDomain(
      TensorDomain::orderedAs(
          output->getLogicalDomain(),
          std::unordered_map<int64_t, int64_t>{{output_sharded_idx, 0}}),
          true);
  
  int64_t c_sharded_idx = mappedAxisInConsumer(output, comm_output, c_sharded_id);
  // TODO: Revert this to output allocation domain instead.
  comm_output->setAllocationDomain(
      TensorDomain::orderedAs(
          comm_output->getLogicalDomain(),
          std::unordered_map<int64_t, int64_t>{{0, c_sharded_idx}}),
          true);

  ir_utils::replaceValInAllExprInputsAndFusionOutputs(output, comm_output);

  propagateDIDTransform(
      input, {comm_input}, -1, PropagationDirection::kForward);
  shardAllLike(input, {comm_input});
  
  propagateDIDTransform(
      output, {comm_output}, -1, PropagationDirection::kForward);
  shardAllLike(output, {comm_output});
}

} // namespace

void ReorderShardedAxisPass::runPass(Fusion* fusion) {
  FusionGuard fg(fusion);

  const std::vector<Expr*>& exprs = fusion->exprs();

  for (auto it = std::rbegin(exprs); it != std::rend(exprs); it++) {
    Expr* expr = *it;

    if (!(expr->isA<LoadStoreOp>() &&
          (expr->as<LoadStoreOp>()->opType() == LoadStoreOpType::Set)) &&
        !expr->isA<ReductionOp>()) {
      continue;
    }

    if (!isResharding(expr)) {
      continue;
    }

    if (isCommLayoutCompliant(expr)) {
      continue;
    }

    auto communication_info = getGatherOrScatterCommInfo(expr);
    NVF_ERROR(
        communication_info.has_value(),
        "Communication info not found for ",
        expr->toString());

    // if ((*communication_info).type == CommunicationType::Gather) {
    //   reorderForGatherBasedComm(expr, *communication_info);
    // } else {
    //   reorderForScatterBasedComm(expr, *communication_info);
    // }
    handleForLoopSplit(expr, *communication_info);
  }
}

} // namespace nvfuser::preseg_passes
