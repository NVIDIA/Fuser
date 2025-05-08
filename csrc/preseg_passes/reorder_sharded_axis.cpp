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
#include <id_model/id_model.h>
#include <transform_replay.h>

namespace nvfuser::preseg_passes {

namespace {

struct CommunicationInfo {
  CommunicationType type;
  IterDomain* sharded_id;
};

std::optional<CommunicationInfo> getGatherOrScatterLogicalAxis(Expr* expr) {
  TensorView* input = expr->inputs().at(0)->as<TensorView>();
  TensorView* output = expr->outputs().at(0)->as<TensorView>();
  bool has_sharding_change = false;
  std::optional<CommunicationInfo> communication_info = std::nullopt;
  for (ParallelType pt: kParallelTypeDIDs) {
    int64_t in_sharded_axis = getShardedLogicalAxisFromDomain(input, pt, input->getLoopDomain());
    int64_t out_sharded_axis = getShardedLogicalAxisFromDomain(output, pt, output->getLoopDomain());
    if (in_sharded_axis == -1 && out_sharded_axis == -1) {
      // Not sharded on this parallel type
      continue;
    }
    bool in_sharded = in_sharded_axis != -1;
    bool out_sharded = out_sharded_axis != -1;
    const auto pairwise_map = PairwiseLogicalDomainMap(input, output);
    const auto p2c_map = pairwise_map.mapProducerToConsumer();
    const auto c2p_map = pairwise_map.mapConsumerToProducer();

    if (expr->isA<LoadStoreOp>()) {
      if (in_sharded && !out_sharded) {
        // Gather / Allgather
        NVF_ERROR(!has_sharding_change, "Expected at most one sharding change");
        has_sharding_change = true;

        IterDomain* input_sharded_id = input->getLogicalDomain().at(in_sharded_axis);
        communication_info = CommunicationInfo{CommunicationType::Gather, input_sharded_id};
      }

      if (!in_sharded && out_sharded) {
        // Scatter
        NVF_ERROR(!has_sharding_change, "Expected at most one sharding change");
        has_sharding_change = true;

        IterDomain* output_sharded_id = output->getLogicalDomain().at(out_sharded_axis);
        communication_info = CommunicationInfo{CommunicationType::Scatter, output_sharded_id};
      }
    }

    if (expr->isA<ReductionOp>()) {
      if (!in_sharded || !out_sharded) {
        // Cannot be a reduce scatter communication.
        continue;
      }
      // Check if the in_sharded_axis is reduced in the output.
      auto out_it = p2c_map.find(input->getLogicalDomain().at(in_sharded_axis));
      if (out_it == p2c_map.end()) {
        continue;
      }
      if (!out_it->second->isReduction()) {
        continue;
      }
      NVF_ERROR(!has_sharding_change, "Expected at most one sharding change");
      has_sharding_change = true;

      IterDomain* output_sharded_id = output->getLogicalDomain().at(out_sharded_axis);
      communication_info = CommunicationInfo{CommunicationType::ReduceScatter, output_sharded_id};
    }
  }
  return communication_info;
}

bool isProcessGroupCompliant(Expr* expr, CommunicationInfo communication_info) {
  TensorView* input = expr->inputs().at(0)->as<TensorView>();
  TensorView* output = expr->outputs().at(0)->as<TensorView>();
  std::vector<IterDomain*> domain;
  if (communication_info.type == CommunicationType::Gather) {
    domain = input->getMaybeAllocationDomain();
  } else {
    domain = output->getMaybeAllocationDomain();
  }
  return allocationIndex(input, communication_info.sharded_id) == 0;
}
}

void reorderForLogicalSplit(Expr* expr, CommunicationInfo communication_info) {
  TensorView* input = expr->inputs().at(0)->as<TensorView>();
  TensorView* output = expr->outputs().at(0)->as<TensorView>();
  IterDomain* sharded_id = communication_info.sharded_id;
  // For gather operations i.e. ID goes from sharded to unsharded
  // this will rematerialize a sharded axis.
  // ProcessGroup expects contiguous tensors.
  // Update input to push the rematerialized axis to the front -> collective
  // -> permute the rematerialized axis to the proper location
  // Example: [i0 DIDx(i1)] -> [i0 i1]
  // Rewritten to: [i0 DIDx(i1)] -> [DIDx(i1) i0] -> [i1 i0] -> [i0 i1]
  // Note: there are no reduction based collectives that
  // materializes an axis so expr is guaranteed to be a set.
  if (communication_info.type == CommunicationType::Gather) {
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
      output_permute->setAllocationDomain(
          output_permute->getLoopDomain(), true);
    }
    // For scatter operations i.e. ID goes from unsharded to sharded
    // Update input to push the scattered axis to the front -> collective ->
    // permute the sharded axis to the proper location.
    // Scatter example: [i0 i1] -> [i0 DIDx(i1)]
    // Rewritten to [i0 i1] -> [i1 i0] -> [DIDx(i1) i0] -> [i0 DIDx(i1)]
    // Reduce Scatter example: [i0 DIDx(i1) i2] -> [i0 r1 DIDx(i2)]
    // Rewritten to: [i0 DIDx(i1) i2] -> [i2 i0 DIDx(i1)] ->
    //                    [DIDx(i2) i0 r1] -> [i0 DIDx(i2)]
    // Note that reduction axis shifts from axis=1 to axis=2.
    else if (!shard_additions.empty() && isInnerResharding(expr)) {
      IterDomain* shard_added_id = shard_additions[0];
      int sharding_axis =
          static_cast<int>(output->domain()->rootPosOf(shard_added_id));

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
} // namespace

void ReorderShardedAxisPass::runPass(Fusion* fusion) {
  FusionGuard fg(fusion);

  const std::vector<Expr*>& exprs = fusion->exprs();

  for (auto it = std::rbegin(exprs); it != std::rend(exprs); it++) {
    Expr* expr = *it;
    if (!(expr->isA<LoadStoreOp>() 
      && (expr->as<LoadStoreOp>()->opType() == LoadStoreOpType::Set)) 
      && !expr->isA<ReductionOp>()) {
      continue;
    }
    if (!isResharding(expr)) {

      // Set allocation domain for comm in/out
      input_permute->setAllocationDomain(input_permute->getLoopDomain(), true);
      output_permute->setAllocationDomain(
          output_permute->getLoopDomain(), true);
    }
    
    auto communication_info = getGatherOrScatterLogicalAxis(expr);
    if (!communication_info.has_value()) {
      continue;
    }

    if (isProcessGroupCompliant(expr, communication_info.value())) {
      continue;
    }

    reorderForLogicalSplit(expr, communication_info.value());
  }
}

} // namespace nvfuser::preseg_passes
