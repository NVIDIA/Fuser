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
  ParallelType pt;
};

std::vector<IterDomain*> getInputsInTargetDomain(
    IterDomain* loop_id,
    const std::vector<IterDomain*>& target_domain) {
  const std::vector<Val*> inputs_as_vals = IterVisitor::getInputsTo(
      {loop_id}, {target_domain.begin(), target_domain.end()});
  std::vector<IterDomain*> inputs_as_iter_domains;
  inputs_as_iter_domains.reserve(inputs_as_vals.size());
  std::transform(
      inputs_as_vals.begin(),
      inputs_as_vals.end(),
      std::back_inserter(inputs_as_iter_domains),
      [](Val* val) { return val->as<IterDomain>(); });
  return inputs_as_iter_domains;
}

IterDomain* getShardedLogicalId(TensorView* tv, ParallelType pt) {
  int64_t loop_axis = getShardedLoopAxis(tv, pt);
  if (loop_axis == -1) {
    return nullptr;
  }
  std::vector<IterDomain*> logical_ids = getInputsInTargetDomain(tv->axis(loop_axis), tv->getLogicalDomain());
  NVF_ERROR(logical_ids.size() == 1, "Expected exactly one logical ID");
  return logical_ids.at(0);
}

std::optional<CommunicationInfo> getGatherOrScatterLogicalId(Expr* expr) {
  TensorView* input = expr->inputs().at(0)->as<TensorView>();
  TensorView* output = expr->outputs().at(0)->as<TensorView>();
  bool has_sharding_change = false;
  std::optional<CommunicationInfo> communication_info = std::nullopt;
  
  for (ParallelType pt: kParallelTypeDIDs) {
    IterDomain* in_sharded_id = getShardedLogicalId(input, pt);
    IterDomain* out_sharded_id = getShardedLogicalId(output, pt);
    if (in_sharded_id == nullptr && out_sharded_id == nullptr) {
      // Not sharded on this parallel type
      continue;
    }
    bool in_sharded = in_sharded_id != nullptr;
    bool out_sharded = out_sharded_id != nullptr;

    if (expr->isA<LoadStoreOp>()) {
      if (in_sharded && !out_sharded) {
        // Gather / Allgather
        NVF_ERROR(!has_sharding_change, "Expected at most one sharding change");
        has_sharding_change = true;
        communication_info = CommunicationInfo{CommunicationType::Gather, in_sharded_id, pt};
      }
      if (!in_sharded && out_sharded) {
        // Scatter
        NVF_ERROR(!has_sharding_change, "Expected at most one sharding change");
        has_sharding_change = true;
        communication_info = CommunicationInfo{CommunicationType::Scatter, out_sharded_id, pt};
      }
    }
    if (expr->isA<ReductionOp>()) {
      if (!in_sharded || !out_sharded) {
        // Cannot be a reduce scatter communication.
        continue;
      }
      // Check if the in_sharded_axis is reduced in the output.
      const auto p2c_map = PairwiseLogicalDomainMap(input, output).mapProducerToConsumer();
      auto out_it = p2c_map.find(in_sharded_id);
      if (out_it == p2c_map.end()) {
        continue;
      }
      if (!out_it->second->isReduction()) {
        continue;
      }
      NVF_ERROR(!has_sharding_change, "Expected at most one sharding change");
      has_sharding_change = true;
      communication_info = CommunicationInfo{CommunicationType::ReduceScatter, out_sharded_id, pt};
    }
  }
  return communication_info;
}

int64_t allocationIndex(const std::vector<IterDomain*>& allocation_domain, IterDomain* logical_id) {
  // This sharded logical ID may not directly present in allocation domain.
  // This indicates allocation domain has DID transformations.
  // Find the derived loop IDs in the allocation domain and their index.
  auto transforms = DependencyCheck::getAllExprsBetween(
          {logical_id},
          {allocation_domain.begin(), allocation_domain.end()});
  std::unordered_set<IterDomain*> reachable_ids;
  // Add the logical id for the case where it is directly in the domain.
  reachable_ids.insert(logical_id);

  for (auto expr : transforms) {
    auto outputs = ir_utils::filterByType<IterDomain>(expr->outputs());
      reachable_ids.insert(outputs.begin(), outputs.end());
    }
  }
  for (IterDomain* alloc_id : reachable_ids) {
    if (alloc_id == id) {
      return index;
    }
    if (alloc_id->isDeviceDim() || alloc_id->isReduction() || alloc_id->isBroadcast()) {
      continue;
    }
    index++;
  }
  return -1;
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
  debug() << "Allocation index of " << communication_info.sharded_id->toString() << " in " << input->toString() << " is " << allocationIndex(domain, communication_info.sharded_id) << std::endl;
  return allocationIndex(domain, communication_info.sharded_id) == 0;
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
    return;
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

  int sharding_axis =
      static_cast<int>(output->domain()->rootPosOf(sharded_id));

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
      continue;
    }
    
    auto communication_info = getGatherOrScatterLogicalId(expr);
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
