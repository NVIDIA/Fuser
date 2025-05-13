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
  CommunicationType type; // Gather/Scatter/ReduceScatter
  IterDomain* sharded_id; // The sharded logical ID
  ParallelType pt; // The parallel type of the sharded ID
};

IterDomain* getShardedLogicalId(TensorView* tv, IterDomain* loop_id) {
  std::vector<IterDomain*> logical_ids = getInputsInTargetDomain(loop_id, tv->getLogicalDomain());
  NVF_ERROR(logical_ids.size() == 1, "Expected exactly one logical ID producing the device dimension ", loop_id->toString());
  return logical_ids.at(0);
}

std::optional<CommunicationInfo> getGatherOrScatterLogicalId(Expr* expr) {
  TensorView* input = expr->inputs().at(0)->as<TensorView>();
  TensorView* output = expr->outputs().at(0)->as<TensorView>();
  bool has_sharding_change = false;
  std::optional<CommunicationInfo> communication_info = std::nullopt;

  auto input_pt_to_did = mapDeviceParallelTypeToId(input->getLoopDomain());
  auto output_pt_to_did = mapDeviceParallelTypeToId(output->getLoopDomain());
  
  for (ParallelType pt: kParallelTypeDIDs) {
    IterDomain* input_loop_did = getOrDefault(input_pt_to_did, pt);
    IterDomain* output_loop_did = getOrDefault(output_pt_to_did, pt);

    if (input_loop_did == nullptr && output_loop_did == nullptr) {
      // Not sharded on this parallel type
      continue;
    }

    bool in_sharded = input_loop_did != nullptr;
    bool out_sharded = output_loop_did != nullptr;

    if (expr->isA<LoadStoreOp>()) {
      if (in_sharded && !out_sharded) {
        // Gather / Allgather
        NVF_ERROR(!has_sharding_change, "Expected at most one sharding change");
        has_sharding_change = true;
        communication_info = CommunicationInfo{CommunicationType::Gather, getShardedLogicalId(input, input_loop_did), pt};
      }
      if (!in_sharded && out_sharded) {
        // Scatter
        NVF_ERROR(!has_sharding_change, "Expected at most one sharding change");
        has_sharding_change = true;
        communication_info = CommunicationInfo{CommunicationType::Scatter, getShardedLogicalId(output, output_loop_did), pt};
      }
    }
    if (expr->isA<ReductionOp>()) {
      if (!in_sharded || !out_sharded) {
        // Cannot be a reduce scatter communication.
        continue;
      }
      // Check if the in_sharded_axis is reduced in the output.
      IterDomain* input_logical_did = getShardedLogicalId(input, input_loop_did);
      IterDomain* output_logical_did = getShardedLogicalId(output, output_loop_did);

      const auto p2c_map = PairwiseLogicalDomainMap(input, output).mapProducerToConsumer();
      auto out_it = p2c_map.find(input_logical_did);
      if (out_it == p2c_map.end()) {
        continue;
      }
      if (!out_it->second->isReduction()) {
        continue;
      }
      NVF_ERROR(!has_sharding_change, "Expected at most one sharding change");
      has_sharding_change = true;
      communication_info = CommunicationInfo{CommunicationType::ReduceScatter, output_logical_did, pt};
    }
  }
  return communication_info;
}

bool isAllocatedOutermost(const std::vector<IterDomain*>& allocation_domain, IterDomain* logical_id) {
  // This sharded logical ID may not directly present in allocation domain.
  // This indicates allocation domain has DID transformations.
  // Find the derived loop IDs in the allocation domain and their index.  
  auto transforms = DependencyCheck::getAllExprsBetween(
          {logical_id},
          {allocation_domain.begin(), allocation_domain.end()});
  std::vector<IterDomain*> derived_ids = {logical_id};
  scheduler_utils::applyTransforms(derived_ids, transforms);

  // The logical ID is the outermost dimension in the allocation domain if
  // the derived ids are present at the front of the allocation domain.
  // Example:
  // logical [i0, i1]
  // loop [DIDx(d), i0, i1/d]
  // allocation [DIDx(d), i0, i1/d]
  // isAllocatedOutermost(allocation_domain, i1) == false
  // isAllocatedOutermost(allocation_domain, i0) == true

  auto no_reductions_allocation = TensorDomain::noReductions(allocation_domain);

  // Check if derived_ids appear at the front.
  size_t derived_idx = 0;
  for (IterDomain* alloc_id : no_reductions_allocation) {
    if (derived_idx >= derived_ids.size()) {
      break;
    }
    if (alloc_id == derived_ids.at(derived_idx)) {
      derived_idx++;
    } 
    if (alloc_id->isDeviceDim()) {
      continue;
    }
    return false;
  }
  NVF_ERROR(derived_idx == derived_ids.size(), "Some derived ids ", derived_ids, " not found in allocation domain ", no_reductions_allocation);
  return true;
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
  return isAllocatedOutermost(domain, communication_info.sharded_id);
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

} 

void handleGather(Expr* expr, CommunicationInfo communication_info) {
// sharded_id is sharded in input,  mapped id not sharded in output.
// 
}

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
    debug() << "Communication info: " << communication_info.value().type << " " << communication_info.value().sharded_id->toString() << " " << communication_info.value().pt << std::endl;
    if (isProcessGroupCompliant(expr, communication_info.value())) {
      continue;
    }

    reorderForLogicalSplit(expr, communication_info.value());
  }
}

} // namespace nvfuser::preseg_passes
