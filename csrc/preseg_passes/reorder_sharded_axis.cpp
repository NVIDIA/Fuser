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
#include <ir/allocation_utils.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <scheduler/utils.h>
#include <transform_replay.h>

namespace nvfuser::preseg_passes {

namespace {

void transform_and_parallelize_like(TensorView* ref, TensorView* tv) {
  auto old2new = reorderDIDToFront(ref);
  propagateDIDTransform(
      ref, {tv}, /*did_pos=*/old2new.size(), PropagateDirection::kForward);
  shardAllLike(ref, {tv}, {kParallelTypeDIDs.begin(), kParallelTypeDIDs.end()});
}

void makeCommunicationLayoutCompliant(
    Expr* expr,
    CommunicationInfo communication_info) {
  TensorView* input = expr->inputs().at(0)->as<TensorView>();
  TensorView* output = expr->outputs().at(0)->as<TensorView>();

  IterDomain* p_sharded_id = communication_info.p_sharded_id;
  IterDomain* c_sharded_id = communication_info.c_sharded_id;

  TensorView* input_copy = nullptr;

  if (!p_sharded_id->isDeviceDim() &&
      !isAllocatedOutermost(input, p_sharded_id)) {
    // Copy input into required memory layout.
    // Note: If input is not a fusion input, we should ideally be able to
    // specify allocation domain of input instead. However, some schedulers
    // (e.g. reduction) may not support this, so creating a copy here.

    input_copy = set(input);

    // Find index of p_sharded_id in input_copy's logical domain.
    // Reduction axis in input can change the position of p_sharded_id in
    // input_copy's logical domain.
    auto p2c = PairwiseLogicalDomainMap(input, input_copy)
                   .mapBroadcast(true)
                   .mapSymbolic(true)
                   .mapProducerToConsumer();

    IterDomain* input_copy_sharded_id = p2c.at(p_sharded_id);
    int64_t input_copy_sharded_idx = posInDomain(
        input_copy->getMaybeAllocationDomain(), input_copy_sharded_id);
    input_copy->setAllocationDomain(
        TensorDomain::orderedAs(
            input_copy->getMaybeAllocationDomain(),
            std::unordered_map<int64_t, int64_t>{{input_copy_sharded_idx, 0}}),
        true);

    // New communication expression.
    // We do not create a new output tensorview for the communication.
    // Instead, we replace the input of the communication with the input_copy
    // in the communication expression. This preserves the sharding annotations
    // of the original communication output.
    if (expr->isA<LoadStoreOp>()) {
      IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, output, input_copy);
    } else {
      IrBuilder::create<ReductionOp>(
          expr->as<ReductionOp>()->getReductionOpType(),
          expr->as<ReductionOp>()->init(),
          output,
          input_copy);
    }
    transform_and_parallelize_like(input, input_copy);
  }

  std::optional<Layout> output_layout = canonicalizeLayout(output);

  NVF_ERROR(
      output_layout.has_value(),
      "Cannot canonicalize layout for ",
      output->domain()->toString(0, false));
  int64_t output_sharded_idx =
      posInDomain((*output_layout).allocation_domain, c_sharded_id);

  // If the output is a fusion output and has allocation domain,
  // and the c_sharded_id is not allocated outermost,
  // create a copy of the output to revert the allocation domain.
  if (output->isFusionOutput() && output->hasAllocation() &&
      !isAllocatedOutermost(output, c_sharded_id)) {
    TensorView* output_copy = set(output);

    auto p2c_map = PairwiseLogicalDomainMap(output, output_copy)
                       .mapBroadcast(true)
                       .mapSymbolic(true)
                       .mapProducerToConsumer();
    std::vector<IterDomain*> output_copy_allocation_domain;

    for (IterDomain* output_id :
         TensorDomain::noReductions((*output_layout).allocation_domain)) {
      IterDomain* output_copy_id = p2c_map.at(output_id);
      output_copy_allocation_domain.push_back(output_copy_id);
    }

    output_copy->setAllocationDomain(output_copy_allocation_domain, true);
    ir_utils::replaceValInAllExprInputsAndFusionOutputs(output, output_copy);
    transform_and_parallelize_like(output, output_copy);
  }

  output->setAllocationDomain(
      TensorDomain::orderedAs(
          (*output_layout).allocation_domain, {{output_sharded_idx, 0}}),
      true);
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

    makeCommunicationLayoutCompliant(expr, *communication_info);
  }
}

} // namespace nvfuser::preseg_passes
