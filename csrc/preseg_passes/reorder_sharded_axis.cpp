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

int64_t mappedAxisInConsumer(
    TensorView* producer,
    TensorView* consumer,
    IterDomain* p_logical_id) {
  const auto pairwise_map =
      PairwiseLogicalDomainMap(producer, consumer).mapProducerToConsumer();
  auto c_it = pairwise_map.find(p_logical_id);
  NVF_ERROR(
      c_it != pairwise_map.end(),
      p_logical_id->toString(),
      " not mapped to any ID in ",
      consumer->toString());
  return posInDomain(consumer->getLogicalDomain(), c_it->second);
}

void makeCommunicationLayoutCompliant(
    Expr* expr,
    CommunicationInfo communication_info) {
  TensorView* input = expr->inputs().at(0)->as<TensorView>();
  TensorView* output = expr->outputs().at(0)->as<TensorView>();

  IterDomain* p_sharded_id = communication_info.p_sharded_id;
  IterDomain* c_sharded_id = communication_info.c_sharded_id;

  // Copy input into required memory layout.
  // Note: If input is not a fusion input, we should ideally be able to specify
  // allocation domain of input instead. However, some schedulers (e.g.
  // reduction) may not support this, so creating a copy here.
  TensorView* comm_input = set(input);
  int64_t comm_input_device_idx =
      mappedAxisInConsumer(input, comm_input, p_sharded_id);
  comm_input->setAllocationDomain(
      TensorDomain::orderedAs(
          comm_input->getLogicalDomain(),
          std::unordered_map<int64_t, int64_t>{{comm_input_device_idx, 0}}),
      true);

  // Communication expression.
  // We do not create a new output tensorview for the communication.
  // Instead, we replace the input of the communication with the comm_input
  // in the communication expression. This preserves the sharding annotations
  // of the original communication output.
  if (expr->isA<LoadStoreOp>()) {
    IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, output, comm_input);
  } else {
    IrBuilder::create<ReductionOp>(
        expr->as<ReductionOp>()->getReductionOpType(),
        expr->as<ReductionOp>()->init(),
        output,
        comm_input);
  }
  int64_t output_device_idx =
      posInDomain(output->getLogicalDomain(), c_sharded_id);
  output->setAllocationDomain(
      TensorDomain::orderedAs(
          output->getLogicalDomain(),
          std::unordered_map<int64_t, int64_t>{{output_device_idx, 0}}),
      true);

  // TODO: Make this conditional on whether the output is a fusion output.
  // and has allocation domain.
  TensorView* comm_output = set(output);

  int64_t c_sharded_idx =
      mappedAxisInConsumer(output, comm_output, c_sharded_id);
  // TODO: Revert this to output allocation domain instead.
  comm_output->setAllocationDomain(
      TensorDomain::orderedAs(
          comm_output->getLogicalDomain(),
          std::unordered_map<int64_t, int64_t>{{0, c_sharded_idx}}),
      true);

  ir_utils::replaceValInAllExprInputsAndFusionOutputs(output, comm_output);
  
  transform_and_parallelize_like(input, comm_input);
  transform_and_parallelize_like(output, comm_output);
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
