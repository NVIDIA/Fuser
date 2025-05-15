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

void handleForLoopSplit(Expr* expr, CommunicationInfo communication_info) {
  TensorView* input = expr->inputs().at(0)->as<TensorView>();
  TensorView* output = expr->outputs().at(0)->as<TensorView>();

  IterDomain* p_sharded_id = communication_info.p_sharded_id;
  IterDomain* c_sharded_id = communication_info.c_sharded_id;
  int64_t reduction_axis = communication_info.reduction_axis;

  TensorView* comm_input = set(input);
  int64_t p_sharded_idx = posInDomain(input->getLogicalDomain(), p_sharded_id);
  comm_input->setAllocationDomain(
      TensorDomain::orderedAs(
          comm_input->getLogicalDomain(),
          std::unordered_map<int64_t, int64_t>{{p_sharded_idx, 0}}),
          true);

  TensorView* comm_output = nullptr;
  if (expr->isA<LoadStoreOp>()) {
    comm_output = set(comm_input);
  } else {
    comm_output = reductionOp(
        expr->as<ReductionOp>()->getReductionOpType(),
        {reduction_axis}, expr->as<ReductionOp>()->init(), comm_input);
  }
  int64_t c_sharded_idx = posInDomain(output->getLogicalDomain(), c_sharded_id);
  comm_output->setAllocationDomain(
      TensorDomain::orderedAs(
          comm_output->getLogicalDomain(),
          std::unordered_map<int64_t, int64_t>{{c_sharded_idx, 0}}),
          true);
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, output, comm_output);

  shardAllLike(input, {comm_input});
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
