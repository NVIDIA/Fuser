// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/reorder_sharded_axis.h>

#include <fusion.h>
#include <host_ir/lower_to_communication.h>
#include <ir/allocation_utils.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <ops/alias.h>
#include <scheduler/utils.h>
#include <transform_replay.h>

namespace nvfuser::preseg_passes {

namespace {

void makeCommunicationLayoutCompliant(Expr* expr) {
  auto* input = expr->inputs().at(0)->as<TensorView>();
  auto* output = expr->outputs().at(0)->as<TensorView>();

  CommunicationInfo communication_info = getCommunicationInfo(expr);
  IterDomain* p_sharded_id = communication_info.p_sharded_id;
  IterDomain* c_sharded_id = communication_info.c_sharded_id;

  Layout p_layout =
      getCommunicationLayout(input, communication_info.type, p_sharded_id);
  if (!isCompliantWith(*canonicalizeLayout(input), p_layout)) {
    TensorView* input_copy = set(input);
    TransformReplay::selfReplay(input->domain(), input_copy->domain());
    ir_utils::replaceValInExprInputs(expr, input, input_copy);
    p_layout = *mapInLayoutToOutRoot(p_layout, input, input_copy);
    input = input_copy;
  }
  input->setAllocationDomain(
      p_layout.allocation_domain(), p_layout.contiguity());

  Layout c_layout =
      getCommunicationLayout(output, communication_info.type, c_sharded_id);
  // When the output doesn't have a specified allocation, we can override it
  // with the communication layout. The same doesn't apply to the input because
  // (1) a fusion input with empty allocation is considered to have the
  // major-to-minor stride order, and (2) there was a bug in the reduction
  // scheduler.
  if (output->hasAllocation() &&
      // Unlike for input, `c_layout` is the actual and `output` is the
      // required. This is because `c_layout` is guaranteed by `p_layout` and
      // the UCC/NCCL call, and `output` is specified by the user. For example,
      // when `c_layout` and `output` have the same allocation order,
      // `c_layout` is contiguous, `output` is non-contiguous, we don't need an
      // extra copy -- a contiguous tensor can be used as non-contiguous.
      !isCompliantWith(c_layout, *canonicalizeLayout(output))) {
    TensorView* output_copy = set(output);
    TransformReplay::selfReplay(output->domain(), output_copy->domain());
    ir_utils::replaceValInAllExprInputsAndFusionOutputs(output, output_copy);
  }
  output->setAllocationDomain(
      c_layout.allocation_domain(), c_layout.contiguity());
}

} // namespace

void ReorderShardedAxisPass::runPass(Fusion* fusion) {
  FusionGuard fg(fusion);

  const std::vector<Expr*>& exprs = fusion->exprs();

  for (auto it = std::rbegin(exprs); it != std::rend(exprs); it++) {
    Expr* expr = *it;
    if (!isResharding(expr)) {
      continue;
    }

    makeCommunicationLayoutCompliant(expr);
  }

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << std::endl
            << "Fusion Transforms after " << name() << ":" << std::endl;
    fusion->printTransforms();
    debug() << std::endl;
  }
}

} // namespace nvfuser::preseg_passes
