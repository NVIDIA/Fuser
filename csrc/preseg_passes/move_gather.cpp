// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/move_gather.h>

#include <expr_simplifier.h>
#include <fusion.h>
#include <ir/builder.h>
#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <ops/indexing.h>
#include <ops/utils.h>
#include <scheduler/utils.h>
#include <transform_replay.h>

namespace nvfuser::preseg_passes {

namespace {

bool isUnaryPointwiseOp(const Expr* expr) {
  // Check if the expression is a UnaryOp and a pointwise operation
  return expr->isA<UnaryOp>() && ir_utils::isPointwiseTvOp(expr);
}

// For now we allow unary pointwise ops like cast/neg or a squeeze.
// We only consider squeeze which has a single dim squeezed.
std::optional<Expr*> getAllowedLookupTvDef(const GatherOp* gather_op) {
  if (gather_op->lookupTv()->isFusionInput()) {
    return std::nullopt;
  }

  auto producing_expr = gather_op->lookupTv()->definition();
  NVF_ERROR(
      producing_expr != nullptr, "Def for Lookup tensor view is nullptr.");

  if (isUnaryPointwiseOp(producing_expr)) {
    return producing_expr;
  }

  // We handle squeeze op where only one dim is squeezed.
  if (producing_expr->isA<SqueezeOp>()) {
    auto squeeze_op = producing_expr->as<SqueezeOp>();
    if (std::count(
            squeeze_op->getSqueezeDimFlags().begin(),
            squeeze_op->getSqueezeDimFlags().end(),
            true) == 1) {
      return producing_expr;
    }
  }

  return std::nullopt;
}

auto positionOfSqueezedDim(const std::vector<bool>& squeeze_dim_flags) {
  auto it = std::find(squeeze_dim_flags.begin(), squeeze_dim_flags.end(), true);
  return it != squeeze_dim_flags.end()
      ? std::distance(squeeze_dim_flags.begin(), it)
      : -1;
}

// If the def of lookUpTv is squeeze, and we're moving the squeeze to
// after the gather/take_along_axis, then we need to add an unsqueeze
// to the IndexTv
auto prepareIndexTv(Fusion* fusion, GatherOp* gather_op, Expr* def) {
  if (def->isA<SqueezeOp>()) {
    auto squeeze_dim_flags = def->as<SqueezeOp>()->getSqueezeDimFlags();
    auto dim = positionOfSqueezedDim(squeeze_dim_flags);
    return unsqueeze(gather_op->indexTv(), dim);
  }
  return gather_op->indexTv();
}

auto addPostGatherSqueeze(Fusion* fusion, GatherOp* new_gather, Expr* def) {
  auto new_squeeze_output = squeeze(
      new_gather->output(0)->as<TensorView>(),
      def->as<SqueezeOp>()->getSqueezeDimFlags());
  return new_squeeze_output;
}

auto addPostGatherOp(Fusion* fusion, GatherOp* new_gather, Expr* def) {
  IrCloner ir_cloner(fusion);

  auto cloned_def = static_cast<Expr*>(def->clone(&ir_cloner));

  // Replace the input with the output of the new gather.
  cloned_def = ir_utils::replaceValInExprInputs(
      cloned_def, cloned_def->input(0), new_gather->output(0));

  // Since it's a unary pointwise op, the output should be like the input
  // but type may vary as in the case of cast.
  auto cloned_output_tv = ops::newValLike(
      cloned_def->input(0)->as<TensorView>(), cloned_def->output(0)->dtype());
  cloned_def =
      ir_utils::transferDefinitionToNewOutputs(cloned_def, {cloned_output_tv});

  return cloned_def->output(0);
}

auto addOrCloneNewNodeAfterGather(
    Fusion* fusion,
    GatherOp* old_gather,
    GatherOp* new_gather,
    Expr* def) {
  // Create a new squeeze op or clone the unary pointwise op.
  auto output_of_post_gather_op = def->isA<SqueezeOp>()
      ? addPostGatherSqueeze(fusion, new_gather, def)
      : addPostGatherOp(fusion, new_gather, def);

  // Update the uses of the old gather.
  for (auto expr : old_gather->output(0)->uses()) {
    ir_utils::replaceValInExprInputs(
        expr, old_gather->output(0), output_of_post_gather_op);
  }
}

GatherOp* moveGatherOp(Fusion* fusion, GatherOp* gather_op, Expr* def) {
  if (def->isA<SqueezeOp>()) {
    // We don't handle the case where the squeezed dim was greater than the
    // take_along_axis dim
    if (positionOfSqueezedDim(def->as<SqueezeOp>()->getSqueezeDimFlags()) >
        gather_op->dim()) {
      return gather_op;
    }
  }

  // If the def for gather's lookup tv is a squeeze op
  // then as we move gather ahead of squeeze, we need to
  // add a unsqueeze to the index tv.
  auto index_tv = prepareIndexTv(fusion, gather_op, def);

  // If we are moving ahead of a squeeze then we need to update
  // the dim on which we are gathering. We restrict ourselves
  // to one squeezed dim which is at a lower pos than the gather dim.
  auto dim_for_gather_op =
      def->isA<SqueezeOp>() ? gather_op->dim() + 1 : gather_op->dim();

  // Create a new take_along_axis.
  auto new_gather_op_output = takeAlongAxis(
      static_cast<TensorView*>(def->input(0)), index_tv, dim_for_gather_op);

  // Add the def to after take_along_axis.
  addOrCloneNewNodeAfterGather(
      fusion,
      gather_op,
      new_gather_op_output->definition()->as<GatherOp>(),
      def);

  fusion->removeExpr(gather_op);

  return new_gather_op_output->definition()->as<GatherOp>();
}

// This function implements the following steps:
// take_along_axis has two inputs, lookupTv and indexTv
// We look at the def of lookupTv, say D.
// If D is a fusion input we stop.
// If D is a squeeze operation or if D is unary pointwise op such as cast of neg
// then we can move the take_along_axis before D. If D is any other type of Op
// we stop.
// If D is a suitable candidate which can moved after take_along_axis, we do the
// following steps:
// [Say the def of the input to D is the op E]
// 1. We create a new take_along_axis/Gather op. The input to the new Gather is
// the output of E.
// 2. We create a clone of D called D'. Set the input of D' as the output of the
// new Gather op. Update the uses of the old Gather op to use the output of D'
// 3. We delete the old Gather op.

// If D is a squeeze op (please not we currently support squeeze with a single
// dim being squeezed) we slightly modify the above algorithm. Let's say the def
// of the index Tv of the Gather Op is I. We add an unsqueeze op to output of I.
// The ouptut of the unsqueeze is then an input to the new
// Gather/take_along_axis.

// We do the above in a loop till the input to the Gather op is a suitable for
// reordering.
void moveGatherOp(Fusion* fusion, GatherOp* gather_op) {
  do {
    auto producer = getAllowedLookupTvDef(gather_op);
    if (!producer.has_value() ||
        (producer.has_value() && producer.value() == nullptr)) {
      return;
    }

    auto def = producer.value();
    auto new_gather_op = moveGatherOp(fusion, gather_op, def);
    if (new_gather_op == gather_op) {
      return;
    }
    gather_op = new_gather_op;
  } while (true);
}

} // namespace

void MoveGatherPass::runPass(Fusion* fusion) {
  auto gather_ops = ir_utils::getOpsOfType<GatherOp>(fusion);

  // no gather op found
  if (gather_ops.empty())
    return;

  // We support this optimization on fusions
  // with a single gather op for now.
  // TODO: support multiple take_along_axis ops.
  if (gather_ops.size() > 1) {
    return;
  }

  // For now we'll only support take_along_axis
  // not the general gather.
  if (!gather_ops[0]->exactSizes()) {
    return;
  }

  moveGatherOp(fusion, gather_ops[0]);
}
} // namespace nvfuser::preseg_passes
