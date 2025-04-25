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
#include <transform_replay.h>

namespace nvfuser::preseg_passes {

namespace {

bool isUnaryPointwiseOp(const Expr* expr) {
  // Check if the expression is a UnaryOp and a pointwise operation
  return expr->isA<UnaryOp>() && ir_utils::isPointwiseTvOp(expr);
}

bool hasGatherOp(const Fusion* fusion) {
  auto exprs = fusion->exprs();
  return ir_utils::filterByType<GatherOp>(exprs).size() > 0;
}

std::optional<Expr*> getLookupTvDef(const GatherOp* gather_op) {
  if (gather_op->lookupTv()->isFusionInput()) {
    return std::nullopt;
  }

  auto producer = gather_op->lookupTv()->definition();
  NVF_ERROR(producer != nullptr, "Def for Lookup tensor view is nullptr.");

  if (isUnaryPointwiseOp(producer) || producer->isA<SqueezeOp>()) {
    return producer;
  }

  return std::nullopt;
}

auto prepareIndexTv(Fusion* fusion, GatherOp* gather_op, Expr* def) {
  if (def->isA<SqueezeOp>()) {
    return unsqueeze(gather_op->indexTv(), 0);
  }
  return gather_op->indexTv();
}

auto addPostGatherSqueeze(
    Fusion* fusion,
    GatherOp* gather,
    TensorView* new_gather_output,
    Expr* def,
    Expr* dest_expr) {
  auto new_squeeze_output =
      squeeze(new_gather_output, def->as<SqueezeOp>()->getSqueezeDimFlags());
  dest_expr = new_gather_output->definition();
  return new_squeeze_output;
}

auto addPostGatherOp(
    Fusion* fusion,
    GatherOp* gather,
    TensorView* new_gather_output,
    Expr* def,
    Expr* dest_expr) {
  IrCloner ir_cloner(fusion);
  auto cloned_def = static_cast<Expr*>(def->clone(&ir_cloner));

  cloned_def = ir_utils::replaceValInExprInputs(
      cloned_def, cloned_def->input(0), dest_expr->output(0));

  auto cloned_output_tv = ops::newValLike(
      cloned_def->input(0)->as<TensorView>(), cloned_def->output(0)->dtype());

  cloned_def =
      ir_utils::transferDefinitionToNewOutputs(cloned_def, {cloned_output_tv});

  return cloned_def->output(0);
}

auto addOrCloneNewNodeAfterGather(
    Fusion* fusion,
    GatherOp* gather,
    TensorView* new_gather_output,
    Expr* def,
    Expr* dest_expr) {
  auto output_of_post_gather_op = def->isA<SqueezeOp>()
      ? addPostGatherSqueeze(fusion, gather, new_gather_output, def, dest_expr)
      : addPostGatherOp(fusion, gather, new_gather_output, def, dest_expr);

  for (auto expr : gather->output(0)->uses()) {
    ir_utils::replaceValInExprInputs(
        expr, gather->output(0), output_of_post_gather_op);
  }
}

GatherOp* moveGatherOp(
    Fusion* fusion,
    GatherOp* gather_op,
    Expr* def,
    Expr* dest_expr) {
  IrCloner ir_cloner(fusion);

  auto index_tv = prepareIndexTv(fusion, gather_op, def);

  auto dim_for_gather_op =
      def->isA<SqueezeOp>() ? gather_op->dim() + 1 : gather_op->dim();

  auto new_gather_op_output = takeAlongAxis(
      static_cast<TensorView*>(def->input(0)), index_tv, dim_for_gather_op);

  addOrCloneNewNodeAfterGather(
      fusion, gather_op, new_gather_op_output, def, dest_expr);

  fusion->removeExpr(gather_op);

  return new_gather_op_output->definition()->as<GatherOp>();
}

void moveGatherOp(Fusion* fusion, GatherOp* gather_op) {
  Expr* dest_expr = gather_op;
  do {
    auto producer = getLookupTvDef(gather_op);
    if (!producer.has_value() || producer.value() == nullptr) {
      return;
    }

    auto def = producer.value();
    auto new_gather_op = moveGatherOp(fusion, gather_op, def, dest_expr);
    if (new_gather_op == gather_op) {
      return;
    }
    gather_op = new_gather_op;
    fusion->printMath();
  } while (true);
}

} // namespace

void MoveGatherPass::runPass(Fusion* fusion) {
  std::cout << "Running MoveGatherPass" << std::endl;
  fusion->printMath();
  if (!hasGatherOp(fusion))
    return;

  auto exprs = fusion->exprs();
  auto gather_ops = ir_utils::filterByType<GatherOp>(exprs);

  // We support this optimization on fusions
  // with a single gather op
  if (gather_ops.size() > 1) {
    return;
  }

  // For now we'll only support take_along_axis
  // not the general gather.
  if (!gather_ops.vector().at(0)->exactSizes()) {
    return;
  }

  moveGatherOp(fusion, gather_ops.vector()[0]);
  fusion->printMath();
}
} // namespace nvfuser::preseg_passes