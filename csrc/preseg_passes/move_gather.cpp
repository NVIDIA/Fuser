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
  // Iterate through all expressions in the fusion
  for (auto expr : fusion->exprs()) {
    // Check if the expression is a GatherOp
    if (expr->isA<GatherOp>()) {
      return true; // Found a GatherOp
    }
  }
  return false; // No GatherOp found
}

std::optional<Expr*> getProducerOfLookup(const GatherOp* gather_op) {
  // Get the producer of the GatherOp
  if (gather_op->lookupTv()->isFusionInput()) {
    return std::nullopt;
  }

  auto producer = gather_op->lookupTv()->definition();
  NVF_ERROR(
      producer != nullptr, "GatherOp must have a producer, but found none.");

  if (isUnaryPointwiseOp(producer)) {
    return producer;
  }

  if (producer->isA<SqueezeOp>()) {
    return producer;
  }

  return std::nullopt;
}

GatherOp* moveGatherOp(Fusion* fusion, GatherOp* gather_op, Expr* def) {
  IrCloner ir_cloner(fusion);

  auto is_take_along_axis = gather_op->exactSizes();

  auto unsqueeze_output = unsqueeze(gather_op->indexTv(), 0);

  auto index_tv =
      def->isA<SqueezeOp>() ? unsqueeze_output : gather_op->indexTv();

  auto dim_for_gather_op =
      def->isA<SqueezeOp>() ? gather_op->dim() + 1 : gather_op->dim();

  auto new_gather_op_output = is_take_along_axis
      ? takeAlongAxis(
            static_cast<TensorView*>(def->input(0)),
            index_tv,
            dim_for_gather_op)
      : gather(
            static_cast<TensorView*>(def->input(0)),
            dim_for_gather_op,
            index_tv);

  if (def->isA<SqueezeOp>()) {
    auto new_squeeze_output = squeeze(
        new_gather_op_output, def->as<SqueezeOp>()->getSqueezeDimFlags());

    for (auto expr : gather_op->output(0)->uses()) {
      ir_utils::replaceValInExprInputs(
          expr, gather_op->output(0), new_squeeze_output);
    }

  } else {
    auto cloned_def = static_cast<Expr*>(def->clone(&ir_cloner));
    cloned_def = ir_utils::replaceValInExprInputs(
        cloned_def, cloned_def->input(0), new_gather_op_output);

    auto cloned_output_tv = ops::newValLike(
        cloned_def->input(0)->as<TensorView>(), cloned_def->output(0)->dtype());

    cloned_def = ir_utils::transferDefinitionToNewOutputs(
        cloned_def, {cloned_output_tv});

    for (auto expr : gather_op->output(0)->uses()) {
      ir_utils::replaceValInExprInputs(
          expr, gather_op->output(0), cloned_def->output(0));
    }
  }

  fusion->removeExpr(gather_op);

  return new_gather_op_output->definition()->as<GatherOp>();
}

void moveGatherOp(Fusion* fusion, GatherOp* gather_op) {
  static int count = 0;
  do {
    auto producer = getProducerOfLookup(gather_op);
    if (!producer.has_value() || producer.value() == nullptr) {
      return;
    }

    auto def = producer.value();
    // Move the GatherOp to the producer
    std::cout << "trying to move across " << def->toString() << std::endl;
    gather_op = moveGatherOp(fusion, gather_op, def);
    std::cout << "After moving GatherOp: Roumd: " << count++ << std::endl;
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
  NVF_ERROR(gather_ops.size() == 1, "Only one GatherOp is supported.");
  moveGatherOp(fusion, gather_ops.vector()[0]);
}
} // namespace nvfuser::preseg_passes