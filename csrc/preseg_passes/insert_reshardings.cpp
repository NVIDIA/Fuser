// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/insert_reshardings.h>

#include <device_lower/utils.h>
#include <fusion.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/utils.h>
#include <multidevice/lower_communication.h>
#include <multidevice/utils.h>
#include <ops/alias.h>

namespace nvfuser::preseg_passes {
namespace {
// TODO: We can either reshard the inputs of a resharding expression or
// the outputs. Currently, we reshard the outputs when there is only one
// input, otherwise we reshard the inputs. This heuristic should be smarter
// and attempt to minimize communication.
// We do no support resharding multi-output expressions. Fusions may contain
// multi-output expressions if they don't require resharding.
bool shouldReshardAfter(Expr* expr) {
  return expr->inputs().size() == 1 && expr->outputs().size() == 1;
}

void insertReshardingsBefore(Fusion* fusion) {
  // Remove this after we refactor this as a pre-segmenter pass.
  FusionGuard fg(fusion);
  for (auto expr : fusion->exprs()) {
    if (isLowerableToCommunication(expr) || shouldReshardAfter(expr)) {
      continue;
    }

    // Verify that multi-output expression requires no resharding.
    if (expr->outputs().size() > 1) {
      for (auto output : ir_utils::filterByType<TensorView>(expr->outputs())) {
        for (auto input : ir_utils::filterByType<TensorView>(expr->inputs())) {
          NVF_CHECK(
              !haveDifferentShardings(input, output),
              "Cannot handle resharding a multi-output expression ",
              expr->toString());
        }
      }
      continue;
    }

    if (!expr->output(0)->isA<TensorView>()) {
      continue;
    }
    auto output = expr->output(0)->as<TensorView>();

    std::unordered_set<TensorView*> inputs;
    for (auto input : ir_utils::filterByType<TensorView>(expr->inputs())) {
      if (haveDifferentShardings(input, output)) {
        inputs.insert(input);
      }
    }

    // Reshard each input of expr to match output if necessary
    std::vector<TensorView*> new_inputs;
    for (auto input : inputs) {
      // TODO: reuse cacheAfter?
      // TODO: here we should add a mechanism to potentially reuse the
      // inserted resharding accross all the consumer of the resharded tensor.
      // This way we could avoid wasteful resharding set insertion.
      TensorView* new_input = set(input);
      new_inputs.push_back(new_input);
      expr = ir_utils::replaceValInExprInputs(expr, input, new_input);
    }
    shardAllLike(output, new_inputs);
  }
}

void insertReshardingsAfter(Fusion* fusion) {
  // Remove this after we refactor this as a pre-segmenter pass.
  FusionGuard fg(fusion);
  // Iterate backwards over fusion expressions. Reshard after will
  // replace expressions that occur downstream from the current expression.
  // This will ensure we don't process an expression that has been deleted.
  auto exprs = fusion->exprs();
  for (auto it = std::rbegin(exprs); it != std::rend(exprs); it++) {
    Expr* expr = *it;
    if (isLowerableToCommunication(expr) || !shouldReshardAfter(expr)) {
      continue;
    }

    if (!expr->output(0)->isA<TensorView>()) {
      continue;
    }
    auto output = expr->output(0)->as<TensorView>();

    std::unordered_set<TensorView*> inputs;
    for (auto input : ir_utils::filterByType<TensorView>(expr->inputs())) {
      if (haveDifferentShardings(input, output)) {
        inputs.insert(input);
      }
    }

    // Insert resharding set after the expr and update
    // output of expr to match input's sharding.
    // input [expr] output [set] new_output
    if (!inputs.empty()) {
      TensorView* input = *inputs.begin();
      TensorView* new_output = set(output);
      ir_utils::replaceValInAllExprInputsAndFusionOutputs(output, new_output);
      // Update shardings new_output takes output's sharding,
      // output takes input's sharding
      shardAllLike(output, {new_output});
      shardAllLike(input, {output});
    }
  }
}
} // namespace

void InsertReshardingsPass::runPass(Fusion* fusion) {
  // shouldReshardAfter selects whether insertReshardingsAfter or
  // insertReshardingsBefore is used.
  insertReshardingsAfter(fusion);
  insertReshardingsBefore(fusion);
}

} // namespace nvfuser::preseg_passes
