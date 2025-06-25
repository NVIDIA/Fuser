// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <id_model/id_model.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <options.h>
#include <preseg_passes/reuse_expensive_computation_results.h>
namespace nvfuser::preseg_passes {
const std::unordered_set<UnaryOpType> expensive_unary_ops{
    UnaryOpType::Exp,
    UnaryOpType::Tanh,
    UnaryOpType::Reciprocal,
    UnaryOpType::Rsqrt,
    UnaryOpType::Log,
    UnaryOpType::Log10,
    UnaryOpType::Log2,
    UnaryOpType::Sin,
    UnaryOpType::Cos};

// Fusion before this pass:
// T1 = ExpensiveOps(T0);
// T2 = ExpensiveOps(T0);
// T3 = T1 + T2;
// Fusion after this pass:
// T2 = ExpensiveOps(T0);
// T3 = T2 + T2;
void ReuseExpensiveComputationResultsPass::runPass(Fusion* fusion) {
  // map from UnaryOpType to exprs using this uop
  std::unordered_map<UnaryOpType, std::vector<UnaryOp*>> uop_map;
  auto exprs = fusion->exprs();
  for (auto iter = exprs.rbegin(); iter != exprs.rend(); ++iter) {
    auto expr = *iter;
    // skip non-expensive uop
    auto uop = dynamic_cast<UnaryOp*>(expr);
    if (!uop) {
      continue;
    }
    auto uop_type = uop->getUnaryOpType();
    if (!expensive_unary_ops.contains(uop_type)) {
      continue;
    }
    // Add to this uop op to map if there is no entry for it
    if (!uop_map.contains(uop_type)) {
      uop_map[uop_type] = {uop};
      continue;
    }
    // Check all the entries of this uop op type
    // If the input is the same, replace the output
    bool is_found = false;
    const auto& visited_uops = uop_map[uop_type];
    for (auto visited_uop : visited_uops) {
      if (uop->in() == visited_uop->in()) {
        ir_utils::replaceValInAllExprInputsAndFusionOutputs(
            uop->out(), visited_uop->out());
        is_found = true;
        break;
      }
    }
    // If not found, add this uop op to the map
    if (!is_found) {
      uop_map[uop_type].push_back(uop);
    }
  }
}

} // namespace nvfuser::preseg_passes
