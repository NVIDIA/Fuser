// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/unsafe_reduce.h>

#include <unordered_map>
#include <vector>

#include <ir/utils.h>

namespace nvfuser::preseg_passes {

enum class NanStatus {
  UNCHANGED,
  SQUELCHED,
  REPAIRED,
};

bool isMinMaxReduce(Expr* expr) {
  if (auto op = dynamic_cast<ReductionOp*>(expr)) {
    auto reduction_type = op->getReductionOpType();
    return reduction_type == BinaryOpType::Min ||
        reduction_type == BinaryOpType::Max;
  }
  return false;
}

void UnsafeReducePass::runPass(Fusion* fusion) {
  FusionGuard fusion_guard(fusion);

  for (Expr* reduceExpr : fusion->exprs()) {
    if (!isMinMaxReduce(reduceExpr)) {
      continue;
    }

    std::unordered_map<TensorView*, NanStatus> status;

    // Mark parent tv as repaired
    auto parentTv = reduceExpr->input(0)->as<TensorView>();
    status[parentTv] = NanStatus::REPAIRED;

    // Mark reduceExpr outputs as squelched
    auto candidateTv = reduceExpr->output(0)->as<TensorView>();
    status[candidateTv] = NanStatus::SQUELCHED;

    // Propagate NanStatus downstream
    auto traversal =
        StmtSort::getExprsBetween({reduceExpr->input(0)}, fusion->outputs());
    for (Expr* expr : traversal) {
      if (expr == reduceExpr) {
        // reduceExpr is the source of SQUELCHED status,
        // do not overwrite it here.
        continue;
      }

      bool anyRepair = false;
      bool anySquelch = false;

      auto input_tvs = ir_utils::filterByType<TensorView>(expr->inputs());
      for (TensorView* tv : input_tvs) {
        if (!status.count(tv)) {
          continue;
        }

        anyRepair |= status[tv] == NanStatus::REPAIRED;
        anySquelch |= status[tv] == NanStatus::SQUELCHED;
      }

      NanStatus propagate = NanStatus::UNCHANGED;
      if (anyRepair) {
        propagate = NanStatus::REPAIRED;
      } else if (anySquelch) {
        propagate = NanStatus::SQUELCHED;
      }

      auto output_tvs = ir_utils::filterByType<TensorView>(expr->outputs());
      for (TensorView* tv : output_tvs) {
        if (propagate != NanStatus::UNCHANGED) {
          status[tv] = propagate;
        }
      }
    }

    // Check whether any squelched status reached output nodes
    bool squelchedOutput = false;
    auto output_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
    for (TensorView* tv : output_tvs) {
      if (!status.count(tv)) {
        continue;
      }

      squelchedOutput |= status[tv] == NanStatus::SQUELCHED;
    }

    std::cout << "DEBUGPRINT: " << squelchedOutput << "\n";
  }

  return;
}

} // namespace nvfuser::preseg_passes
