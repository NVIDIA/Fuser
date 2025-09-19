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
  NONE,
  BAD_REDUCED,
  BAD_BROADCASTED,
  GOOD_DEFAULT,
  GOOD_REDUCED,
  GOOD_BROADCASTED,
};

using NanStatusMap = std::unordered_map<TensorView*, NanStatus>;

bool AnyInputStatus(
    Expr* expr,
    const NanStatusMap& statusMap,
    NanStatus status) {
  auto input_tvs = ir_utils::filterByType<TensorView>(expr->inputs());
  for (TensorView* tv : input_tvs) {
    auto it = statusMap.find(tv);
    if (it != statusMap.end() && it->second == status) {
      return true;
    }
  }

  return false;
}

static std::vector<int64_t> GetReductionAxes(ReductionOp* rop) {
  auto out = rop->output(0)->as<TensorView>();
  std::vector<int64_t> reduction_axes;
  int64_t pos = -1;
  for (IterDomain* out_id : out->getLogicalDomain()) {
    ++pos;
    if (out_id->isReduction()) {
      reduction_axes.push_back(pos);
    }
  }
  return reduction_axes;
}

bool AreReducingOverSameAxes(ReductionOp* rop1, ReductionOp* rop2) {
  std::vector<int64_t> rop1Axes = GetReductionAxes(rop1);
  std::vector<int64_t> rop2Axes = GetReductionAxes(rop2);

  return std::ranges::equal(rop1Axes, rop2Axes);
}

bool AnalyzeMinMaxOp(ReductionOp* targetRop) {
  Fusion* fusion = targetRop->fusion();

  NanStatusMap statusMap;

  // Mark parent tv as repaired
  auto parentTv = targetRop->input(0)->as<TensorView>();
  statusMap[parentTv] = NanStatus::GOOD_DEFAULT;

  // Mark targetRop outputs as squelched
  auto candidateTv = targetRop->output(0)->as<TensorView>();
  statusMap[candidateTv] = NanStatus::BAD_REDUCED;

  // Propagate NanStatus downstream
  auto traversal =
      StmtSort::getExprsBetween({targetRop->input(0)}, fusion->outputs());
  for (Expr* expr : traversal) {
    if (expr == targetRop) {
      // Skip the target rop. We already marked its status.
      continue;
    }

    bool anyGood = AnyInputStatus(expr, statusMap, NanStatus::GOOD_REDUCED);
    bool anyBad = AnyInputStatus(expr, statusMap, NanStatus::BAD_REDUCED);

    NanStatus currStatus = NanStatus::NONE;
    if (anyGood) {
      currStatus = NanStatus::GOOD_REDUCED;
    } else if (anyBad) {
      currStatus = NanStatus::BAD_REDUCED;
    } else if (AnyInputStatus(expr, statusMap, NanStatus::GOOD_DEFAULT)) {
      currStatus = NanStatus::GOOD_DEFAULT;
    }

    if (auto currRop = dynamic_cast<ReductionOp*>(expr); currRop) {
      auto reduction_type = targetRop->getReductionOpType();
      if (reduction_type == BinaryOpType::Min &&
          reduction_type == BinaryOpType::Max) {
        return false;
      }

      if (currStatus == NanStatus::GOOD_REDUCED ||
          currStatus == NanStatus::GOOD_BROADCASTED) {
        return false;
      }

      if (!AreReducingOverSameAxes(targetRop, currRop)) {
        return false;
      }

      if (currStatus == NanStatus::GOOD_DEFAULT) {
        currStatus = NanStatus::GOOD_REDUCED;
      }
    }

    // Propagate NanStatus for this expr to all outputs.
    auto output_tvs = ir_utils::filterByType<TensorView>(expr->outputs());
    for (TensorView* tv : output_tvs) {
      statusMap[tv] = currStatus;
    }
  }

  // Check whether any bad status reached output nodes
  auto output_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
  for (TensorView* tv : output_tvs) {
    auto it = statusMap.find(tv);
    if (it == statusMap.end()) {
      continue;
    }

    if (it->second == NanStatus::BAD_REDUCED) {
      return false;
    }
  }

  return true;
}

void UnsafeReducePass::runPass(Fusion* fusion) {
  FusionGuard fusion_guard(fusion);

  // The outer loop runs over all expressions, filtering out most of them.
  // It stops only on min/max reductions, which become the target for the rest
  // of the analysis.
  //
  // Once we identify a target unsafe reduction, we perform one downward pass
  // start from the unsafe's direct input. The pass propagates NanStatus
  // information. We only allow an unsafe reduce to pair with a single safe
  // reduction. The safe reduction must "fit" the unsafe reduction, i.e. share
  // all of its axes. We allow up to 1 broadcast along the safe path and the
  // unsafe path, these broadcasts must exactly match in shape.
  for (Expr* targetExpr : fusion->exprs()) {
    auto* targetRop = dynamic_cast<ReductionOp*>(targetExpr);

    if (!targetRop) {
      continue;
    }

    auto reduction_type = targetRop->getReductionOpType();

    if (reduction_type != BinaryOpType::Min &&
        reduction_type != BinaryOpType::Max) {
      continue;
    }

    if (AnalyzeMinMaxOp(targetRop)) {
      targetRop->markUnsafe();
    }
  }

  return;
}

} // namespace nvfuser::preseg_passes
