// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/fmin_fmax_promotion.h>

#include <unordered_map>
#include <vector>

#include <ir/utils.h>
#include <logical_domain_map.h>

namespace nvfuser::preseg_passes {

namespace {

enum class ValStatus {
  NONE,
  DEFAULT,
  BAD,
  BAD_DEFAULT,
  GOOD,
};

using ValStatusMap = std::unordered_map<Val*, ValStatus>;

bool isSafeReduction(Expr* expr) {
  if (auto* rop = dynamic_cast<ReductionOp*>(expr)) {
    auto reduction_type = rop->getReductionOpType();

    return reduction_type != BinaryOpType::FMin &&
        reduction_type != BinaryOpType::FMax;
  }

  return false;
}

bool anyBadInputTvs(Expr* expr, ValStatusMap& val_map) {
  for (auto input : expr->inputs()) {
    if (auto* in_tv = dynamic_cast<TensorView*>(input)) {
      ValStatus status = val_map[in_tv];
      if (status == ValStatus::BAD || status == ValStatus::BAD_DEFAULT) {
        return true;
      }
    }
  }

  return false;
}

bool anyGoodInputTvs(Expr* expr, ValStatusMap& val_map) {
  for (auto input : expr->inputs()) {
    if (auto* in_tv = dynamic_cast<TensorView*>(input)) {
      ValStatus status = val_map[in_tv];
      if (status == ValStatus::GOOD) {
        return true;
      }
    }
  }

  return false;
}

bool anyDefaultInputTvs(Expr* expr, ValStatusMap& val_map) {
  for (auto input : expr->inputs()) {
    if (auto* in_tv = dynamic_cast<TensorView*>(input)) {
      ValStatus status = val_map[in_tv];
      if (status == ValStatus::DEFAULT) {
        return true;
      }
    }
  }

  return false;
}

bool anyBadDefaultInputTvs(Expr* expr, ValStatusMap& val_map) {
  for (auto input : expr->inputs()) {
    if (auto* in_tv = dynamic_cast<TensorView*>(input)) {
      ValStatus status = val_map[in_tv];
      if (status == ValStatus::BAD_DEFAULT) {
        return true;
      }
    }
  }

  return false;
}

bool analyzeReduceDomain(
    ReductionOp* targetRop,
    IterDomain* reduceIn,
    ComputeAtLogicalDomainMap& logical_map) {
  Fusion* fusion = targetRop->fusion();

  auto* in_tv = targetRop->input(0)->as<TensorView>();
  auto* out_tv = targetRop->output(0)->as<TensorView>();

  ValStatusMap val_map;

  val_map[in_tv] = ValStatus::DEFAULT;
  val_map[out_tv] = ValStatus::BAD;

  auto traversal =
      StmtSort::getExprsBetween({targetRop->input(0)}, fusion->outputs());

  for (Expr* expr : traversal) {
    std::string opName = expr->getOpString();

    if (expr == targetRop) {
      // Skip the target rop. We already marked its status.
      continue;
    }

    bool anyBad = anyBadInputTvs(expr, val_map);
    bool anyBadDefault = anyBadDefaultInputTvs(expr, val_map);
    bool anyDefault = anyDefaultInputTvs(expr, val_map);

    auto* out_tv = dynamic_cast<TensorView*>(expr->output(0));

    bool canBeAnalyzed = expr->isA<UnaryOp>() || expr->isA<ReductionOp>() ||
        expr->isA<BroadcastOp>() || expr->isA<BinaryOp>();

    if (!out_tv || !canBeAnalyzed) {
      if (anyBad) {
        return false;
      } else {
        continue;
      }
    }

    if (anyGoodInputTvs(expr, val_map)) {
      val_map[out_tv] = ValStatus::GOOD;
      continue;
    }

    bool mappedReduction = false;
    if (isSafeReduction(expr)) {
      if (val_map[expr->input(0)->as<TensorView>()] == ValStatus::DEFAULT ||
          val_map[expr->input(0)->as<TensorView>()] == ValStatus::BAD_DEFAULT) {
        for (IterDomain* out_id : out_tv->getLogicalDomain()) {
          if (out_id->isReduction()) {
            if (logical_map.canMap(
                    targetRop->input(0)->as<TensorView>()->domain(),
                    reduceIn,
                    out_tv->domain(),
                    out_id)) {
              val_map[out_tv] = ValStatus::GOOD;
              mappedReduction = true;
              break;
            }
          }
        }
      }
    }
    if (mappedReduction) {
      continue;
    }

    if (anyBadDefault) {
      val_map[out_tv] = ValStatus::BAD_DEFAULT;
      continue;
    }

    if (anyDefault && anyBad) {
      val_map[out_tv] = ValStatus::BAD_DEFAULT;
      continue;
    }

    if (anyDefault) {
      val_map[out_tv] = ValStatus::DEFAULT;
      continue;
    }

    if (anyBad) {
      val_map[out_tv] = ValStatus::BAD;
    }
  }

  // Check whether any bad status reached output nodes
  auto output_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
  for (TensorView* out_tv : output_tvs) {
    if (val_map[out_tv] == ValStatus::BAD ||
        val_map[out_tv] == ValStatus::BAD_DEFAULT) {
      return false;
    }
  }

  return true;
}

bool analyzeMinMaxOp(ReductionOp* targetRop) {
  auto* in_tv = targetRop->input(0)->as<TensorView>();
  auto* out_tv = targetRop->output(0)->as<TensorView>();

  ComputeAtLogicalDomainMap logical_map;
  logical_map.build(true);

  auto c2p = logical_map.mapBestEffort(
      out_tv->domain(),
      out_tv->getLogicalDomain(),
      in_tv->domain(),
      in_tv->getLogicalDomain());

  for (IterDomain* out_id : out_tv->getLogicalDomain()) {
    if (out_id->isReduction()) {
      IterDomain* in_id = c2p[out_id];

      if (!analyzeReduceDomain(targetRop, in_id, logical_map)) {
        return false;
      }
    }
  }

  return true;
}

} // namespace

void FMinFMaxPromotionPass::runPass(Fusion* fusion) {
  FusionGuard fusion_guard(fusion);

  // The outer loop runs over all expressions, filtering out most of them.
  // It stops only on min/max reductions, which become the target for the rest
  // of the analysis.
  for (Expr* targetExpr : fusion->exprs()) {
    auto* targetRop = dynamic_cast<ReductionOp*>(targetExpr);

    if (!targetRop) {
      continue;
    }

    auto reduction_type = targetRop->getReductionOpType();

    if (reduction_type == BinaryOpType::Min ||
        reduction_type == BinaryOpType::Max) {
      if (analyzeMinMaxOp(targetRop)) {
        targetRop->markUnsafe();
      }
    }
  }

  return;
}

} // namespace nvfuser::preseg_passes
