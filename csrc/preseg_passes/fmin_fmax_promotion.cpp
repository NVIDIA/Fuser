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

#include <id_model/id_model.h>
#include <ir/utils.h>
#include <logical_domain_map.h>

namespace nvfuser::preseg_passes {

namespace {

enum class ValStatus {
  None,
  Unreduced,
  BadReduced,
  Mixed,
  GoodReduced,
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

bool analyzeReduceDomain(
    ReductionOp* targetRop,
    const ValGroup& reduce_axis_id_group) {
  Fusion* fusion = targetRop->fusion();

  auto* in_tv = targetRop->input(0)->as<TensorView>();
  auto* out_tv = targetRop->output(0)->as<TensorView>();

  ValStatusMap val_map;

  val_map[in_tv] = ValStatus::Unreduced;
  val_map[out_tv] = ValStatus::BadReduced;

  auto traversal =
      StmtSort::getExprsBetween({targetRop->input(0)}, fusion->outputs());

  for (Expr* expr : traversal) {
    if (expr == targetRop) {
      // Skip the target rop. We already marked its status.
      continue;
    }

    bool anyUnreduced = false;
    bool anyBadReduced = false;
    bool anyMixed = false;
    bool anyGoodReduced = false;

    for (auto input : expr->inputs()) {
      if (auto* in_tv = dynamic_cast<TensorView*>(input)) {
        ValStatus status = val_map[in_tv];

        if (status == ValStatus::Unreduced) {
          anyUnreduced = true;
        }
        if (status == ValStatus::BadReduced) {
          anyBadReduced = true;
        }
        if (status == ValStatus::Mixed) {
          anyMixed = true;
        }
        if (status == ValStatus::GoodReduced) {
          anyGoodReduced = true;
        }
      }
    }

    auto* out_tv = dynamic_cast<TensorView*>(expr->output(0));

    bool canBeAnalyzed = expr->isA<UnaryOp>() || expr->isA<ReductionOp>() ||
        expr->isA<BroadcastOp>() || expr->isA<BinaryOp>();

    if (!out_tv || !canBeAnalyzed) {
      if (anyBadReduced || anyMixed) {
        return false;
      } else {
        continue;
      }
    }

    ValStatus status = ValStatus::None;

    // Determine this node's status based on its inputs.
    // Status is mostly propped based on priority. For example, GoodReduced
    // beats all other states. There is also one combination rule with
    // BadReduced and Unreduced combining to become Mixed.
    if (anyGoodReduced) {
      status = ValStatus::GoodReduced;
    } else if (anyMixed) {
      status = ValStatus::Mixed;
    } else if (anyUnreduced && anyBadReduced) {
      status = ValStatus::Mixed;
    } else if (anyUnreduced) {
      status = ValStatus::Unreduced;
    } else if (anyBadReduced) {
      status = ValStatus::BadReduced;
    }

    // Here we handle the repair of squelched NAN's. This happens when
    // IterDomain's with Unreduced data are safely reduced.
    if (isSafeReduction(expr)) {
      if (status == ValStatus::Unreduced || status == ValStatus::Mixed) {
        for (IterDomain* out_id : out_tv->getLogicalDomain()) {
          if (out_id->isReduction()) {
            // This id_group check verifies that the reduction axis matches
            // the axis of the target unsafe reduction.
            if (reduce_axis_id_group->has(out_id)) {
              status = ValStatus::GoodReduced;
              break;
            }
          }
        }
      }
    }

    val_map[out_tv] = status;
  }

  // Check whether any bad status reached output nodes
  auto output_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
  for (TensorView* out_tv : output_tvs) {
    if (val_map[out_tv] == ValStatus::BadReduced ||
        val_map[out_tv] == ValStatus::Mixed) {
      return false;
    }
  }

  return true;
}

bool analyzeMinMaxOp(ReductionOp* targetRop) {
  auto* out_tv = targetRop->output(0)->as<TensorView>();

  IdModel id_model(targetRop->fusion(), true);
  const ValGraph& perm_graph = id_model.idGraph(IdMappingMode::PERMISSIVE);

  for (IterDomain* out_id : out_tv->getLogicalDomain()) {
    if (out_id->isReduction()) {
      const ValGroup& reduce_axis_id_group = perm_graph.toGroup(out_id);
      if (!analyzeReduceDomain(targetRop, reduce_axis_id_group)) {
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
