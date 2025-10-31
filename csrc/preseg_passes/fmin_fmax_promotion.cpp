// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/fmin_fmax_promotion.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <id_model/id_model.h>
#include <ir/utils.h>
#include <logical_domain_map.h>

namespace nvfuser::preseg_passes {

namespace {

enum class TensorStatus {
  None = 0,
  Bad,
  Good,
};

using TensorMap = std::unordered_map<TensorView*, TensorStatus>;
using IterDomainMap =
    std::unordered_map<TensorView*, std::unordered_set<IterDomain*>>;

bool isSafeReduction(
    Expr* expr,
    const std::unordered_set<ReductionOp*>& promotedRops) {
  if (auto* rop = dynamic_cast<ReductionOp*>(expr)) {
    return !promotedRops.contains(rop);
  }

  return false;
}

// Checks whether a single reduction domain is "covered" by a later reduction
// op. Runs a dataflow analysis by propagating NanStatus downward along TV's in
// the fusion. The target reduction domain is passed via reduction_id_group,
// which allows us to check whether a downstream IterDomain is connected to that
// targetRop IterDomain.
bool reductionDomainIsCovered(
    ReductionOp* targetRop,
    const std::unordered_set<ReductionOp*>& promotedRops,
    int targetReduceIndex) {
  Fusion* fusion = targetRop->fusion();

  auto* in_tv = targetRop->input(0)->as<TensorView>();
  auto* out_tv = targetRop->output(0)->as<TensorView>();

  TensorMap tensor_map;
  IterDomainMap good_domains;

  good_domains[in_tv].insert(in_tv->getLogicalDomain()[targetReduceIndex]);
  tensor_map[out_tv] = TensorStatus::Bad;

  auto traversal =
      StmtSort::getExprsBetween({targetRop->input(0)}, fusion->outputs());

  for (Expr* expr : traversal) {
    if (expr == targetRop) {
      // Skip the target rop. We already marked its status.
      continue;
    }

    bool anyGoodInputs = false;
    bool anyBadInputs = false;

    for (auto input : expr->inputs()) {
      if (auto* in_tv = dynamic_cast<TensorView*>(input)) {
        TensorStatus status = tensor_map[in_tv];

        if (status == TensorStatus::Good) {
          anyGoodInputs = true;
        }
        if (status == TensorStatus::Bad) {
          anyBadInputs = true;
        }
      }
    }

    TensorStatus status = TensorStatus::None;

    auto* out_tv = dynamic_cast<TensorView*>(expr->output(0));

    bool canBeAnalyzed = expr->isA<UnaryOp>() || expr->isA<ReductionOp>() ||
        expr->isA<BroadcastOp>() || expr->isA<BinaryOp>();

    if (!out_tv || !canBeAnalyzed) {
      // Analysis is blocked for this node, treat it like a fusion output.
      if (anyBadInputs) {
        return false;
      } else {
        continue;
      }
    }

    if (anyGoodInputs) {
      status = TensorStatus::Good;
    } else if (anyBadInputs) {
      status = TensorStatus::Bad;
    }

    for (auto input : expr->inputs()) {
      if (auto* in_tv = dynamic_cast<TensorView*>(input)) {
        PairwiseLogicalDomainMap p2c_map(in_tv, out_tv);
        p2c_map.mapBroadcast(false);
        auto p2c = p2c_map.mapProducerToConsumer();
        for (IterDomain* in_id : good_domains[in_tv]) {
          if (!p2c.contains(in_id)) {
            continue;
          }

          IterDomain* out_id = p2c.at(in_id);
          good_domains[out_tv].insert(out_id);

          if (out_id->isReduction() && isSafeReduction(expr, promotedRops)) {
            status = TensorStatus::Good;
          }
        }
      }
    }

    tensor_map[out_tv] = status;
  }

  auto output_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
  for (TensorView* out_tv : output_tvs) {
    if (tensor_map[out_tv] == TensorStatus::Bad) {
      return false;
    }
  }

  return true;
}

// Iterates over all reduction dimensions in the target ROP and ensures they
// are all "covered" by subsequent reduction ops.
//
// Note that this currently re-runs the traversal/dataflow analysis for every
// single reduction domain. This could be merged into a single traversal,
// however it would require per-domain tracked of the NanStatusMap, and it would
// make the propagation code more complicated.
bool minMaxOpIsCovered(
    ReductionOp* targetRop,
    const std::unordered_set<ReductionOp*>& promotedRops) {
  auto* out_tv = targetRop->output(0)->as<TensorView>();

  int i = -1;
  for (IterDomain* out_id : out_tv->getLogicalDomain()) {
    ++i;
    if (out_id->isReduction()) {
      if (!reductionDomainIsCovered(targetRop, promotedRops, i)) {
        return false;
      }
    }
  }

  return true;
}

} // namespace

void FMinFMaxPromotionPass::runPass(Fusion* fusion) {
  FusionGuard fusion_guard(fusion);

  std::unordered_set<ReductionOp*> promotedRops;

  // This outer loop runs over all expressions, filtering for min/max
  // reductions, which become the target for the rest of the analysis.
  for (Expr* targetExpr : fusion->exprs()) {
    auto* targetRop = dynamic_cast<ReductionOp*>(targetExpr);

    if (!targetRop) {
      continue;
    }

    auto reduction_type = targetRop->getReductionOpType();

    if (reduction_type == BinaryOpType::Min ||
        reduction_type == BinaryOpType::Max) {
      if (minMaxOpIsCovered(targetRop, promotedRops)) {
        promotedRops.insert(targetRop);
      }
    }
  }

  for (auto* rop : promotedRops) {
    // Promote the reduction ops by doing expression replacement
    auto red_op_type = rop->getReductionOpType();
    auto init = rop->init();
    auto out = rop->out();
    auto in = rop->in();

    if (red_op_type == BinaryOpType::Max) {
      red_op_type = BinaryOpType::FMax;
    }
    if (red_op_type == BinaryOpType::Min) {
      red_op_type = BinaryOpType::FMin;
    }

    fusion->removeExpr(rop);
    IrBuilder::create<ReductionOp>(red_op_type, init, out, in, true);
  }

  return;
}

} // namespace nvfuser::preseg_passes
