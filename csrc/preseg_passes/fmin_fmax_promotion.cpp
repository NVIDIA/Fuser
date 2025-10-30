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

// NanStatus are attached to TensorViews and propagated with a
// downward-flow algorithm. The goal is to detect whether lost NANs will reach
// the outputs of a fusion. If so, then it is not valid to do the fast min/max
// promotion for a particular reduction axis.
enum class NanStatus {
  // "None" status corresponds to a lack of relevant information. This status is
  // the default when looking up an un-tracked node in the NanStatusMap. This
  // can happen when checking e.g. an input to a binary op, which is a TV from
  // some other part of the fusion not traversed during analysis. "None" is the
  // lowest precedence state, everything else overwrites it.
  None = 0,

  // "Unreduced" is the state attached to the input of the target reduction op.
  // It becomes "GoodReduced" if it passes through a safe reduction. It's safe
  // in the sense that it is allowed to reach fusion outputs. But it won't fix
  // a bad status.
  Unreduced,

  // "BadReduced" is attached to the output of the target reduction op. It
  // corresponds to a TV that has its data modified by the fmin/fmax promotion.
  // If it reaches an output node, we know we cannot do the fmin/fmax promotion
  // because it might change fusion output data.
  BadReduced,

  // "Mixed" is a combination of "Unreduced" and "BadReduced" states. It
  // contains the NANs from an "Unreduced" state, so a safe reduction will
  // transform it into a "GoodReduced" state. But it is also downstream of an
  // unrepaired bad reduction, so if it reaches an output, that output may have
  // lost NAN data.
  Mixed,

  // "GoodReduced" is the status that is reached after "Unreduced" data passes
  // through a safe reduction. It is the highest precedence state, and repairs
  // all other states.
  GoodReduced,
};

// An example of what each status looks like is below, with *max* reduction:
//
// [0.0  1.0  2.0  3.0  NAN  5.0] <- Unreduced
// [5.0, 5.0, 5.0, 5.0, 5.0, 5.0] <- BadReduced
// [NAN, NAN, NAN, NAN, NAN, NAN] <- GoodReduced
// [5.0  5.0  5.0  5.0  NAN  5.0] <- Mixed
//
// Note that "Mixed" appears the same as "Unreduced" - what is the difference?
// The difference is that "Mixed" signals that a node is downstream of the bad
// reduction. If the bad reduction had been a good reduction, then a TV with
// a "Mixed" state would have data like a "GoodReduced" node.

using NanStatusMap = std::unordered_map<TensorView*, NanStatus>;

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
    const ValGroup& reduction_id_group,
    const std::unordered_set<ReductionOp*>& promotedRops) {
  Fusion* fusion = targetRop->fusion();

  auto* in_tv = targetRop->input(0)->as<TensorView>();
  auto* out_tv = targetRop->output(0)->as<TensorView>();

  NanStatusMap status_map;

  status_map[in_tv] = NanStatus::Unreduced;
  status_map[out_tv] = NanStatus::BadReduced;

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
        NanStatus status = status_map[in_tv];

        if (status == NanStatus::Unreduced) {
          anyUnreduced = true;
        }
        if (status == NanStatus::BadReduced) {
          anyBadReduced = true;
        }
        if (status == NanStatus::Mixed) {
          anyMixed = true;
        }
        if (status == NanStatus::GoodReduced) {
          anyGoodReduced = true;
        }
      }
    }

    auto* out_tv = dynamic_cast<TensorView*>(expr->output(0));

    bool canBeAnalyzed = expr->isA<UnaryOp>() || expr->isA<ReductionOp>() ||
        expr->isA<BroadcastOp>() || expr->isA<BinaryOp>();

    if (!out_tv || !canBeAnalyzed) {
      // Analysis is blocked for this node, treat it like a fusion output.
      if (anyBadReduced || anyMixed) {
        return false;
      } else {
        continue;
      }
    }

    NanStatus status = NanStatus::None;

    // Determine this node's status based on its inputs.
    // Status is mostly propped based on priority. For example, GoodReduced
    // beats all other states. There is also one combination rule with
    // BadReduced and Unreduced combining to become Mixed.
    if (anyGoodReduced) {
      status = NanStatus::GoodReduced;
    } else if (anyMixed) {
      status = NanStatus::Mixed;
    } else if (anyUnreduced && anyBadReduced) {
      status = NanStatus::Mixed;
    } else if (anyUnreduced) {
      status = NanStatus::Unreduced;
    } else if (anyBadReduced) {
      status = NanStatus::BadReduced;
    }

    // Here we handle the repair of squelched NAN's. This happens when
    // IterDomain's with "Unreduced" data are safely reduced.
    if (isSafeReduction(expr, promotedRops)) {
      if (status == NanStatus::Unreduced || status == NanStatus::Mixed) {
        for (IterDomain* out_id : out_tv->getLogicalDomain()) {
          if (out_id->isReduction()) {
            // This id_group check verifies that the reduction axis matches
            // the axis of the target unsafe reduction.
            if (reduction_id_group->has(out_id)) {
              status = NanStatus::GoodReduced;
              break;
            }
          }
        }
      }
    }

    status_map[out_tv] = status;
  }

  // Check whether any bad status reached output nodes
  auto output_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
  for (TensorView* out_tv : output_tvs) {
    if (status_map[out_tv] == NanStatus::BadReduced ||
        status_map[out_tv] == NanStatus::Mixed) {
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
    const ValGraph& graph,
    const std::unordered_set<ReductionOp*>& promotedRops) {
  auto* out_tv = targetRop->output(0)->as<TensorView>();

  for (IterDomain* out_id : out_tv->getLogicalDomain()) {
    if (out_id->isReduction()) {
      const ValGroup& reduction_id_group = graph.toGroup(out_id);
      if (!reductionDomainIsCovered(
              targetRop, reduction_id_group, promotedRops)) {
        return false;
      }
    }
  }

  return true;
}

} // namespace

void FMinFMaxPromotionPass::runPass(Fusion* fusion) {
  FusionGuard fusion_guard(fusion);

  IdModel id_model(fusion, false);
  id_model.buildGraph(IdMappingMode::EXACT);
  const ValGraph& graph = id_model.idGraph(IdMappingMode::EXACT);

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
      if (minMaxOpIsCovered(targetRop, graph, promotedRops)) {
        promotedRops.insert(targetRop);
      }
    }
  }

  for (auto* rop : promotedRops) {
    rop->markUnsafe();
  }

  return;
}

} // namespace nvfuser::preseg_passes
