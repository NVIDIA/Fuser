// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
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

// IterDomainStatus are attached to IterDomains and propagated with a
// downward-flow algorithm. The goal is to detect whether lost NANs propagated
// to the outputs of a fusion. If so, then it is not valid to do the fast
// min/max promotion.
//
// "BAD" statuses indicate ID's that potentially lost their NANs.
// "GOOD" statuses will repair BAD statuses.
// "REDUCE" statuses are reduced dimensions, and should never be mapped to ID's
//   with non-reduced status.
// All other statuses are "full-sized" (besides NONE which says nothing).
//
// A reduction ID will convert a full-size status into a REDUCE status. Likewise
// a broadcast ID will convert a reduced status to a full size status.
//
// Statuses can interact via binary ops in a kind of algebra. The two reduced
// statuses have a simple interaction: GOOD_REDUCE beats BAD-REDUCE.
//
// The 4 full-sized statuses have a slightly more complicated interaction:
// 1. Any status matched with itself, produces itself.
// 2. GOOD_BROADCAST matched with anything produces GOOD_BROADCAST.
// 3. All other cases produce BAD_BROADCAST_DEFAULT
//
// An example of what each status means is below (with *max* reduction):
// [0.0  1.0  2.0  3.0  NAN  5.0] <- DEFAULT
// [5.0]                          <- BAD_REDUCE
// [NAN]                          <- GOOD_REDUCE
// [5.0  5.0  5.0  5.0  5.0  5.0] <- BAD_BROADCAST
// [5.0  5.0  5.0  5.0  NAN  5.0] <- BAD_BROADCAST_DEFAULT
// [NAN  NAN  NAN  NAN  NAN  NAN] <- GOOD_BROADCAST
enum class IterDomainStatus {
  // NONE is the status when hitting untracked ID's.
  NONE,

  // Reduced statuses
  BAD_REDUCE,
  GOOD_REDUCE,

  // Full-size statuses
  DEFAULT,
  BAD_BROADCAST,
  BAD_BROADCAST_DEFAULT,
  GOOD_BROADCAST,
};

using IterStatusMap = std::unordered_map<IterDomain*, IterDomainStatus>;

IterDomainStatus BopStatus(IterDomainStatus lhs, IterDomainStatus rhs) {
  if (lhs == IterDomainStatus::NONE) {
    return rhs;
  }

  if (rhs == IterDomainStatus::NONE) {
    return lhs;
  }

  if (lhs == IterDomainStatus::GOOD_REDUCE ||
      rhs == IterDomainStatus::GOOD_REDUCE) {
    return IterDomainStatus::GOOD_REDUCE;
  } else if (
      lhs == IterDomainStatus::BAD_REDUCE &&
      rhs == IterDomainStatus::BAD_REDUCE) {
    return IterDomainStatus::BAD_REDUCE;
  }

  if (lhs == rhs) {
    return lhs;
  }

  if (lhs == IterDomainStatus::GOOD_BROADCAST ||
      rhs == IterDomainStatus::GOOD_BROADCAST) {
    return IterDomainStatus::GOOD_BROADCAST;
  }

  // The only remaining cases are combinations of DEFAULT,
  // BAD_BROADCAST, and BAD_BROADCAST_DEFAULT.
  return IterDomainStatus::BAD_BROADCAST_DEFAULT;
}

bool StatusIsBad(IterDomainStatus status) {
  return status == IterDomainStatus::BAD_REDUCE ||
      status == IterDomainStatus::BAD_BROADCAST ||
      status == IterDomainStatus::BAD_BROADCAST_DEFAULT;
}

bool AnyBadInputs(Expr* expr, IterStatusMap& iterMap) {
  for (auto input : expr->inputs()) {
    if (auto* in_tv = dynamic_cast<TensorView*>(input)) {
      for (IterDomain* id : in_tv->getLogicalDomain()) {
        IterDomainStatus status = iterMap[id];
        if (StatusIsBad(status)) {
          return true;
        }
      }
    }
  }

  return false;
}

// Once we identify a target reduction, we perform a downward pass starting from
// the target's direct input. The pass propagates IterDomainStatus information.
// At the end, we check all output TV's for bad statuses. If at any point we
// encounter a node we don't know how to propagate information through, we treat
// it like to a graph output and fail if it has any incoming bad statuses.
bool AnalyzeMinMaxOp(ReductionOp* targetRop) {
  Fusion* fusion = targetRop->fusion();

  FusionGuard fg(fusion);
  ComputeAtLogicalDomainMap logical_map;
  logical_map.build(true);

  IterStatusMap iterMap;

  auto* in_tv = targetRop->input(0)->as<TensorView>();
  for (IterDomain* in_id : in_tv->getLogicalDomain()) {
    iterMap[in_id] = IterDomainStatus::DEFAULT;
  }

  auto* out_tv = targetRop->output(0)->as<TensorView>();
  for (IterDomain* out_id : out_tv->getLogicalDomain()) {
    if (out_id->isReduction()) {
      iterMap[out_id] = IterDomainStatus::BAD_REDUCE;
    } else {
      iterMap[out_id] = IterDomainStatus::DEFAULT;
    }
  }

  auto traversal =
      StmtSort::getExprsBetween({targetRop->input(0)}, fusion->outputs());
  for (Expr* expr : traversal) {
    std::string opName = expr->getOpString();

    if (expr == targetRop) {
      // Skip the target rop. We already marked its status.
      continue;
    }

    bool anyBadInputs = AnyBadInputs(expr, iterMap);

    auto* out_tv = dynamic_cast<TensorView*>(expr->output(0));

    if (!out_tv) {
      if (anyBadInputs) {
        return false;
      } else {
        continue;
      }
    }

    if (expr->isA<UnaryOp>() || expr->isA<ReductionOp>() ||
        expr->isA<BroadcastOp>()) {
      auto in_tv = expr->input(0)->as<TensorView>();
      auto p2c = logical_map.mapBestEffort(
          in_tv->domain(),
          in_tv->getLogicalDomain(),
          out_tv->domain(),
          out_tv->getLogicalDomain());

      for (IterDomain* in_id : in_tv->getLogicalDomain()) {
        IterDomainStatus status = iterMap[in_id];
        auto out_id = p2c[in_id];

        if (out_id) {
          if (out_id->isReduction()) {
            if (status == IterDomainStatus::BAD_BROADCAST) {
              status = IterDomainStatus::BAD_REDUCE;
            } else if (status != IterDomainStatus::NONE) {
              status = IterDomainStatus::GOOD_REDUCE;
            }
          }

          if (out_id->isBroadcast()) {
            if (status == IterDomainStatus::BAD_REDUCE) {
              status = IterDomainStatus::BAD_BROADCAST;
            } else if (status != IterDomainStatus::NONE) {
              status = IterDomainStatus::GOOD_BROADCAST;
            }
          }

          iterMap[out_id] = status;
        } else {
          if (StatusIsBad(status)) {
            return false;
          }
        }
      }

    } else if (expr->isA<BinaryOp>()) {
      auto* left_tv = dynamic_cast<TensorView*>(expr->input(0));
      auto* right_tv = dynamic_cast<TensorView*>(expr->input(1));

      // One side (not both) might not be a TensorView.
      // To handle this, just propagate the status of the other side.
      if (!left_tv) {
        left_tv = right_tv;
      } else if (!right_tv) {
        right_tv = left_tv;
      }

      auto left2right = logical_map.mapBestEffort(
          left_tv->domain(),
          left_tv->getLogicalDomain(),
          right_tv->domain(),
          right_tv->getLogicalDomain());

      auto left2out = logical_map.mapBestEffort(
          left_tv->domain(),
          left_tv->getLogicalDomain(),
          out_tv->domain(),
          out_tv->getLogicalDomain());

      for (IterDomain* left_id : left_tv->getLogicalDomain()) {
        // Note: this assumes that the left <-> right mapping exists
        // Does this need to handle left-right mapping failures?

        IterDomainStatus leftStatus = iterMap[left_id];
        IterDomainStatus rightStatus = iterMap[left2right[left_id]];

        IterDomainStatus status = BopStatus(leftStatus, rightStatus);

        auto out_id = left2out[left_id];

        if (out_id) {
          iterMap[out_id] = status;
        } else {
          if (StatusIsBad(status)) {
            return false;
          }
        }
      }

    } else {
      // unknown op type, ensure it has no bad status since information will not
      // flow through it.
      if (anyBadInputs) {
        return false;
      }
    }
  }

  // Check whether any bad status reached output nodes
  auto output_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
  for (TensorView* tv : output_tvs) {
    for (IterDomain* id : tv->getLogicalDomain()) {
      IterDomainStatus status = iterMap[id];
      if (StatusIsBad(status)) {
        return false;
      }
    }
  }

  return true;
}

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
      if (AnalyzeMinMaxOp(targetRop)) {
        targetRop->markUnsafe();
      }
    }
  }

  return;
}

} // namespace nvfuser::preseg_passes
