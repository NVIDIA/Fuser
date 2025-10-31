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

//
// To analyze a single min or max reduction op, the "target" op, we perform
// a downstream dataflow analysis to detect whether a NAN squelched by the
// promotion will reach any fusion outputs, or if it will be repaired by a
// downstream "safe" reductions.
//
// The entire analysis happens on TensorView, not IterDomains.
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

// --- Dataflow Example:
// Let tv0, tv1 be fusion input TVs. Then here are the NanStatus assigned to
// each TV in an example fusion subgraph:
// tv0                                   NONE
// tv1                                   Unreduced
// tv2 = max(tv1, {0, 1})                BadReduced
// tv3 = broadcast(tv2, {true, true})    BadReduced
// tv4 = add(tv3, tv1)                   Mixed
// tv5 = sum(tv4, {0, 1})                GoodReduced
// tv6 = broadcast(tv5, {true, true})    GoodReduced
// tv7 = add(tv6, tv5)                   GoodReduced
// tv8 = add(tv7, tv0)                   GoodReduced
// fusion->addOutput(tv8)
//
// ---- Step-by-step:
// 1. All TV's have a None state by default.
// 2. The analysis is launched on the max reduction which produces tv2
// 3. We assign tv2 a BadReduced status, and tv1 an Unreduced status
// 4. We traverse to the broadcast producing tv3. We propagate the BadReduced
//    state to tv3. This is the first broadcast op we have seen, so we save it
//    to subsequently enforce all broadcasts match its axes.
// 5. We traverse the tv4 add expr. This is a binary op between an Unreduced and
//    a BadReduced state, so tv4 gets a Mixed state.
// 6. We traverse the tv5 sum expr. This is a safe reduction of a mixed state.
//    Since Mixed state carries the original NAN data that entered tv2, this
//    safe reduction creates a GoodReduced state. We also checked to make sure
//    this reduction's axes match the target reduction's axes.
// 7. We traverse the tv6 broadcast expr. We already saw a broadcast before, so
//    we ensure that this broadcast matches the axes of the prior one. It does,
//    so we prop the GoodReduced state and continue.
// 8. The subsequent add() simply propagate GoodReduced state since it is the
//    highest priority.
// 9. No output TV's contain a BadReduced or Mixed state, so the max() op can
//    be promoted.

using NanStatusMap = std::unordered_map<TensorView*, NanStatus>;
using PromotedOpSet = std::unordered_set<ReductionOp*>;

bool isSafeReduction(Expr* expr, const PromotedOpSet& promotedOps) {
  if (auto* rop = dynamic_cast<ReductionOp*>(expr)) {
    // Check that this expr hasn't already been promoted to an unsafe reduction.
    return !promotedOps.contains(rop);
  }

  return false;
}

bool reductionMatches(ReductionOp* left, ReductionOp* right) {
  auto* left_tv = dynamic_cast<TensorView*>(left->output(0));
  auto* right_tv = dynamic_cast<TensorView*>(right->output(0));

  if (left_tv->nDims() != right_tv->nDims()) {
    return false;
  }

  for (int i = 0; i < left_tv->nDims(); ++i) {
    if (left_tv->getLogicalDomain()[i]->isReduction() !=
        right_tv->getLogicalDomain()[i]->isReduction()) {
      return false;
    }
  }

  return true;
}

bool broadcastMatches(BroadcastOp* left, BroadcastOp* right) {
  auto* left_tv = dynamic_cast<TensorView*>(left->output(0));
  auto* right_tv = dynamic_cast<TensorView*>(right->output(0));

  if (left_tv->nDims() != right_tv->nDims()) {
    return false;
  }

  for (int i = 0; i < left_tv->nDims(); ++i) {
    if (left_tv->getLogicalDomain()[i]->isBroadcast() !=
        right_tv->getLogicalDomain()[i]->isBroadcast()) {
      return false;
    }
  }

  return true;
}

bool canBeAnalyzed(
    Expr* expr,
    ReductionOp* compareRop,
    std::optional<BroadcastOp*>& compareBop) {
  // This is where we enforce the restricted-subgraph rules. Arbitrary binary
  // and binary ops are allowed, and do not affect the analysis. Reduction and
  // broadcasts have strict requirements to simplify the state tracking.

  if (expr->isA<UnaryOp>() || expr->isA<BinaryOp>()) {
    return true;
  } else if (auto* rop = dynamic_cast<ReductionOp*>(expr)) {
    // We require all reduction ops exactly match in reduction axes.
    // This avoids the need for complicated IterDomain handling.
    return reductionMatches(rop, compareRop);
  } else if (auto* bop = dynamic_cast<BroadcastOp*>(expr)) {
    // Similarly for reductions, we require all broadcasts to have the same
    // axes.
    if (!compareBop) {
      compareBop = bop;
      return true;
    } else {
      return broadcastMatches(bop, *compareBop);
    }
  }

  return false;
}

// Traverses the restricted subgraph around the target rop and checks whether
// NANs which would be squelched by a promotion, will be subsequently repaired
// by safe reductions.
bool minMaxOpIsRepaired(
    ReductionOp* targetRop,
    const PromotedOpSet& promotedOps) {
  Fusion* fusion = targetRop->fusion();

  auto* in_tv = targetRop->input(0)->as<TensorView>();
  auto* out_tv = targetRop->output(0)->as<TensorView>();

  NanStatusMap status_map;

  status_map.emplace(in_tv, NanStatus::Unreduced);
  status_map.emplace(out_tv, NanStatus::BadReduced);

  std::optional<BroadcastOp*> broadcastMatcher;

  // Topological traversal downstream of the targetRop input.
  // Note we start from the input, not the output, of the targetRop, because
  // we need to track the Unreduced state, so it can make repairs.
  auto traversal =
      StmtSort::getExprsBetween({targetRop->input(0)}, fusion->outputs());

  for (Expr* expr : traversal) {
    if (expr == targetRop) {
      // Skip the target rop. We already marked its status.
      continue;
    }

    // Get aggregate status from all inputs.
    bool anyUnreduced = false;
    bool anyBadReduced = false;
    bool anyMixed = false;
    bool anyGoodReduced = false;

    for (auto input : expr->inputs()) {
      if (auto* in_tv = dynamic_cast<TensorView*>(input)) {
        NanStatus status = NanStatus::None;

        auto it = status_map.find(in_tv);
        if (it != status_map.end()) {
          status = it->second;
        }

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

    if (!canBeAnalyzed(expr, targetRop, broadcastMatcher)) {
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

    if (isSafeReduction(expr, promotedOps)) {
      if (status == NanStatus::Unreduced || status == NanStatus::Mixed) {
        // Unreduced and Mixed states both indicate the targetRop's input have
        // propagated here pointwise, preserving their NAN values unchanged
        // positions. Therefore, this reduction will create the tensor with
        // reduced NAN values matching the original targetRop if it propagated
        // its NANs.
        status = NanStatus::GoodReduced;
      }
    }

    auto* out_tv = dynamic_cast<TensorView*>(expr->output(0));

    status_map.emplace(out_tv, status);
  }

  // Check whether any bad status reached output nodes
  auto output_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
  for (TensorView* out_tv : output_tvs) {
    NanStatus status = NanStatus::None;

    auto it = status_map.find(out_tv);
    if (it != status_map.end()) {
      status = it->second;
    }

    if (status == NanStatus::BadReduced || status == NanStatus::Mixed) {
      return false;
    }
  }

  return true;
}

} // namespace

void FMinFMaxPromotionPass::runPass(Fusion* fusion) {
  FusionGuard fusion_guard(fusion);

  PromotedOpSet promotedOps;

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
      if (minMaxOpIsRepaired(targetRop, promotedOps)) {
        promotedOps.insert(targetRop);
      }
    }
  }

  for (auto* rop : promotedOps) {
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
