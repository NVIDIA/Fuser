// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <multidevice/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <options.h>
#include <preseg_passes/remove_bcast_squeeze.h>
#include <transform_replay.h>

namespace nvfuser::preseg_passes {

namespace {

enum class AxisOp { PRESERVE, SQUEEZE, BROADCAST };

std::ostream& operator<<(std::ostream& os, AxisOp op) {
  switch (op) {
    case AxisOp::PRESERVE:
      os << "PRESERVE";
      break;
    case AxisOp::SQUEEZE:
      os << "SQUEEZE";
      break;
    case AxisOp::BROADCAST:
      os << "BROADCAST";
      break;
  }
  return os;
}

//! This represents the combined state of a collection of broadcast and squeeze
//! operations
using AxisOps = std::vector<AxisOp>;

//! Convert a broadcast Expr to an AxisOps descriptor
AxisOps broadcastToAxisOps(BroadcastOp* bcast) {
  AxisOps ops;
  const std::vector<bool>& flags = bcast->getBroadcastDimFlags();
  ops.reserve(flags.size());
  for (bool flag : flags) {
    ops.push_back(flag ? AxisOp::BROADCAST : AxisOp::PRESERVE);
  }
  return ops;
}

//! Convert a squeeze Expr to an AxisOps descriptor
AxisOps squeezeToAxisOps(SqueezeOp* squeeze) {
  AxisOps ops;
  const std::vector<bool>& flags = squeeze->getSqueezeDimFlags();
  ops.reserve(flags.size());
  for (bool flag : flags) {
    ops.push_back(flag ? AxisOp::SQUEEZE : AxisOp::PRESERVE);
  }
  return ops;
}

//! This defines the types of operations that are eligible for simplification in
//! this pass.
bool isReplaceableExpr(Expr* expr) {
  if (expr == nullptr) {
    return false;
  }
  if (isResharding(expr)) {
    return false;
  }
  return expr->isA<BroadcastOp>() || expr->isA<SqueezeOp>() ||
      ir_utils::isSimpleTVSet(expr);
}

//! Convert a LoadStoreOp to an AxisOps of all PRESERVE ops
AxisOps setToAxisOps(LoadStoreOp* ldst) {
  NVF_ERROR(ir_utils::isSimpleTVSet(ldst));
  return AxisOps(
      TensorDomain::noReductions(
          ldst->in()->as<TensorView>()->getLogicalDomain())
          .size(),
      AxisOp::PRESERVE);
}

//! Convert a replaceable op to an AxisOps object
AxisOps exprToAxisOps(Expr* expr) {
  if (auto* squeeze = dynamic_cast<SqueezeOp*>(expr)) {
    return squeezeToAxisOps(squeeze);
  } else if (auto* bcast = dynamic_cast<BroadcastOp*>(expr)) {
    return broadcastToAxisOps(bcast);
  } else if (auto* ldst = dynamic_cast<LoadStoreOp*>(expr)) {
    return setToAxisOps(ldst);
  }
  NVF_THROW(
      "exprToAxisOps expects BroadcastOp or SqueezeOp. Found ",
      expr->toString());
}

//! Return non-null if we are unable to simplify this combination to a single
//! operation. Otherwise returns the type of operation (PRESERVE indicates this
//! is a set operation).
std::optional<AxisOp> getSimplifiedOpType(const AxisOps& ops) {
  bool has_broadcast = false, has_squeeze = false;
  for (const AxisOp op : ops) {
    switch (op) {
      case AxisOp::PRESERVE:
        break;
      case AxisOp::SQUEEZE:
        has_squeeze = true;
        break;
      case AxisOp::BROADCAST:
        has_broadcast = true;
        break;
    }
  }
  if (has_broadcast && has_squeeze) {
    // Composite op
    return std::nullopt;
  }
  if (has_broadcast) {
    return AxisOp::BROADCAST;
  }
  if (has_squeeze) {
    return AxisOp::SQUEEZE;
  }
  // PRESERVE indicates this is a set op
  return AxisOp::PRESERVE;
}

//! This is useful for getting the axis flags needed to create a new BroadcastOp
//! or SqueezeOp.
std::vector<bool> nonPreservedDims(const AxisOps& ops) {
  std::vector<bool> flags;
  flags.reserve(ops.size());
  for (size_t i : arange(ops.size())) {
    flags.push_back(ops[i] != AxisOp::PRESERVE);
  }
  return flags;
}

//! Given a descriptors of two sequences of broadcast+squeeze ops, return a
//! descriptor of their composition
AxisOps composeOps(const AxisOps& prev, const AxisOps& next) {
  // We build ops by iterating in both prev and next simultaneously. To do this,
  // we hold the indices in each vector of ops that are "aligned". Alignment in
  // this context means that the previous op results in an axes (i.e. the
  // previous op is not a SQUEEZE) that is used by the next op (i.e. the next op
  // is not a BROADCAST).
  //
  // The outer loop below performs one iteration for every dimension in the
  // _intermediate_ tensor between prev and next. This means there is one main
  // loop iteration per "aligned" pair of ops.
  //
  // In each main loop iteration, we first process _unaligned_ ops in next and
  // prev, i.e. previous squeezes and next broadcasts. We cancel as many of
  // these operations as possible by converting them to PRESERVE ops and we
  // insert the remainder unmodified. After processing unaligned ops, we know
  // the next op pair is an aligned pair, meaning the previous op in the pair
  // results in an intermediate axis that is consumed in the next op. This
  // implies that the previous op is not a SQUEEZE and the next op is not a
  // BROADCAST. If either is PRESERVE, then we can simplify the composition by
  // simply using the other op. The only case left is then "broadcast then
  // squeeze", which we ignore since it does not consume an input axis or
  // produce an output axis.
  //
  // For example:
  //   prev = [ S S P B P P S ]
  //   next = [ B P P S P B B ]
  //
  // The input has dimension 6, the intermediate tensor has dimension 4, and the
  // output has dimension 6. The composition is equivalent to [ S P P P P P B ].
  // We perform 4 main loop iterations corresponding to the four intermediate
  // dimensions.
  //
  //   ops = []
  //   prev = [ (prev_pos)S S P B P P S ]
  //   next = [ (next_pos)B P P S P B B ]
  //
  //   (run processUnalignedOps()): // one cancelled SQUEEZE becomes PRESERVE
  //   ops = [ S P ]
  //   prev = [ S S (prev_pos)P B P P S ]
  //   next = [ B (next_pos)P P S P B B ]
  //
  //   First main loop iteration:
  //     (process aligned ops "preserve then preserve")
  //     ops = [ S P P ]
  //     prev = [ S S P (prev_pos)B P P S ]
  //     next = [ B P (next_pos)P S P B B ]
  //
  //     (run processUnalignedOps()) // no change
  //
  //   Second main loop iteration:
  //     (process aligned ops "broadcast then preserve")
  //     ops = [ S P P B ]
  //     prev = [ S S P B (prev_pos)P P S ]
  //     next = [ B P P (next_pos)S P B B ]
  //
  //     (run processUnalignedOps()) // no change
  //
  //   Third main loop iteration:
  //     (process aligned ops "preserve then squeeze")
  //     // Note that we would normally insert SQUEEZE resulting in
  //        ops = [ S P P B S ], but instead we change the
  //        previously-inserted BROADCAST to PRESERVE in pushOp(AxisOp::SQUEEZE)
  //     ops = [ S P P P ]
  //     prev = [ S S P B P (prev_pos)P S ]
  //     next = [ B P P S (next_pos)P B B ]
  //
  //     (run processUnalignedOps()) // no change
  //
  //   Fourth main loop iteration:
  //     (process aligned ops "preserve then preserve")
  //     ops = [ S P P P P ]
  //     prev = [ S S P B P P (prev_pos)S ]
  //     next = [ B P P S P (next_pos)B B ]
  //
  //     (run processUnalignedOps()) // one cancelled SQUEEZE becomes PRESERVE
  //     ops = [ S P P P P P B ]
  //     prev = [ S S P B P S (prev_pos) ]
  //     next = [ B P P S B B (next_pos) ]
  size_t prev_pos = 0, next_pos = 0;
  size_t prev_size = prev.size(), next_size = next.size();
  AxisOps ops;

  // This does op.push_back(op), unless this op can be combined with the
  // previous op on the ops stack
  const auto pushOp = [&](AxisOp op) {
    if (ops.empty()) {
      ops.push_back(op);
      return;
    }
    AxisOp existing_op = ops.back();

    if ((existing_op == AxisOp::SQUEEZE && op == AxisOp::BROADCAST) ||
        (existing_op == AxisOp::BROADCAST && op == AxisOp::SQUEEZE)) {
      ops.back() = AxisOp::PRESERVE;
    } else {
      ops.push_back(op);
    }
  };

  // This is run when we are unsure whether prev[prev_pos] is aligned with
  // next[next_pos]. It processes previous squeezes and next broadcasts,
  // combining them when possible.
  const auto processUnalignedOps = [&]() {
    // Count number of consecutive previous SQUEEZE ops
    while (prev_pos < prev_size && prev[prev_pos] == AxisOp::SQUEEZE) {
      pushOp(AxisOp::SQUEEZE);
      prev_pos++;
    }

    // Count number of consecutive next BROADCAST ops
    while (next_pos < next_size && next[next_pos] == AxisOp::BROADCAST) {
      pushOp(AxisOp::BROADCAST);
      next_pos++;
    }
  };

  // Process unaligned ops at the beginning of each op list
  processUnalignedOps();

  // main loop
  while (prev_pos < prev_size && next_pos < next_size) {
    // Now we have prev[prev_pos] != SQUEEZE and next[next_pos] != BROADCAST,
    // which means prev[prev_pos] provides the next op with its input. So here
    // we are actually composing these aligned ops.

    if (prev[prev_pos] == AxisOp::PRESERVE) {
      pushOp(next[next_pos]);
    } else if (next[next_pos] == AxisOp::PRESERVE) {
      // NOTE: else here implies prev[prev_pos] == AxisOp::BROADCAST
      pushOp(prev[prev_pos]);
    } else {
      // Otherwise this is a "broadcast then squeeze" pattern, which means this
      // intermediate axis did not exist in the input and does not exist in the
      // output, so skip the op altogether
      NVF_ERROR(
          next[next_pos] == AxisOp::SQUEEZE &&
          prev[prev_pos] == AxisOp::BROADCAST);
    }

    // Move to next position and process unaligned ops until we reach the end of
    // the lists or another aligned op
    prev_pos++;
    next_pos++;
    processUnalignedOps();
  }

  // If we have not exhausted prev or next, it means that there are more
  // intermediate axes that have not yet been processed, which should not
  // happen.
  NVF_ERROR(
      prev_pos == prev_size && next_pos == next_size,
      "Failed to align some ops");

  return ops;
}

TensorView* maybeDoReplacement(TensorView* orig) {
  Expr* second = orig->definition();
  if (!isReplaceableExpr(second)) {
    return orig;
  }
  Expr* first = second->input(0)->definition();
  if (!isReplaceableExpr(first)) {
    return orig;
  }

  AxisOps first_ops = exprToAxisOps(first);
  AxisOps second_ops = exprToAxisOps(second);
  AxisOps simplified_ops = composeOps(first_ops, second_ops);
  std::optional<AxisOp> simple_op_type_opt =
      getSimplifiedOpType(simplified_ops);
  if (!simple_op_type_opt.has_value()) {
    return orig;
  }
  TensorView* replacement = orig;
  if (simplified_ops == first_ops) {
    // The second op was simply a "Set" operation, so we just skip it
    replacement = first->output(0)->as<TensorView>();
  } else {
    TensorView* input_tv = first->input(0)->as<TensorView>();
    switch (simple_op_type_opt.value()) {
      case AxisOp::PRESERVE:
        // This is equivalent to a set Op
        replacement = input_tv;
        break;
      case AxisOp::SQUEEZE:
        replacement = squeeze(input_tv, nonPreservedDims(simplified_ops));
        break;
      case AxisOp::BROADCAST:
        replacement = broadcast(input_tv, nonPreservedDims(simplified_ops));
        break;
    }
  }
  NVF_ERROR(replacement != orig, "Expected non-trivial replacement");

  if (orig->isFusionOutput() && replacement->isFusionOutput() &&
      FusionGuard::getCurFusion()->getOutputAlias(orig) !=
          FusionGuard::getCurFusion()->getOutputAlias(replacement)) {
    // Refuse to do replacement of one output with another unless their aliasing
    // settings are identical.
    // See https://github.com/NVIDIA/Fuser/issues/3833
    return orig;
  }

  std::vector<IterDomain*> old_loop =
      TensorDomain::noReductions(orig->getLoopDomain());
  std::vector<IterDomain*> new_loop =
      TensorDomain::noReductions(replacement->getLoopDomain());
  NVF_ERROR(
      new_loop.size() == old_loop.size(),
      "Replacement ",
      replacement->toString(),
      " has different dimension than original ",
      orig->toString());
  // Replacing `orig` with `replacement` can introduce resharding on the
  // consumer of `orig`, e.g., the `cast` in the following example.
  //
  // [bDID{1}, i0]: replacement
  //      |
  //      | squeeze
  //      v
  //     [i0]
  //      |
  //      | broadcast
  //      v
  //  [b{1}, i0]: orig
  //      |
  //      | cast
  //      v
  //  [b{1}, i0]
  //
  // Such resharding expressions won't be resolved by `insert_reshardings`
  // because `insert_reshardings` runs before `remove_bcast_squeeze`.
  // Therefore, if resharding is needed, instead of replacing `orig` with
  // `replacement`, we link them with a resharding `set`.
  bool needs_resharding = false;
  for (size_t i : arange(old_loop.size())) {
    if (old_loop[i]->getParallelType() != new_loop[i]->getParallelType()) {
      NVF_ERROR(
          old_loop[i]->isDeviceDim() || new_loop[i]->isDeviceDim(),
          "Before scheduling, we expect the only parallelized ",
          "dimensions to be device dims");
      needs_resharding = true;
      break;
    }
  }

  // If we are replacing an output, we need to preserve the memory layout by
  // replaying the allocation domain. Otherwise it might alter user semantics,
  // violating memory layout required by aliasing
  if (orig->isFusionOutput()) {
    TransformReplay::selfReplay(
        orig->domain(), replacement->domain(), /*ignore_reductions=*/true);
  }

  if (needs_resharding) {
    IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, orig, replacement);
    return orig;
  } else {
    ir_utils::replaceValInAllExprInputsAndFusionOutputs(orig, replacement);
    return replacement;
  }
}

// Remove broadcast-squeeze and squeeze-broadcast patterns
void removeBcastSqueeze(Fusion* fusion) {
  // Iterate from outputs toward producers using a depth-first search for
  // replaceable patterns
  std::vector<TensorView*> stack;
  for (Val* outp : fusion->outputs()) {
    if (auto* tv = dynamic_cast<TensorView*>(outp)) {
      stack.push_back(tv);
    }
  }
  std::unordered_set<TensorView*> visited;
  while (!stack.empty()) {
    TensorView* tv = stack.back();
    stack.pop_back();
    if (visited.count(tv) > 0) {
      continue;
    }
    // avoid re-visiting producers when they have multiple uses
    visited.insert(tv);
    TensorView* maybe_replaced_tv = maybeDoReplacement(tv);
    if (maybe_replaced_tv != tv) {
      // If we made a replacement, process it before proceeding. For example, if
      // we have broadcast->broadcast->squeeze, we might combine the second
      // broadcast with the squeeze to form a new pattern of broadcast->squeeze,
      // which we then need to process next.
      stack.push_back(maybe_replaced_tv);
    } else if (tv->definition() != nullptr) {
      // Recurse to TensorView producers of tv
      const auto producers =
          ir_utils::filterByType<TensorView>(tv->definition()->inputs());
      stack.insert(stack.end(), producers.begin(), producers.end());
    }
  }
}

} // namespace

void RemoveBcastSqueeze::runPass(Fusion* fusion) {
  removeBcastSqueeze(fusion);
}

} // namespace nvfuser::preseg_passes
