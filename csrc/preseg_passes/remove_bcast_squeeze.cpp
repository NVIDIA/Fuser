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
#include <ops/alias.h>
#include <ops/arith.h>
#include <options.h>
#include <preseg_passes/remove_bcast_squeeze.h>

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

//! Checks whether this is a simple Set of a TensorView. If not, then this might
//! represent a scalar set, or a segment_set.
bool isSimpleTVSet(Expr* expr) {
  auto* ldst = dynamic_cast<LoadStoreOp*>(expr);
  if (ldst == nullptr) {
    return false;
  }
  return ldst->opType() == LoadStoreOpType::Set &&
      ldst->in()->isA<TensorView>();
}

//! This defines the types of operations that are eligible for simplification in
//! this pass.
bool isReplaceableExpr(Expr* expr) {
  if (expr == nullptr) {
    return false;
  }
  return expr->isA<BroadcastOp>() || expr->isA<SqueezeOp>() ||
      isSimpleTVSet(expr);
}

//! Convert a LoadStoreOp to an AxisOps of all PRESERVE ops
AxisOps setToAxisOps(LoadStoreOp* ldst) {
  NVF_ERROR(isSimpleTVSet(ldst));
  return AxisOps(
      ldst->in()->as<TensorView>()->getLogicalDomain().size(),
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

//! Return true if we are unable to simplify this combination to a single
//! operation.
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
  // Preserve indicates this is a set op
  return AxisOp::PRESERVE;
}

//! This is useful for getting the axis flags needed to create a new BroadcastOp
//! or SqueezeOp.
std::vector<bool> nonPreservedDims(const AxisOps& ops) {
  std::vector<bool> flags;
  flags.reserve(ops.size());
  for (size_t i : c10::irange(ops.size())) {
    flags.push_back(ops[i] != AxisOp::PRESERVE);
  }
  return flags;
}

//! Given a descriptors of two sequences of broadcast+squeeze ops, return a
//! descriptor of their composition, with a applied before b
AxisOps composeOps(const AxisOps& prev, const AxisOps& next) {
  // Find dims present after prev (i.e. non-SQUEEZE dims)

  // Add ops from next. Positions in next are relative to the present dims from
  // prev.
  //   If the next op is a SQUEEZE then
  //      If there is an adjacent BROADCAST, remove it
  //      Otherwise, insert the SQUEEZE
  //   If the next op is a BROADCAST then
  //      If there is an adjacent SQUEEZE, replace it with PRESERVE
  //      Otherwise, insert the BROADCAST

  size_t prev_pos = 0, next_pos = 0;
  AxisOps ops;
  // op_from_prev tracks whether the inserted op was from the previous ops or
  // next. This is used in simplification so that we can determine whether to
  // insert a PRESERVE (for squeeze then broadcast) vs remove the ops altogether
  // (for broadcast then squeeze).
  std::vector<bool> op_from_prev;
  const auto pushOp = [&ops, &op_from_prev](AxisOp op, bool from_prev) {
    ops.push_back(op);
    op_from_prev.push_back(from_prev);
  };
  while (prev_pos < prev.size() && next_pos < next.size()) {
    AxisOp prev_op = prev[prev_pos];
    while (prev_op == AxisOp::SQUEEZE) {
      // SQUEEZE is the only op that should not increment next_pos, since there
      // is no corresponding output axis
      pushOp(AxisOp::SQUEEZE, true);
      ++prev_pos;
      if (prev_pos >= prev.size()) {
        break;
      }
      prev_op = prev[prev_pos];
    }
    AxisOp next_op = next[next_pos];
    while (next_op == AxisOp::BROADCAST) {
      pushOp(AxisOp::BROADCAST, false);
      ++next_pos;
      if (next_pos >= next.size()) {
        break;
      }
      next_op = next[next_pos];
    }

    pushOp(prev_op, true);
    if (next_op != AxisOp::PRESERVE) {
      pushOp(next_op, false);
    }

    ++prev_pos;
    ++next_pos;
  }
  while (prev_pos < prev.size()) {
    NVF_ERROR(
        prev[prev_pos] == AxisOp::SQUEEZE,
        "Left-over previous ops must be squeeze");
    pushOp(AxisOp::SQUEEZE, true);
    ++prev_pos;
  }
  while (next_pos < next.size()) {
    NVF_ERROR(
        next[next_pos] == AxisOp::BROADCAST,
        "Left-over previous ops must be broadcast");
    pushOp(AxisOp::BROADCAST, false);
    ++next_pos;
  }
  NVF_ERROR(prev_pos == prev.size());
  NVF_ERROR(next_pos == next.size());

  // Simplify
  while (true) {
    bool changed = false;
    NVF_ERROR(ops.size() == op_from_prev.size());
    for (size_t i : c10::irange(ops.size() - 1)) {
      AxisOp cur = ops[i], next = ops[i + 1];
      if ((cur == AxisOp::SQUEEZE && next == AxisOp::BROADCAST) ||
          (cur == AxisOp::BROADCAST && next == AxisOp::SQUEEZE)) {
        if (op_from_prev[i]) {
          ops[i] = AxisOp::PRESERVE;
          ops.erase(ops.begin() + i + 1);
          op_from_prev.erase(op_from_prev.begin() + i + 1);
          changed = true;
          break;
        } else {
        }
      } else if (cur == AxisOp::BROADCAST && next == AxisOp::SQUEEZE) {
        // ops.erase(next);
        // ops.erase(cur);
        changed = true;
        break;
      }
    }
    if (!changed) {
      break;
    }
  }

  return ops;
}

void maybeDoReplacement(Expr* first, Expr* second) {
  AxisOps simplified_ops =
      composeOps(exprToAxisOps(first), exprToAxisOps(second));
  std::optional<AxisOp> simple_op_type_opt =
      getSimplifiedOpType(simplified_ops);
  if (simple_op_type_opt.has_value()) {
    TensorView* input_tv = first->input(0)->as<TensorView>();
    Val* orig = second->output(0);
    Val* replacement = nullptr;
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
    NVF_ERROR(replacement != nullptr);
    ir_utils::replaceValInAllExprInputsAndFusionOutputs(orig, replacement);
  }
}

// Remove broadcast-squeeze and squeeze-broadcast patterns
// TODO: still remove when have intermediate ops between broadcast and squeeze
void removeBcastSqueeze(Fusion* fusion) {
  // Iterate backwards over fusion expressions.
  // This will ensure we don't process expressions that are no longer valid
  // after replacement.
  auto exprs = fusion->exprs();
  for (auto it = std::rbegin(exprs); it != std::rend(exprs); it++) {
    Expr* expr = *it;
    if (isReplaceableExpr(expr)) {
      Expr* prev_expr = expr->input(0)->definition();
      if (isReplaceableExpr(prev_expr)) {
        maybeDoReplacement(prev_expr, expr);
      }
    }
  }
}

} // namespace

void RemoveBcastSqueeze::runPass(Fusion* fusion) {
  removeBcastSqueeze(fusion);
}

} // namespace nvfuser::preseg_passes
