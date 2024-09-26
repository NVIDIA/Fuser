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
AxisOps broadcastToOps(BroadcastOp* bcast) {
  AxisOps ops;
  const std::vector<bool>& flags = bcast->getBroadcastDimFlags();
  ops.reserve(flags.size());
  for (bool flag : flags) {
    ops.push_back(flag ? AxisOp::BROADCAST : AxisOp::PRESERVE);
  }
  return ops;
}

//! Convert a squeeze Expr to an AxisOps descriptor
AxisOps squeezeToOps(SqueezeOp* squeeze) {
  AxisOps ops;
  const std::vector<bool>& flags = squeeze->getSqueezeDimFlags();
  ops.reserve(flags.size());
  for (bool flag : flags) {
    ops.push_back(flag ? AxisOp::SQUEEZE : AxisOp::PRESERVE);
  }
  return ops;
}

AxisOps exprToOps(Expr* expr) {
  if (auto* squeeze = dynamic_cast<SqueezeOp*>(expr)) {
    return squeezeToOps(squeeze);
  } else if (auto* bcast = dynamic_cast<BroadcastOp*>(expr)) {
    return broadcastToOps(bcast);
  }
  NVF_THROW(
      "exprToOps expects BroadcastOp or SqueezeOp. Found ", expr->toString());
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
  AxisOps out;
  while (prev_pos < prev.size() && next_pos < next.size()) {
    AxisOp prev_op = prev[prev_pos];
    while (prev_op == AxisOp::SQUEEZE) {
      // SQUEEZE is the only op that should not increment next_pos, since there
      // is no corresponding output axis
      out.push_back(AxisOp::SQUEEZE);
      ++prev_pos;
      if (prev_pos >= prev.size()) {
        break;
      }
      prev_op = prev[prev_pos];
    }
    AxisOp next_op = next[next_pos];
    while (next_op == AxisOp::BROADCAST) {
      out.push_back(AxisOp::BROADCAST);
      ++next_pos;
      if (next_pos >= next.size()) {
        break;
      }
      next_op = next[next_pos];
    }

    out.push_back(prev_op);
    if (next_op != AxisOp::PRESERVE) {
      out.push_back(next_op);
    }

    ++prev_pos;
    ++next_pos;
  }
  while (prev_pos < prev.size()) {
    NVF_ERROR(
        prev[prev_pos] == AxisOp::SQUEEZE,
        "Left-over previous ops must be squeeze");
    out.push_back(AxisOp::SQUEEZE);
    ++prev_pos;
  }
  while (next_pos < next.size()) {
    NVF_ERROR(
        next[next_pos] == AxisOp::BROADCAST,
        "Left-over previous ops must be broadcast");
    out.push_back(AxisOp::BROADCAST);
    ++next_pos;
  }
  NVF_ERROR(prev_pos == prev.size());
  NVF_ERROR(next_pos == next.size());
  return out;
}

//! Simplifies a composed op by iteratively removing adjacent SQUEEZE and
//! BROADCAST ops>
AxisOps simplifyOps(AxisOps ops) {
  while (true) {
    bool changed = false;
    for (auto cur = ops.begin(), next = ops.begin() + 1; next != ops.end();
         cur++, next++) {
      // In cases like this, we will have inserted the previous op to the left
      // of the next op.
      // TODO: prove the above. Is it true?
      if (*cur == AxisOp::SQUEEZE && *next == AxisOp::BROADCAST) {
        *cur = AxisOp::PRESERVE;
        ops.erase(next);
        changed = true;
        break;
      } else if (*cur == AxisOp::BROADCAST && *next == AxisOp::SQUEEZE) {
        ops.erase(next);
        ops.erase(cur);
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
  AxisOps composed_ops = composeOps(exprToOps(first), exprToOps(second));
  AxisOps simplified_ops = simplifyOps(composed_ops);
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
    // step-1: find and remove broadcast + squeeze pattern
    // before: Y = broadcast(X); Z = squeeze(Y);  M = someOp(Z)
    // after : M = someOp(X)
    // conditions: (1) broadcast and squeeze have the same dim flags
    if (auto squeeze = dynamic_cast<SqueezeOp*>(expr)) {
      if (auto bcast =
              dynamic_cast<BroadcastOp*>(squeeze->in()->definition())) {
        maybeDoReplacement(bcast, squeeze);
      }
    }

    // step-2: find and remove squeeze + broadcast pattern
    // before: Y = squeeze(X); Z = broadcast(Y);  M = someOp(Z)
    // after : M = someOp(X)
    // conditions: (1) broadcast and squeeze have the same dim flags
    if (auto bcast = dynamic_cast<BroadcastOp*>(expr)) {
      if (auto squeeze = dynamic_cast<SqueezeOp*>(bcast->in()->definition())) {
        maybeDoReplacement(squeeze, bcast);
      }
    }
  }
}

} // namespace

void RemoveBcastSqueeze::runPass(Fusion* fusion) {
  removeBcastSqueeze(fusion);
}

} // namespace nvfuser::preseg_passes
