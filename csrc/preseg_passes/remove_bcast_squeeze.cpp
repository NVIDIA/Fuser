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
#include <options.h>
#include <preseg_passes/remove_bcast_squeeze.h>

namespace nvfuser::preseg_passes {

namespace {

enum class AxisOp { PRESERVE, SQUEEZE, BROADCAST };

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
std::vector<int64_t> nonPreservedDims(const AxisOps& ops) {
  std::vector<int64_t> dims;

  for (size_t i : c10::irange(ops.size())) {
    if (ops[i] != AxisOp::PRESERVE) {
      dims.push_back((int64_t)i);
    }
  }

  return dims;
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
  while (prev_pos < prev.size() || next_pos < next.size()) {
    if (prev_pos < prev.size()) {
      AxisOp prev_op = prev[prev_pos];
      if (next_pos < next.size()) {
        AxisOp next_op = next[next_pos];
        switch (next_op) {
          case AxisOp::PRESERVE:
            out.push_back(prev_op);
            prev_pos++;
            next_pos++;
            break;
          case AxisOp::SQUEEZE:
            switch (prev_op) {
              case AxisOp::PRESERVE:
                out.push_back(AxisOp::SQUEEZE);
                prev_pos++;
                next_pos++;
                break;
            }
            out.push_back(prev_op);
            prev_pos++;
            next_pos++;
            break;
        }
      } else {
        NVF_ERROR(
            prev_op == AxisOp::SQUEEZE,
            "Found non-squeeze prev op but ran out of next ops");
        prev_pos++;
      }
    } else {
      AxisOp next_op = next[next_pos];
      NVF_ERROR(
          next_op == AxisOp::BROADCAST,
          "Found non-broadcast next op but ran out of previous non-squeeze ops");
      out.push_back(next_op);
      next_pos++;
    }
  }
  return out;
}

void maybeDoReplacement(Expr* first, Expr* second) {
  AxisOps composed_ops = composeOps(exprToOps(first), exprToOps(second));
  std::optional<AxisOp> simple_op_type_opt = getSimplifiedOpType(composed_ops);
  if (simple_op_type_opt.has_value()) {
    switch (simple_op_type_opt.value()) {
      TensorView* input_tv = first->input(0)->as<TensorView>();
      Val* orig = second->output(0);
      Val* replacement = nullptr;
      case AxisOp::PRESERVE:
        // This is equivalent to a set Op
        replacement = input_tv;
        break;
      case AxisOp::SQUEEZE:
        replacement = squeeze(input_tv, nonPreservedDims(composed_ops));
        break;
      case AxisOp::BROADCAST:
        replacement = broadcast(input_tv, nonPreservedDims(composed_ops));
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
