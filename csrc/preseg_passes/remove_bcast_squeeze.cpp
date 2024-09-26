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

//! Return true if we are unable to simplify this combination to a single
//! operation.
bool isTrivial(const AxisOps& ops) {
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
  return !has_broadcast || !has_squeeze;
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
        if (bcast->getBroadcastDimFlags() == squeeze->getSqueezeDimFlags()) {
          ir_utils::replaceValInAllExprInputsAndFusionOutputs(
              squeeze->out(), bcast->in());
        }
      }
    }

    // step-2: find and remove squeeze + broadcast pattern
    // before: Y = squeeze(X); Z = broadcast(Y);  M = someOp(Z)
    // after : M = someOp(X)
    // conditions: (1) broadcast and squeeze have the same dim flags
    if (auto bcast = dynamic_cast<BroadcastOp*>(expr)) {
      std::cout << "Found bcast " << bcast->toString() << "  "
                << bcast->getBroadcastDimFlags() << std::endl;
      if (auto squeeze = dynamic_cast<SqueezeOp*>(bcast->in()->definition())) {
        std::cout << "    Found squeeze " << squeeze->toString() << "  "
                  << squeeze->getSqueezeDimFlags() << std::endl;
        if (bcast->getBroadcastDimFlags() == squeeze->getSqueezeDimFlags()) {
          std::cout << "        REPLACING bcast out with squeeze in"
                    << std::endl;
          ir_utils::replaceValInAllExprInputsAndFusionOutputs(
              bcast->out(), squeeze->in());
        } else {
          std::cout << "        NOT REPLACING bcast out with squeeze in"
                    << std::endl;
        }
      }
    }
  }
}

} // namespace

void RemoveBcastSqueeze::runPass(Fusion* fusion) {
  removeBcastSqueeze(fusion);
}

} // namespace nvfuser::preseg_passes
