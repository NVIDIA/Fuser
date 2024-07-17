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
      if (auto squeeze = dynamic_cast<SqueezeOp*>(bcast->in()->definition())) {
        if (bcast->getBroadcastDimFlags() == squeeze->getSqueezeDimFlags()) {
          ir_utils::replaceValInAllExprInputsAndFusionOutputs(
              bcast->out(), squeeze->in());
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
