// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

// Cuda has fmax() and fmin() functions that are faster than our max() and min()
// helper functions. However fmax and fmin don't propagate NAN values, so we
// must generally use max() and min() in kernels (Pytorch behavior is to
// propagate NANs).
//
// Normalization is a common fusion pattern whereby a max or min reduction is
// followed by a sum reduction, and the result is combined with a binary op.
// For example:
//
// tv1 = max(tv0, {0, 1})
// tv2 = broadcast(tv1, {true, true})
// tv3 = add(tv2, tv0)
// tv4 = sum(tv3, {0, 1})
// tv5 = broadcast(tv4, {true, true})
// tv6 = add(tv5, tv3)
//
// For this example, it would actually be OK if the max was performed using
// fmax, because the NAN values will flow into the sum(), and will be combined
// during the final add(). Any loss of NANs that occur at fmax will be repaired,
// assuming tv1 through tv5 aren't consumed elsewhere in the fusion.
//
// The purpose of this pass is to identify situations like this, and "promote"
// max and min ops into fmax and fmin where possible.
//
// The scope of this analysis can be quite large, and could e.g. apply to
// pointwise min/max as well as reductions. However this pass currently only
// targets normalization-style cases, so the promotion algorithm is simplified
// with the following restrictions:
//
// 1. Only promotes min() and max() reduction ops, not pointwise min and max.
// 2. Analyzes a restricted subgraph around the "target" min/max reduction.
//    Specifically, we only analyze UnaryOp, BinaryOp, ReductionOp and
//    BroadcastOp.
// 3. Limited support for reduction and broadcast axes. All ReductionOps in the
//    subgraph must match the target reduction axes. Likewise with BroadcastOps,
//    if we encounter a single broadcast, it becomes the structure that all
//    broadcasts in the subgraph must conform to. This simplifies the analysis
//    and avoids the need for a complicated IterDomain propagation and
//    interaction tracker.
// 4. Restricted subgraph-input analysis. We start from the input of the target
//    ReductionOp, and we do not look any further upstream. This means we
//    conservatively reject the following example:
//      tv1 = abs(tv0)
//      tv2 = max(tv1, {0})
//      tv3 = sum(tv0, {0})
//      tv4 = add(tv2, tv3)
class FMinFMaxPromotionPass : public OptimizationPass<FMinFMaxPromotionPass> {
  friend class OptimizationPass<FMinFMaxPromotionPass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "FMinFMaxPromotionPass";
  }
};

} // namespace nvfuser::preseg_passes
