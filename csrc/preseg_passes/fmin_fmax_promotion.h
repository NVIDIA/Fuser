// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <optimization_pass.h>

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
// Discovering these cases using full IterDomain analysis is quite complicated,
// instead of doing that, this pass currently relies on a couple of restrictions
// to achieve simpler, conservative analysis.
// This means we won't always do the promotion, even when it is safe to do so.
//
// The restrictions work like this:
// 1. We form a subgraph around the "target" min/max reduction. This subgraph
//    only contains a few operators: UnaryOp, BinaryOp, ReductionOp BroadcastOp.
//    Any other expression types are considered "outputs" of the subgraph.
// 2. Restrictions on reduction and broadcast shapes. All reductions in the
//    subgraph must match the target reduction shape. Likewise, all broadcasts
//    must broadcast between the same shapes (if there are any in the subgraph).
//    These restrictions ensure that data doesn't move around domains in ways
//    which would require more fine-grained tracking.
// 3. Restricted subgraph-input analysis. We start from the input of the target
//    ReductionOp, and we do not look any further upstream. This means we
//    conservatively reject the following example where a repair happens,
//    although the data comes from a distant parent:
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
