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

// Converts max & min reductions into faster unsafe versions, which don't
// propagate NANs.
//
// Pytorch propagates NANs for min and max reductions. However, fmax and fmin
// do not propagate NANs in Cuda. So for a kernel to match pytorch behavior, it
// must contain additional branches which are expensive. Other ops such as sum()
// propagate NANs by default, with no loss in performance. Then take for
// example:
//
// tv1 = max(tv0, {0});
// tv2 = sum(tv0, {0});
// tv3 = add(tv1, tv2);
//
// Here, if max() fails to propagate NANs, it will be "repaired" by the
// downstream sum() reduction. This can also work if there are matching
// broadcasts following the reductions.
//
// The analysis can get quite complicated, For example, it might take a
// combination of many safe reductions to patch up an unsafe one. However since
// we are primarily targeting softmax style ops, the current algorithm has
// these scoping assumptions:
//
// 1. We only analyze downstream from the unsafe-reductions direct input. No
//    repairs from combinations of distant parents.
// 2. We do limited shape analysis. The safe reduction must be a superset of the
//    unsafe reductions axes. We only support a single broadcast following the
//    safe & unsafe reductions.
// 3. If we hit any node for which NAN movements aren't trivial, we cancel the
//    analysis for a given unsafe target.

class UnsafeReducePass : public OptimizationPass<UnsafeReducePass> {
  friend class OptimizationPass<UnsafeReducePass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "UnsafeReducePass";
  }
};

} // namespace nvfuser::preseg_passes
