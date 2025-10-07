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

// Converts max & min reductions into faster versions,
// which don't propagate NANs.
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
class FMinFMaxPromotionPass : public OptimizationPass<FMinFMaxPromotionPass> {
  friend class OptimizationPass<FMinFMaxPromotionPass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "FMinFMaxPromotionPass";
  }
};

} // namespace nvfuser::preseg_passes
