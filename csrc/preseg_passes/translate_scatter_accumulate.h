// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

// Look for the pattern like below:
//
// index = [m, 1];
// scatter_inp = zeros([m, n]); // [m, n]
// scatter_out = scatter(scatter_inp, 1, index, src=1);
// reduction_out = scatter_out.sum(0); // [n]
//
// It will be translated as:
//
// index = index.squeeze(-1) // [m]
// scatter_inp = zeros([n]); // [n]
// scatter_out = scatter(scatter_inp, 0, index, src=1, BinaryOpType::Add)
//
// To find if a ScatterOp is a candidate for the above
// scatter-accumulate pattern, the following conditions are checked:
//
// - Scatter input must be 2D
// - Scatter is not scatter-accumulate
// - All ops must be exclusively used by the ops of this pattern
// except for the last output
// - Full op initializes the scatter input using the same value as
// the reduction init value
// - Scatter-accumulate must be deterministic, which means, for
// example, the data type must be integer
// - The scatter dimension of the index tensor must be a broadcast
//
// Additionally, upcast may be automatically inserted between
// scatter and reduction. Skip upcast if any
class TranslateScatterAccumulate
    : public OptimizationPass<TranslateScatterAccumulate> {
  friend class OptimizationPass<TranslateScatterAccumulate>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "TranslateScatterAccumulate";
  }
};

} // namespace nvfuser::preseg_passes
