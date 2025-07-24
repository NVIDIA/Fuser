// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

// A pre-segmenter optimization that moves pad operation across its producers
class MovePadPass : public OptimizationPass<MovePadPass> {
  friend class OptimizationPass<MovePadPass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "MovePadPass";
  }
};

} // namespace nvfuser::preseg_passes
