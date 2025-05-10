// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

// A pre-segmenter optimization that moves gather operations ahead of producer
// unary pointwise ops such as cast.
class MoveGatherPass : public OptimizationPass<MoveGatherPass> {
  friend class OptimizationPass<MoveGatherPass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "MoveGatherPass";
  }
};

} // namespace nvfuser::preseg_passes
