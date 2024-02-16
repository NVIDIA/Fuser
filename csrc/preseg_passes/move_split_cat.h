// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

// A pre-segmenter optimization that moves splits and cats for better codegen.
class MoveSplitCatPass : public OptimizationPass<MoveSplitCatPass> {
  friend class OptimizationPass<MoveSplitCatPass>;

 protected:
  static void runPass(Fusion* fusion);
};

} // namespace nvfuser::preseg_passes
