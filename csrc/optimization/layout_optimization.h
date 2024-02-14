// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/optimization_pass.h>

namespace nvfuser::optimization {

// Realize MemoryFormat propagation on fusion inputs to optimize MemoryFormat of
// output tensor. This optimization pass currently only applies to fusion
// outputs, but not intermediate tensors.
class LayoutOptimizationPass : public OptimizationPass<LayoutOptimizationPass> {
  friend class OptimizationPass<LayoutOptimizationPass>;

 protected:
  static void runPass(Fusion* fusion);
};

} // namespace nvfuser::optimization
