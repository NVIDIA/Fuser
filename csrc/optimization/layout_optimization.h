// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/optimization_pass.h>

namespace nvfuser::optimization {

class LayoutOptimizationPass : public OptimizationPass<LayoutOptimizationPass> {
  friend class OptimizationPass<LayoutOptimizationPass>;

 protected:
  static void runPass(Fusion* fusion);
};

} // namespace nvfuser::optimization
