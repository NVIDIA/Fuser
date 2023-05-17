// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/opt_pass.h>

namespace nvfuser::optimization {

namespace {

class ConsecutiveCastPass : OptimizationPass {
 public:
  static void runPass(Fusion* fusion) {
    std::cout << "running optimization pass on fusion: " << std::endl;
    fusion->printMath();
  }
  std::string name() { return "ConsecutiveCastOptimization"; }

  ConsecutiveCastPass() {
   registerOptimizationPass(OptimizationPassCategory::PreSegmenter, *this);
  }
};

static Register register;

} // namespace nvfuser::optimization
