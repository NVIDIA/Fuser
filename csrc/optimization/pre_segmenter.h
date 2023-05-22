// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <optimization/optimization_pass.h>

namespace nvfuser::optimization {

class TORCH_CUDA_CU_API PreSegmenter : public OptimizationGroup<PreSegmenterOptimizationPass> {
 public:
  static void runPass(Fusion* fusion);
};

} // namespace nvfuser::optimization
