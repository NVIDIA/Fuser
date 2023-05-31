// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/optimization_pass.h>

namespace nvfuser::optimization {

//! ConsecutiveCastPass removes redundant consecutive cast operations that
//! doesn't have any impact on output from fusion.
class TORCH_CUDA_CU_API ConsecutiveCastPass
    : public OptimizationPass<ConsecutiveCastPass> {
 public:
  static void runPass(Fusion* fusion);
};

} // namespace nvfuser::optimization
