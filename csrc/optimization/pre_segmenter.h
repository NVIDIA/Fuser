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

//! PreSegmenter is an optimization group that runs right before fusion executor
//! segments a fusion into multiple kernels.
class PreSegmenter : public OptimizationPass<PreSegmenter> {
  friend class OptimizationPass<PreSegmenter>;

 protected:
  static void runPass(Fusion* fusion);
};

} // namespace nvfuser::optimization
