// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <preseg_passes/optimization_pass.h>
#include <visibility.h>

namespace nvfuser::preseg_passes {

//! PreSegmenter is an optimization group that runs right before fusion executor
//! segments a fusion into multiple kernels.
class NVF_API PreSegmenter : public OptimizationPass<PreSegmenter> {
  friend class OptimizationPass<PreSegmenter>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "PreSegmenter";
  }
};

} // namespace nvfuser::preseg_passes
