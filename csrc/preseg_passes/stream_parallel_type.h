// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

class StreamParallelType : public OptimizationPass<StreamParallelType> {
  friend class OptimizationPass<StreamParallelType>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "StreamParallelType";
  }
};

} // namespace nvfuser::preseg_passes
