// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

//! UnsafeReducePass
class UnsafeReducePass : public OptimizationPass<UnsafeReducePass> {
  friend class OptimizationPass<UnsafeReducePass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "UnsafeReducePass";
  }
};

} // namespace nvfuser::preseg_passes
