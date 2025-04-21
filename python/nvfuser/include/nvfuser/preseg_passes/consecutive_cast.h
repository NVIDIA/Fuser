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

//! ConsecutiveCastPass removes redundant consecutive cast operations that
//! doesn't have any impact on output from fusion.
class ConsecutiveCastPass : public OptimizationPass<ConsecutiveCastPass> {
  friend class OptimizationPass<ConsecutiveCastPass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "ConsecutiveCastPass";
  }
};

} // namespace nvfuser::preseg_passes
