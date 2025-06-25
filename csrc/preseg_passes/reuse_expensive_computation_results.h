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

class ReuseExpensiveComputationResultsPass
    : public OptimizationPass<ReuseExpensiveComputationResultsPass> {
  friend class OptimizationPass<ReuseExpensiveComputationResultsPass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "ReuseExpensiveComputationResultsPass";
  }
};

} // namespace nvfuser::preseg_passes
