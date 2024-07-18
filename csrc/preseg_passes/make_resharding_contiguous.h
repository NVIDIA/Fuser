// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <preseg_passes/optimization_pass.h>

#include <string>

#include <fusion.h>

namespace nvfuser::preseg_passes {

// Resharding expressions are mapped to collective libraries which expect
// contiguous tensors and output contiguous buffers. This pass checks that
// inputs are contiguous and sets the allocation domain of inputs and outputs of
// all resharding expressions. This pass should run after all passes that add or
// update resharding expressions.
class MakeReshardingContiguousPass
    : public OptimizationPass<MakeReshardingContiguousPass> {
  friend class OptimizationPass<MakeReshardingContiguousPass>;

 protected:
  static void runPass(Fusion* fusion);
  static std::string name() {
    return "MakeReshardingContiguousPass";
  }
};

} // namespace nvfuser::preseg_passes
