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

namespace nvfuser::preseg_passes {

// An pre-segmenter pass that decomposes resharding expressions into a series of
// Exprs that are either compute only or directly lowerable to communication.
class DecomposeReshardingsPass
    : public OptimizationPass<DecomposeReshardingsPass> {
  friend class OptimizationPass<DecomposeReshardingsPass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "DecomposeReshardingsPass";
  }
};

} // namespace nvfuser::preseg_passes
