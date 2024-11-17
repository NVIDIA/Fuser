// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

//! Propagates the CPU scalar flag set on any fusion inputs to outputs
class PropagateCpuScalarsPass
    : public OptimizationPass<PropagateCpuScalarsPass> {
  friend class OptimizationPass<PropagateCpuScalarsPass>;

 protected:
  static void runPass(Fusion* fusion);
  static std::string name() {
    return "PropagateCpuScalars";
  }
};

} // namespace nvfuser::preseg_passes
