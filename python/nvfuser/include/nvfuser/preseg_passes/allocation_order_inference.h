// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <unordered_map>

#include <fusion.h>
#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

// Realize allocation order propagation on fusion inputs to optimize allocation
// domain of output tensor. This optimization pass currently only applies to
// fusion outputs, but not intermediate tensors.
class AllocationDomainPass : public OptimizationPass<AllocationDomainPass> {
  friend class OptimizationPass<AllocationDomainPass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "AllocationDomainPass";
  }
};

} // namespace nvfuser::preseg_passes
