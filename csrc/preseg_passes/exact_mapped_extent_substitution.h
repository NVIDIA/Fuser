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

// Fusion may have tensors with const extents and symbolic extents. This pass
// replaces symbolic extents with const extents if they are mapped to the exact
// same root domain set. See https://github.com/NVIDIA/Fuser/issues/1590.
// Additionaly, if there is no const extent, it replaces all symbolic extents
// with the one with the lowest name. This could simplify some cases where we
// recompute expressions inside the kernel that are known to be equal, even if
// they are not constant.
class ExactMappedExtentSubstitutionPass
    : public OptimizationPass<ExactMappedExtentSubstitutionPass> {
  friend class OptimizationPass<ExactMappedExtentSubstitutionPass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "ExactMappedExtentSubstitutionPass";
  }
};

} // namespace nvfuser::preseg_passes
