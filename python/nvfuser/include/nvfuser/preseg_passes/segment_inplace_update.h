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

//! RemoveEmptyPass removes intermediate empty tensors (those with at least one
//! extent zero thar are neither a fusion output or input).
class SegmentInplaceUpdatePass
    : public OptimizationPass<SegmentInplaceUpdatePass> {
  friend class OptimizationPass<SegmentInplaceUpdatePass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "SegmentInplaceUpdate";
  }
};

} // namespace nvfuser::preseg_passes
