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

// Prepares the input fusion for marking aliases. It currently updates layouts
// to enable aliases, and inserts `SegmenterSet`s so segmentation will separate
// out alias-only regions.
class MarkAliasesPreparePass : public OptimizationPass<MarkAliasesPreparePass> {
  friend class OptimizationPass<MarkAliasesPreparePass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "MarkAliasesPreparePass";
  }
};

} // namespace nvfuser::preseg_passes
