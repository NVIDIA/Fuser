// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/optimization_pass.h>
#include <visibility.h>

namespace nvfuser::preseg_passes {

// Prepares the input fusion for marking aliases. It currently updates layouts
// to enable aliases, and inserts `SegmenterSet`s so segmentation will separate
// out alias-only regions.
class MarkAliasesPreparePass : public OptimizationPass<MarkAliasesPreparePass> {
  friend class OptimizationPass<MarkAliasesPreparePass>;

 protected:
  NVF_API static void runPass(Fusion* fusion);
};

} // namespace nvfuser::preseg_passes
