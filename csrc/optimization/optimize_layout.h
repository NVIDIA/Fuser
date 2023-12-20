// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/optimization_pass.h>

namespace nvfuser::optimization {

// Updates layouts to enable aliases.
// TODO(wujingyue): Rename. It also inserts segment_set to help segmentation.
class OptimizeLayoutPass : public OptimizationPass<OptimizeLayoutPass> {
  friend class OptimizationPass<OptimizeLayoutPass>;

 protected:
  static void runPass(Fusion* fusion);
};

} // namespace nvfuser::optimization
