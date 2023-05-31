// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/pre_segmenter.h>

#include <optimization/consecutive_cast.h>

namespace nvfuser::optimization {

void PreSegmenter::runPass(Fusion* fusion) {
  // removes consecutive cast operations
  OptimizationPass<ConsecutiveCastPass>::runPass(fusion);
}

} // namespace nvfuser::optimization
