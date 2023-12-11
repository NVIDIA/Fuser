// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/pre_segmenter.h>

#include <optimization/add_axioms.h>
#include <optimization/consecutive_cast.h>
#include <optimization/optimize_layout.h>
#include <optimization/remove_empty.h>

namespace nvfuser::optimization {

void PreSegmenter::runPass(Fusion* fusion) {
  // Replace TensorViews with zero extent. Outputs and inputs may still be empty
  OptimizationPass<RemoveEmptyPass>::runPass(fusion);
  // removes consecutive cast operations
  OptimizationPass<ConsecutiveCastPass>::runPass(fusion);
  OptimizationPass<AddAxiomsPass>::runPass(fusion);
  OptimizationPass<OptimizeLayoutPass>::runPass(fusion);
}

} // namespace nvfuser::optimization
