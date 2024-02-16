// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/pre_segmenter.h>

#include <preseg_passes/add_axioms.h>
#include <preseg_passes/consecutive_cast.h>
#include <preseg_passes/exact_mapped_extent_substitution.h>
#include <preseg_passes/mark_aliases_prepare.h>
#include <preseg_passes/remove_empty.h>

namespace nvfuser::optimization {

void PreSegmenter::runPass(Fusion* fusion) {
  // Replace TensorViews with zero extent. Outputs and inputs may still be empty
  OptimizationPass<RemoveEmptyPass>::runPass(fusion);
  // removes consecutive cast operations
  OptimizationPass<ConsecutiveCastPass>::runPass(fusion);
  OptimizationPass<AddAxiomsPass>::runPass(fusion);
  OptimizationPass<MarkAliasesPreparePass>::runPass(fusion);
  OptimizationPass<ExactMappedExtentSubstitutionPass>::runPass(fusion);
}

} // namespace nvfuser::optimization
