// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/pre_segmenter.h>

#include <instrumentation.h>
#include <preseg_passes/add_axioms.h>
#include <preseg_passes/allocation_order_inference.h>
#include <preseg_passes/consecutive_cast.h>
#include <preseg_passes/exact_mapped_extent_substitution.h>
#include <preseg_passes/mark_aliases_prepare.h>
#include <preseg_passes/move_split_cat.h>
#include <preseg_passes/remove_bcast_squeeze.h>
#include <preseg_passes/remove_empty.h>

namespace nvfuser::preseg_passes {

/*static*/ void PreSegmenter::runPass(Fusion* fusion) {
  FUSER_PERF_SCOPE("PreSegmenter::runPass");
  using PassFunction = std::function<void(Fusion*)>;
  using TaggedPass = std::pair<PassFunction, std::string>;
  std::vector<TaggedPass> passes = {
      {&OptimizationPass<RemoveEmptyPass>::runPass, "RemoveEmptyPass"},
      {&OptimizationPass<ConsecutiveCastPass>::runPass, "ConsecutiveCastPass"},
      {&OptimizationPass<AddAxiomsPass>::runPass, "AddAxiomsPass"},
      {&OptimizationPass<MoveSplitCatPass>::runPass, "MoveSplitCatPass"},
      {&OptimizationPass<MarkAliasesPreparePass>::runPass,
       "MarkAliasesPreparePass"},
      {&OptimizationPass<ExactMappedExtentSubstitutionPass>::runPass,
       "ExactMappedExtentSubstitutionPass"},
      {&OptimizationPass<AllocationDomainPass>::runPass,
       "AllocationDomainPass"},
      {&OptimizationPass<RemoveBcastSqueeze>::runPass, "RemoveBcastSqueeze"}};

  bool is_log_enabled =
      isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging);
  if (is_log_enabled) {
    debug() << "Fusion before PreSegmenter:" << std::endl;
    fusion->printMath();
  }
  for (auto [pass_fun, pass_tag] : passes) {
    pass_fun(fusion);
    if (is_log_enabled) {
      debug() << "Fusion after pre-segmenter pass: " << pass_tag << std::endl;
      fusion->printMath();
    }
  }
}

} // namespace nvfuser::preseg_passes
