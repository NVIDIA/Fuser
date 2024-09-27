// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <options.h>

#include <preseg_passes/pre_segmenter.h>

#include <instrumentation.h>
#include <preseg_passes/add_axioms.h>
#include <preseg_passes/allocation_order_inference.h>
#include <preseg_passes/consecutive_cast.h>
#include <preseg_passes/exact_mapped_extent_substitution.h>
#include <preseg_passes/insert_reshardings.h>
#include <preseg_passes/make_resharding_contiguous.h>
#include <preseg_passes/mark_aliases_prepare.h>
#include <preseg_passes/move_split_cat.h>
#include <preseg_passes/propagate_shardings.h>
#include <preseg_passes/remove_bcast_squeeze.h>
#include <preseg_passes/remove_empty.h>
#include <preseg_passes/reorder_sharded_axis.h>

namespace nvfuser::preseg_passes {

/*static*/ void PreSegmenter::runPass(Fusion* fusion) {
  FUSER_PERF_SCOPE("PreSegmenter::runPass");

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion before " << name() << ":" << std::endl;
    fusion->printMath();
    debug() << "========================================" << std::endl;
  }

  // For resharding across GPUs.
  OptimizationPass<PropagateShardingsPass>::runPass(fusion);
  OptimizationPass<InsertReshardingsPass>::runPass(fusion);
  OptimizationPass<ReorderShardedAxisPass>::runPass(fusion);
  OptimizationPass<MakeReshardingContiguousPass>::runPass(fusion);

  // Replace TensorViews with zero extent. Outputs and inputs may still be empty
  OptimizationPass<RemoveEmptyPass>::runPass(fusion);
  // removes consecutive cast operations
  OptimizationPass<ConsecutiveCastPass>::runPass(fusion);
  OptimizationPass<AddAxiomsPass>::runPass(fusion);
  OptimizationPass<MoveSplitCatPass>::runPass(fusion);
  OptimizationPass<MarkAliasesPreparePass>::runPass(fusion);
  OptimizationPass<ExactMappedExtentSubstitutionPass>::runPass(fusion);
  OptimizationPass<AllocationDomainPass>::runPass(fusion);
  OptimizationPass<RemoveBcastSqueeze>::runPass(fusion);
}

} // namespace nvfuser::preseg_passes
