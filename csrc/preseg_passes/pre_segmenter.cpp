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
#include <preseg_passes/move_pad.h>
#include <preseg_passes/move_split_cat.h>
#include <preseg_passes/propagate_shardings.h>
#include <preseg_passes/remove_bcast_squeeze.h>
#include <preseg_passes/remove_empty.h>
#include <preseg_passes/reorder_sharded_axis.h>
#include <preseg_passes/segment_inplace_update.h>

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
  // MovePadPass needs to happen:
  // 1. before MarkAliasPrepare; and
  //    avoid moving pad operatoins around, which could disturb the analysis
  //    from MarkAliasPrepare
  // 2. after MoveSplitCat
  //    to avoid this pass moving PadOp around to break the
  // MoveSplitCat.
  if (getenv("MOVE_PAD")) {
    OptimizationPass<MovePadPass>::runPass(fusion);
  }
  // NOTE vvv this doesn't really work, since our type promotion to higher
  // precision for Add cannot be canceled out with previous cast to lower
  // precision. Since it's not an no-op and it has a quantization effect. I'll
  // open an issue for this and see if we want to have a more aggressive
  // approach inside MovePadPass instead. removes extra cast added from pushing
  // pad out OptimizationPass<ConsecutiveCastPass>::runPass(fusion);
  OptimizationPass<MarkAliasesPreparePass>::runPass(fusion);
  OptimizationPass<ExactMappedExtentSubstitutionPass>::runPass(fusion);
  OptimizationPass<AllocationDomainPass>::runPass(fusion);
  OptimizationPass<RemoveBcastSqueeze>::runPass(fusion);
  OptimizationPass<SegmentInplaceUpdatePass>::runPass(fusion);
}

} // namespace nvfuser::preseg_passes
