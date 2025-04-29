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
#include <preseg_passes/translate_no_reduction_matmul_to_mul_squeeze.h>
#include <preseg_passes/translate_repeat_to_expand.h>

namespace nvfuser::preseg_passes {

/*static*/ void PreSegmenter::runPass(Fusion* fusion) {
  FUSER_PERF_SCOPE("PreSegmenter::runPass");

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion before " << name() << ":" << std::endl;
    fusion->printMath();
    debug() << "========================================" << std::endl;
  }

  // Replace TensorViews with zero extent. Outputs and inputs may still be empty
  debug() << "[PreSegmenter] Running RemoveEmptyPass..." << std::endl;
  OptimizationPass<RemoveEmptyPass>::runPass(fusion);
  debug() << "[PreSegmenter] Finished RemoveEmptyPass." << std::endl;
  // This pass should be placed before ConsecutiveCastPass as more
  // consecutive cast ops may be exposed by this pass
  debug() << "[PreSegmenter] Running TranslateRepeatToExpand..." << std::endl;
  OptimizationPass<TranslateRepeatToExpand>::runPass(fusion);
  debug() << "[PreSegmenter] Finished TranslateRepeatToExpand." << std::endl;
  // removes consecutive cast operations
  debug() << "[PreSegmenter] Running ConsecutiveCastPass..." << std::endl;
  OptimizationPass<ConsecutiveCastPass>::runPass(fusion);
  debug() << "[PreSegmenter] Finished ConsecutiveCastPass." << std::endl;
  debug() << "[PreSegmenter] Running AddAxiomsPass..." << std::endl;
  OptimizationPass<AddAxiomsPass>::runPass(fusion);
  debug() << "[PreSegmenter] Finished AddAxiomsPass." << std::endl;
  debug() << "[PreSegmenter] Running MoveSplitCatPass..." << std::endl;
  OptimizationPass<MoveSplitCatPass>::runPass(fusion);
  debug() << "[PreSegmenter] Finished MoveSplitCatPass." << std::endl;
  // MovePadPass needs to happen:
  // 1. before MarkAliasPrepare; and
  //    avoid moving pad operatoins around, which could disturb the analysis
  //    from MarkAliasPrepare
  // 2. after MoveSplitCat
  //    to avoid this pass moving PadOp around to break the
  // MoveSplitCat.
  //
  // Moving a pad backward means all preceding operations would be
  // executed for the whole padded region too. Since the resize
  // scheduler does not have the issue, let it take care of padding
  // whenever enabled. Note that even when it is enabled, it is
  // currently only limited to pointwise patterns and does not
  // support, for example, reductions, etc, so this preseg pass still
  // may be preferable in some cases.
  if (isOptionDisabled(DisableOption::ResizeScheduler)) {
    debug() << "[PreSegmenter] Running MovePadPass..." << std::endl;
    OptimizationPass<MovePadPass>::runPass(fusion);
    debug() << "[PreSegmenter] Finished MovePadPass." << std::endl;
  } else {
    debug() << "[PreSegmenter] Skipping MovePadPass due to ResizeScheduler enabled." << std::endl;
  }

  debug() << "After MovePadPass" << std::endl;
  
  // NOTE vvv this doesn't really work, since our type promotion to higher
  // precision for Add cannot be canceled out with previous cast to lower
  // precision. Since it's not an no-op and it has a quantization effect. I'll
  // open an issue for this and see if we want to have a more aggressive
  // approach inside MovePadPass instead. removes extra cast added from pushing
  // pad out OptimizationPass<ConsecutiveCastPass>::runPass(fusion);
  debug() << "[PreSegmenter] Running MarkAliasesPreparePass..." << std::endl;
  OptimizationPass<MarkAliasesPreparePass>::runPass(fusion);
  debug() << "[PreSegmenter] Finished MarkAliasesPreparePass." << std::endl;
  debug() << "[PreSegmenter] Running ExactMappedExtentSubstitutionPass..." << std::endl;
  OptimizationPass<ExactMappedExtentSubstitutionPass>::runPass(fusion);
  debug() << "[PreSegmenter] Finished ExactMappedExtentSubstitutionPass." << std::endl;
  debug() << "[PreSegmenter] Running AllocationDomainPass..." << std::endl;
  OptimizationPass<AllocationDomainPass>::runPass(fusion);
  debug() << "[PreSegmenter] Finished AllocationDomainPass." << std::endl;

  // All the multidevice passes are moved after allocation related passes:
  // MarkAliasesPreparePass, and AllocationDomainPass Multidevice passes will
  // try to set the allocation domain for tvs with device mesh which will
  // conflict with these passes.
  debug() << "[PreSegmenter] Running PropagateShardingsPass..." << std::endl;
  OptimizationPass<PropagateShardingsPass>::runPass(fusion);
  debug() << "[PreSegmenter] Finished PropagateShardingsPass." << std::endl;
  debug() << "[PreSegmenter] Running InsertReshardingsPass..." << std::endl;
  OptimizationPass<InsertReshardingsPass>::runPass(fusion);
  debug() << "[PreSegmenter] Finished InsertReshardingsPass." << std::endl;
  debug() << "[PreSegmenter] Running ReorderShardedAxisPass..." << std::endl;
  OptimizationPass<ReorderShardedAxisPass>::runPass(fusion);
  debug() << "[PreSegmenter] Finished ReorderShardedAxisPass." << std::endl;
  debug() << "[PreSegmenter] Running MakeReshardingContiguousPass..." << std::endl;
  OptimizationPass<MakeReshardingContiguousPass>::runPass(fusion);
  debug() << "[PreSegmenter] Finished MakeReshardingContiguousPass." << std::endl;

  debug() << "[PreSegmenter] Running RemoveBcastSqueeze..." << std::endl;
  OptimizationPass<RemoveBcastSqueeze>::runPass(fusion);
  debug() << "[PreSegmenter] Finished RemoveBcastSqueeze." << std::endl;
  debug() << "[PreSegmenter] Running SegmentInplaceUpdatePass..." << std::endl;
  OptimizationPass<SegmentInplaceUpdatePass>::runPass(fusion);
  debug() << "[PreSegmenter] Finished SegmentInplaceUpdatePass." << std::endl;
  debug() << "[PreSegmenter] Running TranslateNoReductionMatmulToMulSqueeze..." << std::endl;
  OptimizationPass<TranslateNoReductionMatmulToMulSqueeze>::runPass(fusion);
  debug() << "[PreSegmenter] Finished TranslateNoReductionMatmulToMulSqueeze." << std::endl;
}

} // namespace nvfuser::preseg_passes
