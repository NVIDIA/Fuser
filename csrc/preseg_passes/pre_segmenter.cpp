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
#include <preseg_passes/finalize_multidevice_domains.h>
#include <preseg_passes/insert_reshardings.h>
#include <preseg_passes/mark_aliases_prepare.h>
#include <preseg_passes/move_gather.h>
#include <preseg_passes/move_pad.h>
#include <preseg_passes/move_repeat_forward.h>
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
  OptimizationPass<RemoveEmptyPass>::runPass(fusion);
  // This pass should be placed before ConsecutiveCastPass as more
  // consecutive cast ops may be exposed by this pass
  OptimizationPass<TranslateRepeatToExpand>::runPass(fusion);
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
  //
  // Moving a pad backward means all preceding operations would be
  // executed for the whole padded region too. Since the resize
  // scheduler does not have the issue, let it take care of padding
  // whenever enabled. Note that even when it is enabled, it is
  // currently only limited to pointwise patterns and does not
  // support, for example, reductions, etc, so this preseg pass still
  // may be preferable in some cases.
  if (isOptionDisabled(DisableOption::ResizeScheduler)) {
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

  // All the multidevice passes are moved after allocation related passes:
  // MarkAliasesPreparePass, and AllocationDomainPass Multidevice passes will
  // try to set the allocation domain for tvs with device mesh which will
  // conflict with these passes.
  OptimizationPass<PropagateShardingsPass>::runPass(fusion);
  OptimizationPass<InsertReshardingsPass>::runPass(fusion);
  OptimizationPass<ReorderShardedAxisPass>::runPass(fusion);

  OptimizationPass<RemoveBcastSqueeze>::runPass(fusion);
  OptimizationPass<SegmentInplaceUpdatePass>::runPass(fusion);
  OptimizationPass<TranslateNoReductionMatmulToMulSqueeze>::runPass(fusion);
  OptimizationPass<MoveRepeatForwardPass>::runPass(fusion);
  OptimizationPass<MoveGatherPass>::runPass(fusion);

  // This pass should be the last presegmentation pass.
  // It transforms the allocation domains of tvs with device mesh to
  // inherit DID splits. Before this pass, the allocation domains are
  // permutations of the logical domains.
  OptimizationPass<FinalizeMultideviceDomainsPass>::runPass(fusion);
}

} // namespace nvfuser::preseg_passes
