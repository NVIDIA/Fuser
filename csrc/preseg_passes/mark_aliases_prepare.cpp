// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <alias_analysis.h>
#include <debug.h>
#include <ir/utils.h>
#include <optimization/mark_aliases_prepare.h>
#include <options.h>

namespace nvfuser::optimization {

void MarkAliasesPreparePass::runPass(Fusion* fusion) {
  const AliasAnalysisResult analysis =
      findAliases(fusion, /*can_override_empty_allocation_domain=*/true);
  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion before MarkAliasesPreparePass:" << std::endl;
    fusion->printMath();
    debug() << "Alias analysis result:" << std::endl;
    debug() << analysis.toString(/*indent_size=*/1) << std::endl;
  }

  // Fusion outputs that are (1) aliased by another fusion output, (2) not
  // aliases themselves, and (3) not fusion inputs (yes, a fusion may trivially
  // forward an input). Code will later add `segment_set` before them so aliases
  // are separated from non-aliases and more likely to be accepted by the no-op
  // scheduler.
  std::unordered_set<TensorView*> aliased_outs;

  for (TensorView* tv : ir_utils::allTvs(fusion)) {
    TensorView* aliased_io = analysis.getNearestAliasedIo(tv);
    if (aliased_io == nullptr) {
      continue;
    }

    if (tv->isFusionOutput() && aliased_io->isFusionOutput() &&
        !aliased_io->isFusionInput() &&
        analysis.getNearestAliasedIo(aliased_io) == nullptr) {
      aliased_outs.insert(aliased_io);
    }

    // `AliasAnalysisResult::finalize` already checked the alias-enabling layout
    // is compliant with `tv`'s existing layout before adding `tv` to
    // `alias_to_root_`. So the existing layout can remain unchanged.
    if (tv->hasAllocation()) {
      continue;
    }

    // A scalar `tv` triggers a corner case that crashes
    // `validateDomainEquivalence`.
    if (tv->isZeroDim()) {
      continue;
    }

    const Layout preferred_layout = analysis.preferredLayout(tv);
    tv->setAllocationDomain(
        preferred_layout.allocation_domain, preferred_layout.contiguity);
    if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
      debug() << "Set the layout of " << ir_utils::varName(tv) << " to "
              << preferred_layout.toString() << std::endl;
    }
  }

  for (TensorView* aliased_out : aliased_outs) {
    // Rarely, if `aliased_out` is already defined by `segment_set`, don't
    // create another `segment_set`.
    if (LoadStoreOp* def =
            dynamic_cast<LoadStoreOp*>(aliased_out->definition())) {
      if (def != nullptr && def->opType() == LoadStoreOpType::SegmenterSet) {
        continue;
      }
    }

    // This is suboptimal in many uncommon cases. My head hurts when thinking
    // about them, so go simple for now :)
    //
    // Legend:
    //   M* = a meta op defining a **fusion output**
    //   N/M = a non-meta op defining a **fusion output**
    //
    // Case 1:
    //
    //   N/M -> N/M
    //      |
    //      -->  M
    //
    // We should put a `segment_set` on the **edge** from N/M to M, so the two
    // `N/M`s go to the same kernel.
    //
    // Case 2:
    //
    //   N/M -> M1 -> M2
    //           |
    //           --> N/M
    //
    // We should change it to
    //
    //   N/M -> M1 -> M2
    //      |
    //      --> M1' (non-output copy of M1) -> N/M
    //
    // and then put a `segment_set` on N/M->M1.
    aliased_out->cacheBefore(LoadStoreOpType::SegmenterSet);
  }

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion after MarkAliasesPreparePass:" << std::endl;
    fusion->printMath();
    fusion->printTransforms();
  }
}

} // namespace nvfuser::optimization
