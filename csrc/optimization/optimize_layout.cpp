// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <ir/utils.h>
#include <optimization/alias_analysis.h>
#include <optimization/optimize_layout.h>
#include <options.h>

namespace nvfuser::optimization {

void OptimizeLayoutPass::runPass(Fusion* fusion) {
  const AliasAnalysisResult analysis =
      findAliases(fusion, /*can_override_empty_allocation_domain=*/true);
  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion before OptimizeLayoutPass:" << std::endl;
    fusion->printMath();
    debug() << "Alias analysis result:" << std::endl;
    debug() << analysis.toString(/*indent_size=*/1) << std::endl;
  }

  // Fusion outputs that are (1) aliased by others and (2) not aliases
  // themselves. Code will later add `segment_set` before them so aliases are
  // separated from non-aliases and more likely to be accepted by the no-op
  // scheduler.
  std::vector<TensorView*> aliased_outs;
  aliased_outs.reserve(fusion->outputs().size());

  for (TensorView* out :
       ir_utils::filterByType<TensorView>(fusion->outputs())) {
    TensorView* aliased_io = analysis.getNearestAliasedIo(out);
    if (aliased_io == nullptr) {
      continue;
    }

    if (aliased_io->isFusionOutput() &&
        analysis.getNearestAliasedIo(aliased_io) == nullptr) {
      aliased_outs.push_back(aliased_io);
    }

    // We already checked it's compatible; no need to change.
    if (out->hasAllocation()) {
      continue;
    }

    // A scalar `out` triggers a corner case that crashes
    // `validateDomainEquivalence`.
    if (out->isZeroDim()) {
      continue;
    }

    const optimization::Layout preferred_layout = analysis.preferredLayout(out);
    out->setAllocationDomain(
        preferred_layout.allocation_domain, preferred_layout.contiguity);
    if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
      debug() << "Set the layout of " << ir_utils::varName(out) << " to "
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
    debug() << "Fusion after OptimizeLayoutPass:" << std::endl;
    fusion->printMath();
    fusion->printTransforms();
  }
}

} // namespace nvfuser::optimization
