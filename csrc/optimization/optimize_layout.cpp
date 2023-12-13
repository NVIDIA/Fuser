// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <ir/utils.h>
#include <ops/alias.h>
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

  std::vector<TensorView*> root_outs;
  root_outs.reserve(fusion->outputs().size());

  for (TensorView* out :
       ir_utils::filterByType<TensorView>(fusion->outputs())) {
    TensorView* in = analysis.getAliasedInput(out);
    if (in == nullptr) {
      continue;
    }

    if (in->isFusionOutput()) {
      root_outs.push_back(in);
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

  for (TensorView* root_out : root_outs) {
    // Rarely, if `root_out` is already defined by `segment_set`, don't replace
    // it with another `segment_set`.
    if (LoadStoreOp* def = dynamic_cast<LoadStoreOp*>(root_out->definition())) {
      if (def != nullptr && def->opType() == LoadStoreOpType::SegmenterSet) {
        continue;
      }
    }

    TensorView* new_root_out = segment_set(root_out);
    if (root_out->hasAllocation()) {
      new_root_out->setAllocationDomain(
          root_out->getAllocationDomain(), root_out->getContiguity());
    }
    fusion->replaceOutput(root_out, new_root_out);
    for (Expr* use : root_out->uses()) {
      if (use != new_root_out->definition()) {
        ir_utils::replaceValInExprInputs(use, root_out, new_root_out);
      }
    }
  }

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion after OptimizeLayoutPass:" << std::endl;
    fusion->printMath();
    fusion->printTransforms();
  }
}

} // namespace nvfuser::optimization
