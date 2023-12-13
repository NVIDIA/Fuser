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
    debug() << "Alias analysis result:" << std::endl;
    debug() << analysis.toString(/*indent_size=*/1) << std::endl;
  }

  for (TensorView* out :
       ir_utils::filterByType<TensorView>(fusion->outputs())) {
    TensorView* in = analysis.getAliasedInput(out);
    if (in == nullptr) {
      continue;
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
              << preferred_layout.toString();
    }
  }
}
} // namespace nvfuser::optimization
