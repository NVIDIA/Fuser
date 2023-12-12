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
#include <optimization/mark_alias.h>
#include <options.h>

namespace nvfuser::optimization {

void MarkAliasPass::runPass(Fusion* fusion) {
  const AliasAnalysisResult analysis = findAliases(fusion);
  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Alias analysis result:" << std::endl;
    debug() << analysis.toString(/*indent_size=*/1) << std::endl;
  }

  for (TensorView* out :
       ir_utils::filterByType<TensorView>(fusion->outputs())) {
    const Val* in = analysis.getAliasedInput(out);
    if (in == nullptr) {
      continue;
    }

    fusion->aliasOutputToInput(
        out,
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<Val*>(in),
        AliasType::PointerArithmetic);
    if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
      debug() << "MarkAliasPass marked " << out->toString()
              << " as an alias of " << in->toString() << std::endl;
    }

    // We already checked it's compatible; no need to change.
    if (out->hasAllocation()) {
      continue;
    }

    // When `out` is a scalar, `out->setAllocationDomain` triggers a corner case
    // that crashes `validateDomainEquivalence`.
    if (out->isZeroDim()) {
      continue;
    }

    const Layout preferred_layout = analysis.preferredLayout(out);
    out->setAllocationDomain(
        preferred_layout.allocation_domain, preferred_layout.contiguity);
    if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
      debug() << "MarkAliasPass set the layout of " << out->toString() << " to "
              << preferred_layout.toString() << std::endl;
    }
  }
}

} // namespace nvfuser::optimization
