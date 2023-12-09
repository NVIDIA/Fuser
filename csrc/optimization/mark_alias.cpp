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
    const Val* in = analysis.getRoot(out);
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

    // When `out` is a scalar, `out->setAllocationDomain` triggers a corner case
    // that crashes `validateDomainEquivalence`.
    if (out->isZeroDim()) {
      continue;
    }

    const Layout out_layout = analysis.preferredLayout(out);
    if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
      debug() << "MarkAliasPass changed the layout of " << out->toString()
              << std::endl;
      debug() << "  Old TensorDomain:" << std::endl;
      debug() << out->domain()->toString(4, /*leaf_only=*/false) << std::endl;
      debug() << "  New layout:" << out_layout.toString() << std::endl;
    }
    out->setAllocationDomain(
        out_layout.allocation_domain, out_layout.contiguity);
  }
}

} // namespace nvfuser::optimization
