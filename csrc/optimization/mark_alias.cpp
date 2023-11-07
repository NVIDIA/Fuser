// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/utils.h>
#include <optimization/alias_analysis.h>
#include <optimization/mark_alias.h>

namespace nvfuser::optimization {

void MarkAliasPass::runPass(Fusion* fusion) {
  const AliasAnalysisResult alias_analysis = findAliases(fusion);
  for (TensorView* out :
       ir_utils::filterByType<TensorView>(fusion->outputs())) {
    if (const Val* in = alias_analysis.findRoot(out); in->isFusionInput()) {
      fusion->aliasOutputToInput(
          out,
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<Val*>(in),
          AliasType::PointerCast);

      // A scalar `out` triggers a corner case that crashes
      // `validateDomainEquivalence`.
      if (!out->isZeroDim()) {
        const Layout out_layout = alias_analysis.preferredLayout(out);
        out->setAllocationDomain(
            out_layout.allocation_domain, out_layout.contiguity);
      }
    }
  }
}

} // namespace nvfuser::optimization
