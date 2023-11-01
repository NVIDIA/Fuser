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
  fusion->print();

  const AliasAnalysisResult alias_analysis = findAliases(fusion);
  for (Val* out : fusion->outputs()) {
    if (Val* in = const_cast<Val*>(alias_analysis.findRoot(out));
        in->isFusionInput()) {
      fusion->aliasOutputToInput(out, in, AliasType::PointerCast);
    }
  }
}

} // namespace nvfuser::optimization
