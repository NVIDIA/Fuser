// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/add_axioms.h>

#include <unordered_set>
#include <vector>

#include <ir/utils.h>

namespace nvfuser::optimization {

void AddAxiomsPass::runPass(Fusion* fusion) {
  auto all_vals = fusion->usedMathVals();
  std::unordered_set<Val*> assumed_vals;
  for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
    std::vector<const std::vector<nvfuser::IterDomain*>*> interested_domains{
        &tv->getRootDomain()};
    if (tv->hasRFactor()) {
      interested_domains.push_back(&tv->getRFactorDomain());
    }
    if (tv->hasAllocation()) {
      interested_domains.push_back(&tv->getAllocationDomain());
    }
    for (auto dom : interested_domains) {
      for (auto id : *dom) {
        auto extent = id->extent();
        if (extent->definition() == nullptr && !extent->isConstScalar() &&
            assumed_vals.insert(extent).second) {
          fusion->assumePositive(extent);
        }
      }
    }
  }
}

} // namespace nvfuser::optimization
