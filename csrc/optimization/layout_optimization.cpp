// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/layout_inference.h>
#include <optimization/layout_optimization.h>

namespace nvfuser::optimization {

void LayoutOptimizationPass::runPass(Fusion* fusion) {
  std::unordered_map<const TensorView*, MemoryFormat> stride_mapping =
      inferenceMemoryFormat(fusion);

  for (auto out_val : fusion->outputs()) {
    auto out_tv = dynamic_cast<TensorView*>(out_val);
    // skip:
    //   1. non-tensor output;
    //   2. tensor output with allocation specified, assuming everything is
    //   semantical
    //   3. tensor output that's aliasing (Does aliased src matter?)
    if (out_tv == nullptr || out_tv->hasAllocation() ||
        fusion->getOutputAlias(out_val).type != AllocationType::NoAlias) {
      continue;
    }

    auto mapped_entry = stride_mapping.find(out_tv);
    if (mapped_entry == stride_mapping.end() || mapped_entry->second.empty()) {
      continue;
    }

    auto rfactor_dom = out_tv->getMaybeRFactorDomain();

    auto rank = rfactor_dom.size();
    std::vector<IterDomain*> allocation_domain(rank, nullptr);
    for (auto i : c10::irange(rank)) {
      allocation_domain.at(i) = rfactor_dom.at(mapped_entry->second.at(i));
    }
    out_tv->setAllocationDomain(allocation_domain, true);
  }
}

} // namespace nvfuser::optimization
