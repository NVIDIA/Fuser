// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <unordered_map>
#include <vector>

#include <fusion.h>
#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <optimization/alias_analysis.h>

namespace nvfuser::optimization {

namespace {

// Returns whether the input TensorView is allocated contiguously.
bool isContiguous(const TensorView& tv) {
  const std::vector<IterDomain*>& allocation_domain =
      tv.getMaybeAllocationDomain();
  for (size_t i = 0; i < allocation_domain.size(); i++) {
    // Broadcast and reduction dims are always contiguous because their sizes
    // are essentially 1.
    if (allocation_domain[i]->isBroadcast() ||
        allocation_domain[i]->isReduction()) {
      continue;
    }
    // Note: getContiguity() returns a vector of optional<bool>, so the `*` is
    // necessary.
    if (*tv.getContiguity()[i] == false) {
      return false;
    }
  }
  return true;
}

// Finds aliases of `source` and stores the findings in `alias_to_source`.
void findAliasesOfSource(
    const TensorView* source,
    AliasAnalysisResult& alias_to_source) {
  // The current implementation does the bare minimum to detect some aliasing
  // that the codegen can use to generate a kernel skipping unnecessary
  // computation.
  std::queue<const TensorView*> q;
  if (source->getMaybeAllocationDomain() == source->getMaybeRFactorDomain() &&
      isContiguous(*source)) {
    q.push(source);
  }

  while (!q.empty()) {
    const TensorView* in_tv = q.front();
    q.pop();

    for (Expr* use : in_tv->uses()) {
      if (!use->isA<ViewOp>()) {
        continue;
      }

      Val* out = use->output(0);
      TensorView* out_tv = dynamic_cast<TensorView*>(out);
      if (out_tv == nullptr) {
        continue;
      }

      // This is a sufficient but not necessary condition for `out_tv` to alias
      // `in_tv`.
      if (out_tv->getMaybeAllocationDomain() ==
              out_tv->getMaybeRFactorDomain() &&
          isContiguous(*out_tv)) {
        // Both `in_tv` and `out_tv` are allocated contiguously per the rfactor
        // domain.
        q.push(out_tv);
        alias_to_source[out_tv] = in_tv;
      }
    }
  }
}

} // namespace

AliasAnalysisResult findAliases(const Fusion& fusion) {
  AliasAnalysisResult alias_to_source;
  for (const Val* in : fusion.inputs()) {
    if (const TensorView* in_tv = dynamic_cast<const TensorView*>(in)) {
      findAliasesOfSource(in_tv, alias_to_source);
    }
  }
  return alias_to_source;
}

} // namespace nvfuser::optimization
