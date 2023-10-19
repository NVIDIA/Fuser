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
    // We skip std::nullopt contiguity. It represents a broadcast or reduction
    // dimension, which is of size 1 and always contiguous.
    std::optional<bool> contiguity = tv.getContiguity()[i];
    if (contiguity.has_value() && contiguity.value() == false) {
      return false;
    }
  }
  return true;
}

// Finds aliases of `root` and stores the findings in `alias_to_source`.
void findAliasesOfRoot(
    const TensorView* root,
    AliasAnalysisResult& alias_to_source) {
  // The current implementation does the bare minimum to detect some aliasing
  // that the codegen can use to generate a kernel skipping unnecessary
  // computation.
  //
  // Many improvements are to be made. For example,
  // 1. Alias analysis should recommend non-default allocation domain
  // to proactively make output aliases.
  // 2. It should handle more op types such as `Set.Permute`.
  // 3. It should detect alias between non-packed tensors.
  std::queue<const TensorView*> q;
  if (root->getMaybeAllocationDomain() == root->getMaybeRFactorDomain() &&
      isContiguous(*root)) {
    q.push(root);
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
      // A potential improvement suggested by @tfogal: Let findAliasesOfRoot
      // return the AliasAnalysisResult instead of taking a mutable
      // `alias_to_source` arg. This might be somewhat easily parallelizable
      // (albeit with a serialized merge step afterwards that inserts the
      // results).
      findAliasesOfRoot(in_tv, alias_to_source);
    }
  }
  return alias_to_source;
}

} // namespace nvfuser::optimization
