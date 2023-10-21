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

// Finds aliases between `expr`'s inputs and outputs and stores the findings in
// `alias_to_source`.
void findAliasesFromExpr(Expr* expr, AliasAnalysisResult& alias_to_source) {
  // The current implementation does the bare minimum to detect some aliasing
  // that the codegen can use to generate a kernel skipping unnecessary
  // computation.
  //
  // Many improvements are to be made. For example,
  // 1. Alias analysis should recommend non-default allocation domain
  // to proactively make output aliases.
  // 2. It should handle more op types such as `Set.Permute`.
  // 3. It should detect alias between non-packed tensors.
  if (ViewOp* view = dynamic_cast<ViewOp*>(expr)) {
    TensorView* in_tv = dynamic_cast<TensorView*>(view->in());
    if (in_tv == nullptr) {
      return;
    }

    TensorView* out_tv = dynamic_cast<TensorView*>(view->out());
    if (out_tv == nullptr) {
      return;
    }

    const std::vector<IterDomain*>& out_root = out_tv->getRootDomain();
    const std::vector<IterDomain*>& out_rfactor =
        out_tv->getMaybeRFactorDomain();
    std::vector<Expr*> transforms = DependencyCheck::getAllExprsBetween(
        {out_root.begin(), out_root.end()},
        {out_rfactor.begin(), out_rfactor.end()});

    std::unordered_set<Val*> expanded_broadcast_dims;
    expanded_broadcast_dims.reserve(in_tv->getMaybeRFactorDomain().size());
    for (size_t i = 0, size = in_tv->getMaybeRFactorDomain().size(); i < size;
         i++) {
      IterDomain* id = in_tv->getMaybeRFactorDomain()[i];
      if (id->isBroadcast() && id->hasExpandedExtent()) {
        expanded_broadcast_dims.insert(out_root[i]);
      }
    }

    for (const auto* transform : transforms) {
      for (Val* input : transform->inputs()) {
        if (expanded_broadcast_dims.count(input)) {
          return;
        }
      }
    }

    // This is a sufficient but not necessary condition for `out_tv` to alias
    // `in_tv`.
    if (in_tv->getMaybeAllocationDomain() == in_tv->getMaybeRFactorDomain() &&
        isContiguous(*in_tv) &&
        out_tv->getMaybeAllocationDomain() == out_tv->getMaybeRFactorDomain() &&
        isContiguous(*out_tv)) {
      // Both `in_tv` and `out_tv` are allocated contiguously per the rfactor
      // domain.
      alias_to_source[out_tv] = in_tv;
    }
  }
}

} // namespace

AliasAnalysisResult findAliases(Fusion* fusion) {
  fusion->print();

  AliasAnalysisResult alias_to_source;
  // Fusion::exprs() returns topological order.
  for (Expr* expr : fusion->exprs()) {
    // A potential improvement suggested by @tfogal: Let findAliasesFromExpr
    // return the AliasAnalysisResult instead of taking a mutable
    // `alias_to_source` arg. This might be somewhat easily parallelizable
    // (albeit with a serialized merge step afterwards that inserts the
    // results).
    findAliasesFromExpr(expr, alias_to_source);
  }
  return alias_to_source;
}

} // namespace nvfuser::optimization
