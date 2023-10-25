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

// Whether a ViewOp transforms any expanded broadcast IterDomain in the input.
// This is a corner case in which we can't always turn `out` into an alias.
//
// For example, given
//
//   t0 = makeContigConcreteTensor({4, 5});
//   t1 = broadcast(t0, {false, false, true});
//   t2 = expand(t1, {4, 5, 6});
//
// `reshape(t2, {40, 3})` and `reshape(t2, {4, 30})` because both merge the
// expanded broadcast IterDomain (6) or a subspace of it with preceding
// IterDomains.  However, the output of `reshape(t2, {20, 6})` can simply be an
// alias because the expanded broadcast IterDomain is forwarded not transformed.
//
// As a future improvement, when an expanded broadcast dimension is only split,
// the output of the reshape can be an alias. However, nvFuser currently decides
// to materialize the expansion, making the output not an alias (#1126).
//
// Obviously, this function assumes `in` and `out` are the input and output
// TensorView of the same ViewOp.
bool transformsExpandedBroadcastIterDomain(TensorView* in, TensorView* out) {
  const std::vector<IterDomain*>& in_rfactor = in->getMaybeRFactorDomain();
  const std::vector<IterDomain*>& out_root = out->getRootDomain();
  const std::vector<IterDomain*>& out_rfactor = out->getMaybeRFactorDomain();

  std::unordered_set<Val*> expanded_broadcast_dims;
  for (size_t i = 0, size = in_rfactor.size(); i < size; i++) {
    IterDomain* id = in_rfactor[i];
    if (id->isBroadcast() && id->hasExpandedExtent()) {
      expanded_broadcast_dims.insert(out_root[i]);
    }
  }

  const std::vector<Expr*> transforms = DependencyCheck::getAllExprsBetween(
      {out_root.begin(), out_root.end()},
      {out_rfactor.begin(), out_rfactor.end()});

  for (const auto* transform : transforms) {
    for (Val* input : transform->inputs()) {
      if (expanded_broadcast_dims.count(input)) {
        return true;
      }
    }
  }
  return false;
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
    TensorView* in = view->in();
    TensorView* out = view->out();

    if (in->getMaybeAllocationDomain() == in->getMaybeRFactorDomain() &&
        isContiguous(*in) &&
        out->getMaybeAllocationDomain() == out->getMaybeRFactorDomain() &&
        isContiguous(*out) && !transformsExpandedBroadcastIterDomain(in, out)) {
      // This is a sufficient but not necessary condition for `out` to alias
      // `in`. Both `in` and `out` are allocated contiguously per the
      // rfactor domain. Also, the ViewOp can't transform any expanded broadcast
      // IterDomain.
      alias_to_source[out] = in;
    }
  }
}

} // namespace

AliasAnalysisResult findAliases(Fusion* fusion) {
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
