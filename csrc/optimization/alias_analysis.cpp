// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <unordered_map>
#include <vector>

#include <dispatch.h>
#include <fusion.h>
#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <optimization/alias_analysis.h>

namespace nvfuser::optimization {

namespace {

bool isContiguous(const std::vector<std::optional<bool>>& contiguity) {
  for (const auto& id_contiguity : contiguity) {
    // We skip std::nullopt contiguity. It represents a broadcast or reduction
    // dimension, which is of size 1 and always contiguous.
    if (id_contiguity.has_value() && id_contiguity.value() == false) {
      return false;
    }
  }
  return true;
}

// Returns whether the input TensorView is allocated contiguously.
bool isContiguous(const TensorView& tv) {
  return isContiguous(tv.getContiguity());
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
// `analysis`.
//
// The current implementation does the bare minimum to detect some aliasing
// that the codegen can use to generate a kernel skipping unnecessary
// computation.
//
// Many improvements are to be made. For example,
// 1. It should handle more op types such as `Slice`.
// 2. It should detect alias between non-packed tensors.
class AliasFinder : public OptOutConstDispatch {
 public:
  AliasFinder(AliasAnalysisResult& analysis) : analysis_(analysis) {}

  void handle(const ViewOp* view) override;
  void handle(const LoadStoreOp* ldst) override;

 private:
  AliasAnalysisResult& analysis_;
};

void AliasFinder::handle(const ViewOp* view) {
  TensorView* in = view->in();
  TensorView* out = view->out();

  Layout in_layout = analysis_.preferredLayout(in);
  const std::vector<IterDomain*>& out_allocation =
      out->getMaybeAllocationDomain();
  if (in_layout.allocation_domain == in->getMaybeRFactorDomain() &&
      isContiguous(in_layout.contiguity) &&
      out_allocation == out->getMaybeRFactorDomain() && isContiguous(*out) &&
      !transformsExpandedBroadcastIterDomain(in, out)) {
    // This is a sufficient but not necessary condition for `out` to alias
    // `in`. Both `in` and `out` are allocated contiguously per the
    // rfactor domain. Also, the ViewOp can't transform any expanded broadcast
    // IterDomain.
    analysis_.add(
        out,
        in,
        {out_allocation,
         TensorDomain::getContiguityFilledWith(out_allocation, true)});
  }
}

void AliasFinder::handle(const LoadStoreOp* permute) {
  TensorView* out = dynamic_cast<TensorView*>(permute->out());
  if (!out->hasRFactor()) {
    // Not a permute. It's actually an easier case to propagate aliases. I'm
    // too lazy.
    return;
  }

  // Another lazy move: we could check compatibility and only give up when
  // the allocation domain is incompatible with what we prefer for aliasing.
  if (out->hasAllocation()) {
    return;
  }

  TensorView* in = permute->in()->as<TensorView>();
  // Look at the preferred layout not `in`'s current layout.
  Layout in_layout = analysis_.preferredLayout(in);
  if (!ir_utils::computePermutation(
           in->getMaybeRFactorDomain(), in_layout.allocation_domain)
           .has_value()) {
    // Give up when `in`'s allocation domain is not an rfactor permutation.
    return;
  }

  // Compute `out`'s preferred allocation domain for aliasing.
  //
  // For example,
  //
  // in: rfactor=[i0,i1,i2], allocation=[i2,i0,i1]
  // out = permute(in, {2, 0, 1})
  // out: root=[i3,i4,i5], rfactor=[i4,i3,i5]
  //
  // `out`'s preferred allocation domain is [i5,i3,i4]. This allocation domain
  // is not affected by `out`'s rfactor domain or the permutation, because
  // `permute` changes the logical shape but not the physical layout.
  //
  // Therefore, `out`'s preferred allocation domain can be computed in two
  // steps:
  // 1. Construct the map from `in`'s rfactor to `out`'s root:
  // {i0->i3,i1->i4,i2->i5}.
  // 2. Apply the map to `in`'s allocation and get [i5,i3,i4].
  std::unordered_map<IterDomain*, IterDomain*> in_rfactor_to_out_root;
  for (auto i : c10::irange(out->getRootDomain().size())) {
    in_rfactor_to_out_root[in->getMaybeRFactorDomain()[i]] =
        out->getRootDomain()[i];
  }

  Layout out_layout;
  for (const auto i : c10::irange(in_layout.allocation_domain.size())) {
    IterDomain* allocation_id = in_layout.allocation_domain[i];
    out_layout.allocation_domain.push_back(
        in_rfactor_to_out_root.at(allocation_id));
    out_layout.contiguity.push_back(in_layout.contiguity[i]);
  }
  analysis_.add(out, in, out_layout);
}

} // namespace

void AliasAnalysisResult::add(
    const TensorView* alias,
    const TensorView* source,
    const Layout& layout) {
  std::pair<const TensorView*, Layout>& old_source = alias_to_source_[alias];
  NVF_ERROR(
      old_source.first == nullptr,
      "The current implementation of alias analysis shouldn't find two sources for an alias. However, it's trying to make ",
      alias->toString(),
      " an alias of ",
      source->toString(),
      " while it's already an alias of ",
      old_source.first->toString());
  old_source = {source, layout};
}

const Val* AliasAnalysisResult::findRoot(const Val* alias) const {
  const TensorView* root = dynamic_cast<const TensorView*>(alias);
  if (root == nullptr) {
    return nullptr;
  }

  // This can be made faster by path compression at the cost of losing
  // the potentially useful immediate sources. Go simple for now.
  while (alias_to_source_.count(root)) {
    root = alias_to_source_.at(root).first;
  }
  return root;
}

Layout AliasAnalysisResult::preferredLayout(const Val* v) const {
  const TensorView* tv = dynamic_cast<const TensorView*>(v);
  NVF_CHECK(
      tv != nullptr,
      "`v` is expected to be a TensorView. Found: ",
      v->toString());

  if (auto i = alias_to_source_.find(tv); i != alias_to_source_.end()) {
    return i->second.second;
  }
  return {tv->getMaybeAllocationDomain(), tv->getContiguity()};
}

AliasAnalysisResult findAliases(Fusion* fusion) {
  AliasAnalysisResult analysis;
  AliasFinder finder(analysis);
  // Fusion::exprs() computes and returns topological order.
  for (Expr* expr : fusion->exprs()) {
    // A potential improvement suggested by @tfogal: Let AliasFinder
    // return the AliasAnalysisResult instead of taking a mutable
    // `analysis` arg. This might be somewhat easily parallelizable
    // (albeit with a serialized merge step afterwards that inserts the
    // results).
    finder.dispatch(expr);
  }
  return analysis;
}

std::string Layout::toString(const int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "<allocation=["
                          << toDelimitedString(allocation_domain)
                          << "], contiguity=["
                          << toDelimitedString(contiguity, /*delim=*/" ")
                          << "]>";
  return ss.str();
}

} // namespace nvfuser::optimization
