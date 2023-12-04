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
#include <linked_hash_map.h>
#include <optimization/alias_analysis.h>
#include <root_domain_map.h>

namespace nvfuser::optimization {

namespace {

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
  void handle(const SliceOp* slice) override;

 private:
  AliasAnalysisResult& analysis_;
};

// Computes `Split`'s output contiguity. Returns the outer contiguity and then
// the inner contiguity.
std::pair<std::optional<bool>, std::optional<bool>> splitContiguity(
    const std::optional<bool>& contiguity) {
  // Credits to @jacobhinkle:
  // https://github.com/NVIDIA/Fuser/pull/1124#discussion_r1368682735
  if (!contiguity.has_value()) {
    return {std::nullopt, std::nullopt};
  }
  if (*contiguity) {
    return {true, true};
  } else {
    return {true, false};
  }
}

// Computes `Merge`'s output contiguity. Returns a pair
// `<mergeable,contiguity>`. `mergeable` indicates whether the two IterDomains
// can be merged without materialization. For example, there's no way to merge
// `outer=f,inner=t` while keeping the output as an alias, because a dimension
// can only have one stride. `contiguity` is the contiguity of the merged output
// IterDomain.
//
// Credits to @jacobhinkle:
// https://github.com/NVIDIA/Fuser/pull/1124#discussion_r1368682735
std::pair<bool, std::optional<bool>> mergeContiguity(
    const IterDomain* outer_id,
    const std::optional<bool>& outer_contiguity,
    const IterDomain* inner_id,
    const std::optional<bool>& inner_contiguity) {
  // Statuses `b` and `e` are represented in the IR with isBroadcast() and
  // hasExpandedExtent(). Status `C` means stops propagating because we know we
  // can't alias at that point.
  //
  // o\i | t  f  b  e
  // ----+-----------
  //  t  | t  f  t  C
  //  f  | C  C  f  C
  //  b  | t  f  b  e
  //  e  | C  C  e  e
  if (!outer_contiguity.has_value() && !outer_id->hasExpandedExtent()) {
    return {true, inner_contiguity};
  }
  if (!inner_contiguity.has_value() && !inner_id->hasExpandedExtent()) {
    return {true, outer_contiguity};
  }

  // o\i | t  f  b  e
  // ----+-----------
  //  t  | t  f     C
  //  f  | C  C     C
  //  b  |
  //  e  | C  C     e
  if (outer_id->hasExpandedExtent() && inner_id->hasExpandedExtent()) {
    return {true, std::nullopt};
  }
  if (outer_id->hasExpandedExtent() || inner_id->hasExpandedExtent()) {
    return {false, std::nullopt};
  }

  // o\i | t  f  b  e
  // ----+-----------
  //  t  | t  f
  //  f  | C  C
  //  b  |
  //  e  |
  if (*outer_contiguity) {
    return {true, inner_contiguity};
  }
  return {false, std::nullopt};
}

void AliasFinder::handle(const ViewOp* view) {
  TensorView* in = view->in();
  TensorView* out = view->out();

  const std::vector<IterDomain*>& in_rfactor = in->getMaybeRFactorDomain();
  const std::vector<IterDomain*>& out_root = out->getRootDomain();
  const std::vector<IterDomain*>& out_rfactor = out->getMaybeRFactorDomain();

  Layout in_layout = analysis_.preferredLayout(in);
  if (!ir_utils::computePermutation(in_rfactor, in_layout.allocation_domain)
           .has_value()) {
    // Give up when `in`'s allocation domain is not an rfactor permutation.
    return;
  }

  std::unordered_map<IterDomain*, IterDomain*> in_rfactor_to_out_root =
      PairwiseRootDomainMap(in, out).mapBroadcast(true).mapProducerToConsumer();

  // Collect the allocation order of `in`'s rfactor domain and thus `out`'s root
  // domain.
  LinkedHashMap<IterDomain*, std::optional<bool>> allocation_to_contiguity;
  for (const auto i : c10::irange(in_layout.allocation_domain.size())) {
    IterDomain* in_allocation_id = in_layout.allocation_domain[i];
    if (!in_rfactor_to_out_root.count(in_allocation_id)) {
      // `in_allocation_id` is a reduction product.
      continue;
    }
    IterDomain* out_root_id = in_rfactor_to_out_root.at(in_allocation_id);
    allocation_to_contiguity.pushBack(out_root_id, in_layout.contiguity[i]);
  }

  // Replay `Expr`s from `out`'s root to `out`'s rfactor on `out`'s root.
  // Stop when an `Expr` requires a data copy; otherwise generate the allocation
  // order of `out`'s rfactor domain and the corresponding contiguity flags.
  for (Expr* transform : DependencyCheck::getAllExprsBetween(
           {out_root.begin(), out_root.end()},
           {out_rfactor.begin(), out_rfactor.end()})) {
    if (Split* split = dynamic_cast<Split*>(transform)) {
      const auto [contiguity, split_i] =
          allocation_to_contiguity.erase(split->in());
      auto [outer_contiguity, inner_contiguity] = splitContiguity(contiguity);
      allocation_to_contiguity.insert(
          split_i, split->outer(), outer_contiguity);
      allocation_to_contiguity.insert(
          split_i, split->inner(), inner_contiguity);
    } else if (Merge* merge = dynamic_cast<Merge*>(transform)) {
      const auto [outer_contiguity, inner_i] =
          allocation_to_contiguity.erase(merge->outer());
      if (inner_i == allocation_to_contiguity.end() ||
          inner_i->first != merge->inner()) {
        // Outer and inner are not adjacent in allocation order.
        return;
      }
      const auto [inner_contiguity, merge_i] =
          allocation_to_contiguity.erase(merge->inner());
      const auto [mergeable, contiguity] = mergeContiguity(
          merge->outer(), outer_contiguity, merge->inner(), inner_contiguity);
      if (!mergeable) {
        return;
      }
      allocation_to_contiguity.insert(merge_i, merge->out(), contiguity);
    } else {
      NVF_ERROR(
          false, "Expect Split or Merge, but found: ", transform->toString());
    }
  }

  Layout out_layout;
  for (const auto& [allocation_id, contiguity] : allocation_to_contiguity) {
    out_layout.allocation_domain.push_back(allocation_id);
    out_layout.contiguity.push_back(contiguity);
  }
  analysis_.add(out, in, std::move(out_layout));
}

void AliasFinder::handle(const LoadStoreOp* permute) {
  TensorView* in = dynamic_cast<TensorView*>(permute->in());
  if (in == nullptr) {
    return;
  }
  // Look at the preferred layout not `in`'s current layout.
  Layout in_layout = analysis_.preferredLayout(in);
  if (!ir_utils::computePermutation(
           in->getMaybeRFactorDomain(), in_layout.allocation_domain)
           .has_value()) {
    // Give up when `in`'s allocation domain is not an rfactor permutation.
    return;
  }

  TensorView* out = permute->out()->as<TensorView>();
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
  std::unordered_map<IterDomain*, IterDomain*> in_rfactor_to_out_root =
      PairwiseRootDomainMap(in, out).mapBroadcast(true).mapProducerToConsumer();

  Layout out_layout;
  for (const auto i : c10::irange(in_layout.allocation_domain.size())) {
    IterDomain* in_allocation_id = in_layout.allocation_domain[i];
    if (!in_rfactor_to_out_root.count(in_allocation_id)) {
      // `in_allocation_id` is a reduction product.
      continue;
    }
    out_layout.allocation_domain.push_back(
        in_rfactor_to_out_root.at(in_allocation_id));
    out_layout.contiguity.push_back(in_layout.contiguity[i]);
  }
  analysis_.add(out, in, std::move(out_layout));
}

// For future improvement, a PadOp with negative padding amount can also be
// treated as a slice.
void AliasFinder::handle(const SliceOp* slice) {
  TensorView* in = slice->in();
  TensorView* out = slice->out();

  const std::vector<IterDomain*>& in_rfactor = in->getMaybeRFactorDomain();
  const std::vector<IterDomain*>& out_root = out->getRootDomain();
  const std::vector<IterDomain*>& out_rfactor = out->getMaybeRFactorDomain();

  std::unordered_map<IterDomain*, IterDomain*> in_rfactor_to_out_root =
      PairwiseRootDomainMap(in, out).mapBroadcast(true).mapProducerToConsumer();

  const auto out_rank = out_rfactor.size();
  std::unordered_map<IterDomain*, IterDomain*> out_root_to_rfactor;
  out_root_to_rfactor.reserve(out_rank);
  for (auto i : c10::irange(out_rank)) {
    out_root_to_rfactor[out_root[i]] = out_rfactor[i];
  }

  Layout in_layout = analysis_.preferredLayout(in);
  if (!ir_utils::computePermutation(in_rfactor, in_layout.allocation_domain)
           .has_value()) {
    // Give up when `in`'s allocation domain is not an rfactor permutation.
    return;
  }

  // Inherit the allocation order from the input.  However, refine the
  // contiguity flags.
  Layout out_layout;
  out_layout.allocation_domain.reserve(out_rank);
  for (IterDomain* in_allocation_id : in_layout.allocation_domain) {
    if (!in_rfactor_to_out_root.count(in_allocation_id)) {
      // `in_allocation_id` is a reduction product.
      continue;
    }
    IterDomain* out_root_id = in_rfactor_to_out_root.at(in_allocation_id);
    out_layout.allocation_domain.push_back(out_root_to_rfactor.at(out_root_id));
  }

  // Scan through the allocation domain in minor-to-major order. If an
  // IterDomain is sliced, the next non-broadcast IterDomain has to be marked
  // non-contiguous. For example,
  //
  // in = makeContigConcreteTensor({16, 128, 3072});
  // out = slice(in, {0, 0, 0}, {16, 128, 1024});
  //
  // For `out` to alias `in`, its contiguity has to be updated to [t, f, t].
  out_layout.contiguity.resize(out_rank);
  bool next_non_broadcast_is_non_contiguous = false;
  for (auto i = static_cast<int64_t>(out_rank) - 1; i >= 0; i--) {
    if (out_layout.allocation_domain[i]->isBroadcast()) {
      out_layout.contiguity[i] = std::nullopt;
    } else if (next_non_broadcast_is_non_contiguous) {
      out_layout.contiguity[i] = false;
      next_non_broadcast_is_non_contiguous = false;
    } else {
      out_layout.contiguity[i] = in_layout.contiguity[i];
    }

    // A broadcast dimension can be a slicing product as well.
    std::vector<Expr*> dependencies = DependencyCheck::getAllExprsBetween(
        {out_root.begin(), out_root.end()}, {out_layout.allocation_domain[i]});
    if (std::find_if(
            dependencies.begin(), dependencies.end(), [](const Expr* expr) {
              return expr->isA<Resize>();
            }) != dependencies.end()) {
      // out_layout.allocation_domain[i] is sliced.
      next_non_broadcast_is_non_contiguous = true;
    }
  }

  analysis_.add(out, in, std::move(out_layout));
}

} // namespace

void AliasAnalysisResult::add(
    const TensorView* alias,
    const TensorView* source,
    Layout&& layout) {
  auto [i, inserted] = alias_to_source_.emplace(
      alias, std::make_pair(source, std::move(layout)));
  NVF_ERROR(
      inserted,
      "The current implementation of alias analysis shouldn't find two sources for an alias. However, it's trying to make ",
      alias->toString(),
      " an alias of ",
      source->toString(),
      " while it's already an alias of ",
      i->second.first->toString());
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

std::string AliasAnalysisResult::toString(const int indent_size) const {
  std::stringstream ss;
  for (const auto& [alias, source_and_layout] : alias_to_source_) {
    const auto& [source, layout] = source_and_layout;
    indent(ss, indent_size)
        << alias->toString() << " is an alias of " << source->toString()
        << " if its layout is " << layout.toString() << std::endl;
  }
  return ss.str();
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
