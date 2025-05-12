// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <unordered_map>
#include <vector>

#include <alias_analysis.h>
#include <dispatch.h>
#include <exceptions.h>
#include <fusion.h>
#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <linked_hash_map.h>
#include <logical_domain_map.h>
#include <multidevice/utils.h>

namespace nvfuser {

namespace {

// Finds aliases between `expr`'s inputs and outputs and stores the findings in
// `analysis`.
//
// The current implementation does the bare minimum to detect some aliasing
// that the codegen can use to generate a kernel skipping unnecessary
// computation.
class AliasFinder : public OptOutConstDispatch {
 public:
  AliasFinder(
      EmptyAllocationAs empty_allocation_as,
      AliasAnalysisResult& analysis)
      : empty_allocation_as_(empty_allocation_as), analysis_(analysis) {}

  void handle(const ViewOp*) override;
  void handle(const LoadStoreOp*) override;
  void handle(const SliceOp*) override;
  void handle(const BroadcastOp*) override;
  void handle(const SqueezeOp*) override;
  void handle(const ExpandOp*) override;

 private:
  // Marks `alias` and `source` alias if `layout` is compliant with `alias`'s
  // existing allocation domain. Returns true if succeeded.
  bool aliasIfCompliant(
      const TensorView* alias,
      TensorView* source,
      Layout&& layout);

  EmptyAllocationAs empty_allocation_as_;
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
    const bool outer_is_expanded,
    const std::optional<bool>& outer_contiguity,
    const bool inner_is_expanded,
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
  if (!outer_contiguity.has_value() && !outer_is_expanded) {
    return {true, inner_contiguity};
  }
  if (!inner_contiguity.has_value() && !inner_is_expanded) {
    return {true, outer_contiguity};
  }

  // o\i | t  f  b  e
  // ----+-----------
  //  t  | t  f     C
  //  f  | C  C     C
  //  b  |
  //  e  | C  C     e
  if (outer_is_expanded && inner_is_expanded) {
    return {true, std::nullopt};
  }
  if (outer_is_expanded || inner_is_expanded) {
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

// A helper function used to compute the perferred output layout. It computes
// the mapping from `in_logical` to `out_root` and applies that mapping to
// `preferred_in_layout`. For many ops, this function returns a good initial
// preferred output layout for aliasing because it tries to preserve the input
// layout. An op (e.g. ViewOp and SliceOp) that transforms root to logical
// using expressions will have to modify this initial layout so its allocation
// domain will be a function of its logical domain.
//
// Returns `nullopt` if computation fails, so the caller can handle things
// conservatively.
std::optional<Layout> mapInLayoutToOutRoot(
    const Layout& preferred_in_layout,
    TensorView* in,
    TensorView* out) {
  if (!ir_utils::computePermutation(
           in->getLogicalDomain(), preferred_in_layout.allocation_domain)
           .has_value()) {
    // Give up when `in`'s allocation domain is not an logical permutation. As
    // an extension, we could map in_alloc to in_logical and apply the inverse
    // mapping to out_root.
    return std::nullopt;
  }

  std::unordered_map<IterDomain*, IterDomain*> in_logical_to_out_root =
      PairwiseLogicalDomainMap(in, out).mapProducerToConsumer();

  Layout preferred_out_layout;
  for (auto&& [in_alloc_id, contiguity] :
       zip(preferred_in_layout.allocation_domain,
           preferred_in_layout.contiguity)) {
    IterDomain* out_root_id = getOrDefault(in_logical_to_out_root, in_alloc_id);
    if (out_root_id == nullptr) {
      // This can happen when in_alloc_id is of type reduction or squeezed out.
      continue;
    }
    preferred_out_layout.allocation_domain.push_back(out_root_id);
    preferred_out_layout.contiguity.push_back(contiguity);
  }
  return preferred_out_layout;
}

namespace {
bool okToRelayout(
    const TensorView* tv,
    const Layout& new_layout,
    const EmptyAllocationAs empty_allocation_as) {
  const std::vector<IterDomain*> allocation =
      (empty_allocation_as == EmptyAllocationAs::kUndetermined
           ? tv->getAllocationDomain()
           : tv->getMaybeAllocationDomain());
  return new_layout.isCompliantWith({allocation, tv->getContiguity()});
}
} // namespace

bool AliasFinder::aliasIfCompliant(
    const TensorView* alias,
    TensorView* source,
    Layout&& layout) {
  if (!okToRelayout(alias, layout, empty_allocation_as_)) {
    return false;
  }
  analysis_.add(alias, source, std::move(layout));
  return true;
}

void AliasFinder::handle(const ViewOp* view) {
  TensorView* in = view->in();
  TensorView* out = view->out();

  // Collect the allocation order of `in`'s logical domain and thus `out`'s root
  // domain.
  std::optional<Layout> out_root_layout =
      mapInLayoutToOutRoot(analysis_.createOrGetPreferredLayout(in), in, out);
  if (!out_root_layout.has_value()) {
    return;
  }

  LinkedHashMap<IterDomain*, std::optional<bool>> allocation_to_contiguity;
  for (const auto i : arange(out_root_layout->size())) {
    if (!out_root_layout->contiguity[i].has_value() &&
        !out_root_layout->allocation_domain[i]->isBroadcast()) {
      // TODO(#1126): Due to #1126, `out_root` materializes an expanded
      // broadcast IterDomain from `in_logical` when `view` splits or merges
      // that IterDomain. We return no alias when this happen; otherwise
      // AliasTest.MergeBroadcastsBetweenConcretes would fail.
      return;
    }
    allocation_to_contiguity.pushBack(
        out_root_layout->allocation_domain[i], out_root_layout->contiguity[i]);
  }

  // Replay `Expr`s from `out_root` to `out_logical` on
  // `allocation_to_contiguity`. Stop when an `Expr` requires a data copy;
  // otherwise generate the allocation order of `out_logical` and the
  // corresponding contiguity flags.
  const std::vector<IterDomain*>& out_root = out->getRootDomain();
  const std::vector<IterDomain*>& out_logical = out->getLogicalDomain();
  for (Expr* transform : DependencyCheck::getAllExprsBetween(
           {out_root.begin(), out_root.end()},
           {out_logical.begin(), out_logical.end()})) {
    if (auto* split = dynamic_cast<Split*>(transform)) {
      const auto [contiguity, split_i] =
          allocation_to_contiguity.erase(split->in());
      auto [outer_contiguity, inner_contiguity] = splitContiguity(contiguity);
      allocation_to_contiguity.insert(
          split_i, split->outer(), outer_contiguity);
      allocation_to_contiguity.insert(
          split_i, split->inner(), inner_contiguity);
    } else if (auto* merge = dynamic_cast<Merge*>(transform)) {
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
          merge->outer()->hasExpandedExtent(),
          outer_contiguity,
          merge->inner()->hasExpandedExtent(),
          inner_contiguity);
      if (!mergeable) {
        return;
      }
      allocation_to_contiguity.insert(merge_i, merge->out(), contiguity);
    } else {
      NVF_THROW("Expect Split or Merge, but found: ", transform);
    }
  }

  Layout out_logical_layout;
  for (auto&& [alloc_id, contiguity] : allocation_to_contiguity) {
    out_logical_layout.allocation_domain.push_back(alloc_id);
    out_logical_layout.contiguity.push_back(contiguity);
  }
  aliasIfCompliant(out, in, std::move(out_logical_layout));
}

void AliasFinder::handle(const LoadStoreOp* set) {
  TensorView* in = dynamic_cast<TensorView*>(set->in());
  if (in == nullptr) {
    return;
  }
  auto* out = set->out()->as<TensorView>();

  // Compute `out`'s preferred allocation domain for aliasing.
  //
  // For example,
  //
  // in: logical=[i0,i1,i2], allocation=[i2,i0,i1]
  // out = permute(in, {1, 0, 2})
  // out: root=[i3,i4,i5], logical=[i4,i3,i5]
  //
  // `out`'s preferred allocation domain is [i5,i3,i4]. This allocation domain
  // is not affected by `out`'s logical domain or the permutation, because
  // `permute` changes the logical shape but not the physical layout.
  //
  // Therefore, `out`'s preferred allocation domain can be computed in two
  // steps:
  // 1. Construct the map from `in`'s logical to `out`'s root:
  // {i0->i3,i1->i4,i2->i5}.
  // 2. Apply the map to `in`'s allocation and get [i5,i3,i4].
  std::optional<Layout> out_root_layout =
      mapInLayoutToOutRoot(analysis_.createOrGetPreferredLayout(in), in, out);
  if (!out_root_layout.has_value()) {
    return;
  }
  aliasIfCompliant(out, in, std::move(*out_root_layout));
}

// For future improvement, a PadOp with negative padding amount can also be
// treated as a slice.
void AliasFinder::handle(const SliceOp* slice) {
  TensorView* in = slice->in();
  TensorView* out = slice->out();

  std::optional<Layout> out_layout =
      mapInLayoutToOutRoot(analysis_.createOrGetPreferredLayout(in), in, out);
  if (!out_layout.has_value()) {
    return;
  }

  const std::vector<IterDomain*>& out_root = out->getRootDomain();
  std::unordered_map<IterDomain*, IterDomain*> out_root_to_logical;
  {
    const std::vector<IterDomain*>& out_logical = out->getLogicalDomain();
    NVF_ERROR_EQ(out_root.size(), out_logical.size());
    out_root_to_logical.reserve(out_root.size());
    for (auto&& [root_id, logical_id] : zip(out_root, out_logical)) {
      out_root_to_logical[root_id] = logical_id;
    }
  }

  // Inherit the allocation order from the input. However, refine the
  // contiguity flags. This is done by scanning through the allocation domain in
  // minor-to-major order. If an IterDomain is sliced, the next non-broadcast
  // IterDomain has to be marked non-contiguous. For example,
  //
  //   in = makeContigConcreteTensor({16, 128, 3072});
  //   out = slice(in, {0, 0, 0}, {16, 128, 1024});
  //
  // For `out` to alias `in`, its contiguity has to be updated to [t, f, t].
  bool next_non_broadcast_is_non_contiguous = false;
  for (int64_t i = out_layout->size() - 1; i >= 0; i--) {
    IterDomain*& alloc_id = out_layout->allocation_domain[i];
    std::optional<bool>& contiguity = out_layout->contiguity[i];

    alloc_id = out_root_to_logical.at(alloc_id);

    if (alloc_id->isBroadcast()) {
      // A broadcast dimension may be a slicing product as well. So, don't
      // prematurely skip the rest of the loop.
      contiguity = std::nullopt;
    } else if (next_non_broadcast_is_non_contiguous) {
      contiguity = false;
      next_non_broadcast_is_non_contiguous = false;
    }

    // Set `next_non_broadcast_is_non_contiguous` if this dimension is sliced.
    std::vector<Expr*> dependencies = DependencyCheck::getAllExprsBetween(
        {out_root.begin(), out_root.end()}, {alloc_id});
    if (std::find_if(
            dependencies.begin(), dependencies.end(), [](const Expr* expr) {
              return expr->isA<Resize>();
            }) != dependencies.end()) {
      // `alloc_id` is sliced.
      next_non_broadcast_is_non_contiguous = true;
    }
  }

  aliasIfCompliant(out, in, std::move(*out_layout));
}

void AliasFinder::handle(const BroadcastOp* bcast) {
  TensorView* in = dynamic_cast<TensorView*>(bcast->in());
  if (in == nullptr) {
    return;
  }
  auto* out = bcast->out()->as<TensorView>();

  std::optional<Layout> out_layout =
      mapInLayoutToOutRoot(analysis_.createOrGetPreferredLayout(in), in, out);
  if (!out_layout.has_value()) {
    return;
  }

  // Put new, broadcast dimensions to the end.
  const std::vector<IterDomain*> out_logical = out->getLogicalDomain();
  for (const auto i : arange(out_logical.size())) {
    if (bcast->isBroadcastDim(i)) {
      out_layout->allocation_domain.push_back(out_logical[i]);
      out_layout->contiguity.emplace_back(std::nullopt);
    }
  }

  aliasIfCompliant(out, in, std::move(*out_layout));
}

void AliasFinder::handle(const SqueezeOp* squeeze) {
  TensorView* in = dynamic_cast<TensorView*>(squeeze->in());
  if (in == nullptr) {
    return;
  }
  auto* out = squeeze->out()->as<TensorView>();

  // Preserve the allocation order of existing dimensions.
  std::optional<Layout> out_layout =
      mapInLayoutToOutRoot(analysis_.createOrGetPreferredLayout(in), in, out);
  if (!out_layout.has_value()) {
    return;
  }

  aliasIfCompliant(out, in, std::move(*out_layout));
}

void AliasFinder::handle(const ExpandOp* expand) {
  auto* in = dynamic_cast<TensorView*>(expand->in());
  if (in == nullptr) {
    return;
  }
  auto* out = expand->out()->as<TensorView>();

  // Preserve the allocation order of existing dimensions.
  std::optional<Layout> out_layout =
      mapInLayoutToOutRoot(analysis_.createOrGetPreferredLayout(in), in, out);
  if (!out_layout.has_value()) {
    return;
  }

  aliasIfCompliant(out, in, std::move(*out_layout));
}

} // namespace

void AliasAnalysisResult::add(
    const TensorView* alias,
    TensorView* source,
    Layout&& layout) {
  auto [i, inserted] = alias_to_source_.emplace(
      alias, std::make_pair(source, std::move(layout)));
  NVF_ERROR(
      inserted,
      "The current implementation of alias analysis shouldn't find two "
      "sources for an alias. However, it's trying to make ",
      alias,
      " an alias of ",
      source,
      " while it's already an alias of ",
      i->second.first);
}

TensorView* AliasAnalysisResult::getRoot(const TensorView* alias) const {
  return getOrDefault(alias_to_root_, alias);
}

void AliasAnalysisResult::finalize() {
  for (auto [alias, source_and_layout] : alias_to_source_) {
    auto [root, preferred_layout] = source_and_layout;

    // if alias has reuse buffer, it's an inplace update and shouldn't be marked
    // as an alias op, since we will have a set operation on it.
    if (alias->fusion()->getOutputAlias(alias).type ==
        AllocationType::ReuseBuffer) {
      continue;
    }

    // Walks up the `alias_to_source_` chain.
    while (true) {
      const auto i = alias_to_source_.find(root);
      if (i == alias_to_source_.end()) {
        break;
      }

      TensorView* source = i->second.first;
      if (source == root) {
        break;
      }
      root = source;
    }

    if (root != alias) {
      alias_to_root_[alias] = root;
    }
  }
}

Layout AliasAnalysisResult::preferredLayout(const TensorView* tv) const {
  auto i = alias_to_source_.find(tv);
  NVF_ERROR(
      i != alias_to_source_.end(),
      "`tv` must be an alias when calling preferredLayout: ",
      tv);

  return i->second.second;
}

Layout AliasAnalysisResult::createOrGetPreferredLayout(const TensorView* tv) {
  auto&& [source, layout] = alias_to_source_[tv];
  if (source == nullptr) {
    source = const_cast<TensorView*>(tv);
    layout = {tv->getMaybeAllocationDomain(), tv->getContiguity()};
  }
  return layout;
}

std::string AliasAnalysisResult::toString(const int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "Potential aliases:"
                          << (alias_to_source_.empty() ? " <empty>" : "")
                          << std::endl;
  for (const auto& [alias, source_and_layout] : alias_to_source_) {
    const auto& [source, layout] = source_and_layout;
    indent(ss, indent_size + 1)
        << ir_utils::varName(alias) << " is an alias of "
        << ir_utils::varName(source) << " if its layout is "
        << layout.toString() << std::endl;
  }
  indent(ss, indent_size) << "Finalized aliases:" << std::endl;
  for (const auto& [alias, root] : alias_to_root_) {
    indent(ss, indent_size + 1)
        << ir_utils::varName(alias) << " of allocation domain ["
        << toDelimitedString(alias->getAllocationDomain())
        << "] and logical domain ["
        << toDelimitedString(alias->getLogicalDomain())
        << "] is a transitive alias of " << ir_utils::varName(root)
        << std::endl;
  }
  return ss.str();
}

AliasAnalysisResult findAliases(
    Fusion* fusion,
    const EmptyAllocationAs empty_allocation_as) {
  AliasAnalysisResult analysis;
  AliasFinder finder(empty_allocation_as, analysis);
  // Fusion::exprs() computes and returns topological order.
  for (Expr* expr : fusion->exprs()) {
    // A potential improvement suggested by @tfogal: Let AliasFinder
    // return the AliasAnalysisResult instead of taking a mutable
    // `analysis` arg. This might be somewhat easily parallelizable
    // (albeit with a serialized merge step afterwards that inserts the
    // results).
    finder.dispatch(expr);
  }
  analysis.finalize();
  return analysis;
}

int64_t Layout::size() const {
  NVF_ERROR_EQ(allocation_domain.size(), contiguity.size());
  return std::ssize(allocation_domain);
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

namespace {
bool contiguityIsCompliant(
    const std::optional<bool>& actual,
    const std::optional<bool>& required) {
  if (actual == true && required == false) {
    return true;
  }
  return actual == required;
}
} // namespace

bool Layout::isCompliantWith(const Layout& required) const {
  if (required.allocation_domain.empty()) {
    return true;
  }

  if (allocation_domain != required.allocation_domain) {
    // This can be relaxed by allowing broadcast dimensions to be ordered
    // differently.
    return false;
  }

  for (const auto i : arange(allocation_domain.size())) {
    if (!contiguityIsCompliant(contiguity[i], required.contiguity[i])) {
      return false;
    }
  }
  return true;
}

} // namespace nvfuser
