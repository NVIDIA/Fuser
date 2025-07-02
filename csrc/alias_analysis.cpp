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

bool okToRelayout(
    const TensorView* tv,
    const Layout& new_layout,
    const EmptyAllocationAs empty_allocation_as) {
  if (empty_allocation_as == EmptyAllocationAs::kUndetermined &&
      !tv->hasAllocation()) {
    return true;
  }

  std::optional<Layout> old_layout = canonicalizeLayout(tv);
  if (!old_layout.has_value()) {
    return false;
  }
  return isCompliantWith(new_layout, *old_layout);
}

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
      mapInLayoutToOutRoot(analysis_.preferredLayout(in), in, out);
  if (!out_root_layout.has_value()) {
    return;
  }

  LinkedHashMap<IterDomain*, std::optional<bool>> allocation_to_contiguity;
  for (const auto i : arange(out_root_layout->size())) {
    if (!out_root_layout->contiguity(i).has_value() &&
        !out_root_layout->allocation_domain(i)->isBroadcast()) {
      // TODO(#1126): Due to #1126, `out_root` materializes an expanded
      // broadcast IterDomain from `in_logical` when `view` splits or merges
      // that IterDomain. We return no alias when this happen; otherwise
      // AliasTest.MergeBroadcastsBetweenConcretes would fail.
      return;
    }
    allocation_to_contiguity.pushBack(
        out_root_layout->allocation_domain(i), out_root_layout->contiguity(i));
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

  std::vector<IterDomain*> out_allocation;
  out_allocation.reserve(allocation_to_contiguity.size());
  std::vector<std::optional<bool>> out_contiguity;
  out_contiguity.reserve(allocation_to_contiguity.size());
  for (auto&& [alloc_id, contiguity] : allocation_to_contiguity) {
    out_allocation.push_back(alloc_id);
    out_contiguity.push_back(contiguity);
  }

  aliasIfCompliant(
      out, in, Layout(std::move(out_allocation), std::move(out_contiguity)));
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
      mapInLayoutToOutRoot(analysis_.preferredLayout(in), in, out);
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
      mapInLayoutToOutRoot(analysis_.preferredLayout(in), in, out);
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
  std::vector<IterDomain*> out_allocation = out_layout->allocation_domain();
  std::vector<std::optional<bool>> out_contiguity = out_layout->contiguity();
  bool next_non_broadcast_is_non_contiguous = false;
  for (const auto i : arange(out_layout->size()) | std::views::reverse) {
    IterDomain*& alloc_id = out_allocation[i];
    std::optional<bool>& contiguity = out_contiguity[i];

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

  aliasIfCompliant(
      out, in, Layout(std::move(out_allocation), std::move(out_contiguity)));
}

void AliasFinder::handle(const BroadcastOp* bcast) {
  TensorView* in = dynamic_cast<TensorView*>(bcast->in());
  if (in == nullptr) {
    return;
  }
  auto* out = bcast->out()->as<TensorView>();

  std::optional<Layout> out_layout =
      mapInLayoutToOutRoot(analysis_.preferredLayout(in), in, out);
  if (!out_layout.has_value()) {
    return;
  }

  // Put new, broadcast dimensions to the end.
  std::vector<IterDomain*> out_allocation = out_layout->allocation_domain();
  std::vector<std::optional<bool>> out_contiguity = out_layout->contiguity();
  const std::vector<IterDomain*> out_logical = out->getLogicalDomain();
  for (const auto i : arange(out_logical.size())) {
    if (bcast->isBroadcastDim(i)) {
      out_allocation.push_back(out_logical[i]);
      out_contiguity.emplace_back(std::nullopt);
    }
  }

  aliasIfCompliant(
      out, in, Layout(std::move(out_allocation), std::move(out_contiguity)));
}

void AliasFinder::handle(const SqueezeOp* squeeze) {
  TensorView* in = dynamic_cast<TensorView*>(squeeze->in());
  if (in == nullptr) {
    return;
  }
  auto* out = squeeze->out()->as<TensorView>();

  // Preserve the allocation order of existing dimensions.
  std::optional<Layout> out_layout =
      mapInLayoutToOutRoot(analysis_.preferredLayout(in), in, out);
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
      mapInLayoutToOutRoot(analysis_.preferredLayout(in), in, out);
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
      root = i->second.first;
    }
    alias_to_root_[alias] = root;
  }
}

std::optional<Layout> AliasAnalysisResult::preferredLayout(
    const TensorView* tv) const {
  if (auto i = alias_to_source_.find(tv); i != alias_to_source_.end()) {
    return i->second.second;
  }

  return canonicalizeLayout(tv);
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

} // namespace nvfuser
