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
#include <fusion.h>
#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <linked_hash_map.h>
#include <root_domain_map.h>

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
  AliasFinder(AliasAnalysisResult& analysis) : analysis_(analysis) {}

  void handle(const ViewOp*) override;
  void handle(const LoadStoreOp*) override;
  void handle(const SliceOp*) override;
  void handle(const BroadcastOp*) override;
  void handle(const SqueezeOp*) override;

 private:
  // A helper function used to compute the perferred output layout. It computes
  // the mapping from `in_rfactor` to `out_root` and applies that mapping to
  // `preferred_in_layout`. For many ops, this function returns a good initial
  // preferred output layout for aliasing because it tries to preserve the input
  // layout. An op (e.g. ViewOp and SliceOp) that transforms root to rfactor
  // using expressions will have to modify this initial layout so its allocation
  // domain will be a function of its rfactor domain.
  //
  // Returns `nullopt` if computation fails, so the caller can handle things
  // conservatively.
  static std::optional<Layout> mapInLayoutToOutRoot(
      const Layout& preferred_in_layout,
      TensorView* in,
      TensorView* out);

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

/*static*/ std::optional<Layout> AliasFinder::mapInLayoutToOutRoot(
    const Layout& preferred_in_layout,
    TensorView* in,
    TensorView* out) {
  if (!ir_utils::computePermutation(
           in->getMaybeRFactorDomain(), preferred_in_layout.allocation_domain)
           .has_value()) {
    // Give up when `in`'s allocation domain is not an rfactor permutation. As
    // an extension, we could map in_alloc to in_rfactor and apply the inverse
    // mapping to out_root.
    return std::nullopt;
  }

  std::unordered_map<IterDomain*, IterDomain*> in_rfactor_to_out_root =
      PairwiseRootDomainMap(in, out).mapProducerToConsumer();

  Layout preferred_out_layout;
  for (const auto i : c10::irange(preferred_in_layout.size())) {
    IterDomain* in_alloc_id = preferred_in_layout.allocation_domain[i];
    IterDomain* out_root_id = getOrDefault(in_rfactor_to_out_root, in_alloc_id);
    if (out_root_id == nullptr) {
      // This can happen when in_alloc_id is of type reduction or squeezed out.
      continue;
    }
    preferred_out_layout.allocation_domain.push_back(out_root_id);
    preferred_out_layout.contiguity.push_back(
        preferred_in_layout.contiguity[i]);
  }
  return preferred_out_layout;
}

void AliasFinder::handle(const ViewOp* view) {
  TensorView* in = view->in();
  TensorView* out = view->out();

  // Collect the allocation order of `in`'s rfactor domain and thus `out`'s root
  // domain.
  std::optional<Layout> out_root_layout =
      mapInLayoutToOutRoot(analysis_.preferredLayout(in), in, out);
  if (!out_root_layout.has_value()) {
    return;
  }

  LinkedHashMap<IterDomain*, std::optional<bool>> allocation_to_contiguity;
  for (const auto i : c10::irange(out_root_layout->size())) {
    allocation_to_contiguity.pushBack(
        out_root_layout->allocation_domain[i], out_root_layout->contiguity[i]);
  }

  // Replay `Expr`s from `out_root` to `out_rfactor` on
  // `allocation_to_contiguity`. Stop when an `Expr` requires a data copy;
  // otherwise generate the allocation order of `out_rfactor` and the
  // corresponding contiguity flags.
  std::unordered_map<IterDomain*, IterDomain*> out_root_to_in_rfactor =
      PairwiseRootDomainMap(in, out).mapConsumerToProducer();
  auto has_expanded_extent = [&out_root_to_in_rfactor](IterDomain* id) -> bool {
    // TODO(#1174): Preserve expanded extents in `out_root` so we don't have to
    // look for expanded extents in `in_rfactor`.
    if (const auto i = out_root_to_in_rfactor.find(id);
        i != out_root_to_in_rfactor.end()) {
      id = i->second;
    }
    return id->hasExpandedExtent();
  };

  const std::vector<IterDomain*>& out_root = out->getRootDomain();
  const std::vector<IterDomain*>& out_rfactor = out->getMaybeRFactorDomain();
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
          has_expanded_extent(merge->outer()),
          outer_contiguity,
          has_expanded_extent(merge->inner()),
          inner_contiguity);
      if (!mergeable) {
        return;
      }
      allocation_to_contiguity.insert(merge_i, merge->out(), contiguity);
    } else {
      NVF_ERROR(false, "Expect Split or Merge, but found: ", transform);
    }
  }

  Layout out_rfactor_layout;
  for (const auto& [allocation_id, contiguity] : allocation_to_contiguity) {
    out_rfactor_layout.allocation_domain.push_back(allocation_id);
    out_rfactor_layout.contiguity.push_back(contiguity);
  }
  analysis_.add(out, in, std::move(out_rfactor_layout));
}

void AliasFinder::handle(const LoadStoreOp* permute) {
  TensorView* in = dynamic_cast<TensorView*>(permute->in());
  if (in == nullptr) {
    return;
  }
  TensorView* out = permute->out()->as<TensorView>();

  // Compute `out`'s preferred allocation domain for aliasing.
  //
  // For example,
  //
  // in: rfactor=[i0,i1,i2], allocation=[i2,i0,i1]
  // out = permute(in, {1, 0, 2})
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
  std::optional<Layout> out_root_layout =
      mapInLayoutToOutRoot(analysis_.preferredLayout(in), in, out);
  if (!out_root_layout.has_value()) {
    return;
  }
  analysis_.add(out, in, std::move(*out_root_layout));
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
  std::unordered_map<IterDomain*, IterDomain*> out_root_to_rfactor;
  {
    const std::vector<IterDomain*>& out_rfactor = out->getMaybeRFactorDomain();
    const auto out_rank = out_root.size();
    NVF_ERROR(out_rfactor.size() == out_rank);
    out_root_to_rfactor.reserve(out_rank);
    for (auto i : c10::irange(out_rank)) {
      out_root_to_rfactor[out_root[i]] = out_rfactor[i];
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

    alloc_id = out_root_to_rfactor.at(alloc_id);

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

  analysis_.add(out, in, std::move(*out_layout));
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
  const std::vector<IterDomain*> out_rfactor = out->getMaybeRFactorDomain();
  for (const auto i : c10::irange(out_rfactor.size())) {
    if (bcast->isBroadcastDim(i)) {
      out_layout->allocation_domain.push_back(out_rfactor[i]);
      out_layout->contiguity.emplace_back(std::nullopt);
    }
  }

  analysis_.add(out, in, std::move(*out_layout));
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

  analysis_.add(out, in, std::move(*out_layout));
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

TensorView* AliasAnalysisResult::findNearestAliasedIo(
    TensorView* fusion_out) const {
  TensorView* root = fusion_out;
  do {
    const auto i = alias_to_source_.find(root);
    root = (i == alias_to_source_.end() ? nullptr : i->second.first);
  } while (root != nullptr && !root->isFusionInput() &&
           !root->isFusionOutput());
  return root;
}

TensorView* AliasAnalysisResult::getNearestAliasedIo(
    const TensorView* fusion_out) const {
  const auto i = out_to_root_.find(fusion_out);
  return i == out_to_root_.end() ? nullptr : i->second;
}

namespace {
bool okToRelayout(
    const TensorView* out,
    const Layout& new_layout,
    const bool can_override_empty_allocation_domain) {
  const std::vector<IterDomain*> out_allocation =
      (can_override_empty_allocation_domain ? out->getAllocationDomain()
                                            : out->getMaybeAllocationDomain());
  return new_layout.isCompliantWith({out_allocation, out->getContiguity()});
}
} // namespace

void AliasAnalysisResult::finalize(
    Fusion* fusion,
    const bool can_override_empty_allocation_domain) {
  for (TensorView* out :
       ir_utils::filterByType<TensorView>(fusion->outputs())) {
    TensorView* root = findNearestAliasedIo(out);
    if (root == nullptr) {
      continue;
    }

    const Layout preferred_layout = preferredLayout(out);
    if (!okToRelayout(
            out, preferred_layout, can_override_empty_allocation_domain)) {
      continue;
    }

    out_to_root_[out] = root;
  }
}

Layout AliasAnalysisResult::preferredLayout(const Val* v) const {
  const TensorView* tv = dynamic_cast<const TensorView*>(v);
  NVF_ERROR(tv != nullptr, "`v` is expected to be a TensorView. Found: ", v);

  if (auto i = alias_to_source_.find(tv); i != alias_to_source_.end()) {
    return i->second.second;
  }
  return {tv->getMaybeAllocationDomain(), tv->getContiguity()};
}

std::string AliasAnalysisResult::toString(const int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "All aliases:"
                          << (alias_to_source_.empty() ? " <empty>" : "")
                          << std::endl;
  for (const auto& [alias, source_and_layout] : alias_to_source_) {
    const auto& [source, layout] = source_and_layout;
    indent(ss, indent_size + 1)
        << ir_utils::varName(alias) << " is an alias of "
        << ir_utils::varName(source) << " if its layout is "
        << layout.toString() << std::endl;
  }
  indent(ss, indent_size) << "Output aliases only:"
                          << (out_to_root_.empty() ? " <empty>" : "")
                          << std::endl;
  for (const auto& [out, root] : out_to_root_) {
    indent(ss, indent_size + 1)
        << ir_utils::varName(out) << " is a transitive alias of "
        << ir_utils::varName(root) << std::endl;
  }
  return ss.str();
}

AliasAnalysisResult findAliases(
    Fusion* fusion,
    const bool can_override_empty_allocation_domain) {
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
  analysis.finalize(fusion, can_override_empty_allocation_domain);
  return analysis;
}

int64_t Layout::size() const {
  NVF_ERROR(allocation_domain.size() == contiguity.size());
  return static_cast<int64_t>(allocation_domain.size());
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

  for (const auto i : c10::irange(allocation_domain.size())) {
    if (!contiguityIsCompliant(contiguity[i], required.contiguity[i])) {
      return false;
    }
  }
  return true;
}

} // namespace nvfuser
