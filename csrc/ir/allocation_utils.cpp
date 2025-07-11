// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/allocation_utils.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <linked_hash_map.h>
#include <logical_domain_map.h>

namespace nvfuser {

int64_t Layout::size() const {
  NVF_ERROR_EQ(allocation_domain_.size(), contiguity_.size());
  return std::ssize(allocation_domain_);
}

std::string Layout::toString(const int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "<allocation=["
                          << toDelimitedString(allocation_domain_)
                          << "], contiguity=["
                          << toDelimitedString(contiguity_, /*delim=*/" ")
                          << "]>";
  return ss.str();
}

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

std::optional<Layout> canonicalizeLayout(const TensorView* tv) {
  const std::vector<IterDomain*>& logical = tv->getLogicalDomain();
  const std::vector<IterDomain*>& allocation = tv->getMaybeAllocationDomain();
  LinkedHashMap<IterDomain*, std::optional<bool>> allocation_to_contiguity;
  for (auto&& [alloc_id, contiguity] : zip(allocation, tv->getContiguity())) {
    allocation_to_contiguity.pushBack(alloc_id, contiguity);
  }

  for (Expr* transform : DependencyCheck::getAllExprsBetween(
                             {logical.begin(), logical.end()},
                             {allocation.begin(), allocation.end()}) |
           std::views::reverse) {
    auto* split = dynamic_cast<Split*>(transform);
    if (split == nullptr) {
      // We can handle merges using a similar logic if/when we need to.
      return std::nullopt;
    }

    // When split->outer() is parallelized and split->inner() is serial, we
    // remove split->outer() regardless of its position and replace
    // split->inner() with split->in(). This way, even when split->outer() is
    // not adjacent to split->inner() (e.g. when it's outermost), we can
    // still undo the split.
    //
    // Several other cases that I haven't implemented for simplicity.
    //
    // When split->outer() is serial and split->inner() is parallelized, we
    // could remove split->inner() and replace split->outer() with
    // split->in() regardless of split->inner()'s position.
    //
    // When split->outer() and split->inner() are both parallelized, we could
    // replace either of them with split->in() and remove the other.
    NVF_ERROR(!split->inner()->isParallelized());

    const auto [outer_contiguity, next_i] =
        allocation_to_contiguity.erase(split->outer());
    if (!split->outer()->isParallelized()) {
      // Check adjacency only if split->outer() is not parallelized.
      if (next_i == allocation_to_contiguity.end() ||
          next_i->first != split->inner()) {
        return std::nullopt;
      }
    }
    const auto [inner_contiguity, merge_i] =
        allocation_to_contiguity.erase(split->inner());
    const auto [mergeable, contiguity] = mergeContiguity(
        split->outer()->hasExpandedExtent(),
        outer_contiguity,
        split->inner()->hasExpandedExtent(),
        inner_contiguity);
    if (!mergeable) {
      return std::nullopt;
    }
    allocation_to_contiguity.insert(merge_i, split->in(), contiguity);
  }

  std::vector<IterDomain*> canonicalized_allocation;
  canonicalized_allocation.reserve(allocation.size());
  std::vector<std::optional<bool>> canonicalized_contiguity;
  canonicalized_contiguity.reserve(allocation.size());
  for (auto&& [alloc_id, contiguity] : allocation_to_contiguity) {
    canonicalized_allocation.push_back(alloc_id);
    canonicalized_contiguity.push_back(contiguity);
  }

  NVF_ERROR(
      std::is_permutation(
          tv->getLogicalDomain().begin(),
          tv->getLogicalDomain().end(),
          canonicalized_allocation.begin(),
          canonicalized_allocation.end()),
      "This indicates that logical and allocation are not connected via "
      "transforms. This is most often caused by forgetting to concretize "
      "a fusion with dynamic reshapes.");

  return Layout(
      std::move(canonicalized_allocation), std::move(canonicalized_contiguity));
}

Layout Layout::contiguous() const {
  return Layout(
      allocation_domain(),
      TensorDomain::getContiguityFilledWith(allocation_domain(), true));
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

bool isCompliantWith(const Layout& layout, const Layout& required) {
  if (layout.allocation_domain() != required.allocation_domain()) {
    // This can be relaxed by allowing broadcast dimensions to be ordered
    // differently.
    return false;
  }

  for (const auto i : arange(layout.size())) {
    if (!contiguityIsCompliant(layout.contiguity(i), required.contiguity(i))) {
      return false;
    }
  }
  return true;
}

std::optional<Layout> mapInLayoutToOutRoot(
    const std::optional<Layout>& in_layout,
    TensorView* in,
    TensorView* out) {
  if (!in_layout.has_value()) {
    return std::nullopt;
  }

  if (!ir_utils::computePermutation(
           in->getLogicalDomain(), in_layout->allocation_domain())
           .has_value()) {
    // Give up when `in`'s allocation domain is not an logical permutation. As
    // an extension, we could map in_alloc to in_logical and apply the inverse
    // mapping to out_root.
    return std::nullopt;
  }

  std::unordered_map<IterDomain*, IterDomain*> in_logical_to_out_root =
      PairwiseLogicalDomainMap(in, out).mapProducerToConsumer();

  std::vector<IterDomain*> out_allocation;
  out_allocation.reserve(in_layout->size());
  std::vector<std::optional<bool>> out_contiguity;
  out_contiguity.reserve(in_layout->size());
  for (auto&& [in_alloc_id, contiguity] :
       zip(in_layout->allocation_domain(), in_layout->contiguity())) {
    IterDomain* out_root_id = getOrDefault(in_logical_to_out_root, in_alloc_id);
    if (out_root_id == nullptr) {
      // This can happen when in_alloc_id is of type reduction or squeezed out.
      continue;
    }
    out_allocation.push_back(out_root_id);
    out_contiguity.push_back(contiguity);
  }
  return Layout(std::move(out_allocation), std::move(out_contiguity));
}

} // namespace nvfuser
