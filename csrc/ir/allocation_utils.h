// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <string>
#include <vector>

#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>

namespace nvfuser {

// With respect to the logical domain. `allocation_domain` must be a
// permutation of the corresponding logcial domain, and `contiguity` must be of
// the same length as `allocation`. See canonicalizeLayout for how we handle DID
// loop splits.
class Layout {
 public:
  Layout(
      std::vector<IterDomain*> allocation_domain,
      std::vector<std::optional<bool>> contiguity)
      : allocation_domain_(std::move(allocation_domain)),
        contiguity_(std::move(contiguity)) {
    NVF_ERROR_EQ(allocation_domain_.size(), contiguity_.size());
  }

  const std::vector<IterDomain*>& allocation_domain() const {
    return allocation_domain_;
  }

  IterDomain* allocation_domain(int64_t i) const {
    return allocation_domain_.at(i);
  }

  const std::vector<std::optional<bool>>& contiguity() const {
    return contiguity_;
  }

  std::optional<bool> contiguity(int64_t i) const {
    return contiguity_.at(i);
  }

  // The size of `allocation_domain` and therefore the size of `contiguity`.
  int64_t size() const;

  std::string toString(int indent_size = 0) const;

  // Returns a new Layout that has the same allocation domain but `true`
  // contiguity.
  Layout contiguous() const;

 private:
  std::vector<IterDomain*> allocation_domain_;
  std::vector<std::optional<bool>> contiguity_;
};

// Computes `Split`'s output contiguity. Returns the outer contiguity and then
// the inner contiguity.
std::pair<std::optional<bool>, std::optional<bool>> splitContiguity(
    const std::optional<bool>& contiguity);

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
    const std::optional<bool>& inner_contiguity);

// Given a TV, returns its layout with repsect to the logical domain. When
// `allocation` is a split of `logical`, walks backwards from `allocation` to
// `logical` to find a permutation of `logical` that satisfies the order in
// `allocation`. The returned contiguity is computed according to
// splitContiguity and mergeContiguity.
//
// Example:
//   input TV:
//     logical: [b, s, h]
//     allocation: [s, d, h/d, b]
//     contiguity: [t, t, f, t]
//   output layout:
//     allocation: [s, h, b]
//     contiguity: [t, f, t]
std::optional<Layout> canonicalizeLayout(const TensorView* tv);

// Returns whether `layout` is compliant with `required`. This is
// uni-directional. For example, `contiguity=[t,t]` is compliant with
// `contiguity=[f,f]` but not vice versa.
bool isCompliantWith(const Layout& layout, const Layout& required);

// A helper function used to compute the perferred output layout. It computes
// the mapping from `in_logical` to `out_root` and applies that mapping to
// `preferred_in_layout`. For many ops, this function returns a good initial
// preferred output layout for aliasing because it tries to preserve the input
// layout. An op (e.g. ReshapeOp and SliceOp) that transforms root to logical
// using expressions will have to modify this initial layout so its allocation
// domain will be a function of its logical domain.
//
// Returns `nullopt` if computation fails, so the caller can handle things
// conservatively.
std::optional<Layout> mapInLayoutToOutRoot(
    const std::optional<Layout>& in_layout,
    TensorView* in,
    TensorView* out);

} // namespace nvfuser
