// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <ir/all_nodes.h>
#include <iter_visitor.h>
#include <val_graph.h>
#include <val_graph_visitor.h>
#include <visibility.h>

namespace nvfuser {

//! See doc/reading/divisibility-of-split.md#predication
//! If an IterDomain is split and its inner output domain is
//! eventually split too, the second split must be divisible or the
//! inner domain must be predicated. This class finds Split
//! expressions that need to be divisible or predicated.
//!
//! Second splits are not limited to just direct output domains of
//! first splits but also indirect descendent domains as well.
//!
//! Predicating non-divisible split domains does not work if split
//! output domains are vectorized where ParallelType::Vectorize is
//! applied to an inner domain of splits. If it's non-divisible,
//! predicating the input domain of the non-divisible split results in
//! a vectoried operation is predicated out entirely since we do not
//! generate a fall-back non-vectorized else path. Runtime check is
//! done for those domains.
class NVF_API NonDivisibleSplitInfo : public IterVisitor {
 public:
  NonDivisibleSplitInfo(Fusion* fusion);

  const auto& splitsToPredicate() const {
    return splits_to_predicate_;
  }

  const auto& splitsToValidate() const {
    return splits_to_validate_;
  }

  // Check if a given tensor has non-divisible predicates
  bool hasPredicate(TensorView* tv) const;

 private:
  using IterVisitor::handle;

  void handle(Split* split) override;

  void handle(Merge* merge) override;

  //! True if reachable from inner domains of splits
  bool isReachableFromInnerDomains(IterDomain* id) const;

  //! Forward propagate the reachability information
  void propagateReachability(Split* split, bool is_protected);

  //! Forward propagate the reachability information
  void propagateReachability(Merge* merge);

  void clearReachability();

  //! Returns the extent of a split output domain if it's not proven to
  //! be divisible.
  Val* getMaybeNonDivisibleExtent(Split* split) const;

  //! Remove redundant predicates as divisibility may be validated at
  //! run time
  void removeRedundancy();

  //! Add validations to GpuLower::current()->validations()
  void addValidations();

 private:
  //! Split expressions whose input domain must be predicated
  std::unordered_map<TensorView*, std::vector<Split*>> splits_to_predicate_;
  //! Split expressions whose divisibility must be validated at run
  //! time. This is not a complete set of all splits that must be
  //! divisible but just includes those discovered during the this
  //! analysis. As it is not a complete set, it is unclear if keeping
  //! track of these splits actually helps.
  std::unordered_set<Split*> splits_to_validate_;

  //! Temporarily used for analyzing each tensor
  TensorView* current_tv_ = nullptr;
  std::unordered_set<IterDomain*> inner_domains_;
};

// This class is meant to replace NonDivisibleSplitInfo.
//
// Traverse the indexing path of all tensors and find all IDs that
// need to be predicated due to non-divisible splits. The last IDs of
// the traversal path are predicated, so not all non-divisible splits need to be
// predicated. More specifically, when index propagation through an expr uses
// the extent of an input of the extent and the index of the input is not
// guaranteed to be within the valid range, the input ID needs to be predicated.
//
// For example, consider the below split:
//
// Logical: [i0(10)]
// split i0(10) by 3 -> i1(4), i2(3)
// Loop: [i1(4), i2(3)]
//
// Suppose i0 is the sole logical ID and {i1, i2} is the loop
// domain. This split is a non-divisible split, but we don't need to
// have any additional predicate as i0 is going to be predicated
// anyway since it's a logical ID.
//
// Consider the following:
//
// Logical: [i0(10)]
// split i0(10) by 3 -> i1(4), i2(3)
// split i2(3) by 5 -> i3(1), i4(5)
// Loop: [i1(4), i3(1), i4(5)]
//
// Now i2 needs to be predicated as its index may be larger than 3
// due to the non-divisible split to i3 and i4.
//
// This is not limited to split, but it may be required for merge as
// well.
//
// Root: [i0(4), i1(2)]
// Logical: [i2(8)]
// merge i0(4), i1(2) -> i2(8)
// split i1(2) by 5 -> i3(1), i4(5)
// Loop: [i0(4), i3(1), i4(5)]
//
// The index of i1 may be larger than its extent,
// and the index propagation through the merge still uses the
// extent of 2 to construct the index of i2 as (index_of_i0 * 2 +
// index_of_i1). Here, if the i1 index exceeds 2, i2 may be
// redundantly indexed. See
// PredicateIndexingTest.NonDivisibleSplitWithNonLogicalToLoopDomains
// for a concrete example that would fail without this analysis.
class NonDivisiblePredicateInfo {
 public:
  NonDivisiblePredicateInfo(Fusion* fusion);

  const std::unordered_map<TensorView*, std::vector<ValGroup>>& idsToPredicate()
      const {
    return ids_to_predicate_;
  }

  // Check if a given tensor has non-divisible predicates
  bool hasPredicate(TensorView* tv) const;

  // Get all IDs that need to be predicated due to non-divisible
  // splits in the given indexing path
  static std::vector<ValGroup> getNonDivisibleSplitsToPredicate(
      const ValGraph& graph,
      const ValGraphBFS::ExprPath& indexing_path);

 private:
  // ID groups of each tensor that need to be predicated in addition
  // to the default predicates
  std::unordered_map<TensorView*, std::vector<ValGroup>> ids_to_predicate_;
};

} // namespace nvfuser
