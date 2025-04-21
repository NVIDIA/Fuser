// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <id_model/id_model.h>

namespace nvfuser {

// Get the domains to predicate for a given tensor used as a consumer
// of a given expr.
std::vector<IterDomain*> getPredicateDomains(
    TensorView* consumer_tv,
    const Expr* expr);

// Get a replace map for predicate indexing of a given tensor appearing
// in a given loop-nest.
//
// The unswitched_loop parameter is an optional ForLoop that is used
// when this predicate is for an unswitched, unrolled or vectorized
// loop.
std::unordered_map<Val*, Val*> getPredicateIndexReplacementMap(
    TensorView* tv,
    const std::vector<ForLoop*>& for_loops,
    const std::unordered_map<ValGroup, Val*>& index_map,
    const ValGraph& traversal_graph,
    const ExprPath<ExprGroup>& traversal_path,
    const IdModel& id_model,
    bool is_start_predicate,
    ForLoop* unswitched_loop = nullptr);

// Check if a given ExprGroup is a split that needs an additional
// predicate due to its non-divisibility.
inline bool isNonDivisibleSplit(const ExprGroup& expr_group) {
  if (!expr_group->front()->isA<Split>()) {
    return false;
  }

  const auto& non_divisible_split_info =
      GpuLower::current()->nonDivisibleSplitInfo();

  std::vector<PredicateDomainInfo> pred_info_vec;

  // The splitsToPredicate map is for each tensor. Here, it's assumed
  // that if any tensor has a non-divisible split that's mapped with
  // expr_group, it should be considered a non-divisible split. This
  // may result in duplicate predicates, which should be removed by
  // the expression simplifier.
  //
  // For example, suppose tv0 is a 1D tensor of size 16:
  //
  // auto tv1 = reshape(tv0, {16}, {2, 8});
  // tv1->split(1, 3);
  //
  // propagate_transformation(to: tv0, from: tv1)
  //
  // Here, the split by 3 of tv1 is not included in its non-divisible
  // split list even though it is indeed non-divisible. The reason is
  // that the input to the non-divisible split, the inner logical
  // domain of extent 8, is predicated anyway since it's a logical
  // domain. Specifically, its predicate should consist of:
  //
  // - Predicate for the outer logical domain
  // - Predicate for the inner logical domain
  //
  // However, for tv0, it is indeed included in the non-divisible
  // split list since the domain of extent 8 is not part
  // of its logical domain.
  //
  // - Predicate for the sole logical domain
  // - Predicate for the non-divisible split
  //
  // This would mean that when generating a predicate for tv1, since
  // the below check would find a mapping with the non-divisible split
  // for tv0, the tv1 predicate would be:
  //
  // - Predicate for the outer logical domain
  // - Predicate for the inner logical domain
  // - Predicate for the non-divisible split
  //
  // Here, the last two predicates are redundant since both of them
  // guard the index with respect to the domain of extent 8, which is
  // redundant. This is a bit annonying but should have no actual
  // impact as the redundancy should be removed by the expression
  // simplifier.
  for (const auto& [tv, splits] :
       non_divisible_split_info.splitsToPredicate()) {
    if (std::find_if(splits.begin(), splits.end(), [&](Split* split) {
          return expr_group->has(split);
        }) != splits.end()) {
      return true;
    }
  }

  return false;
}

} // namespace nvfuser
