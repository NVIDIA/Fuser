// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "multidevice/allocation_utils.h"

#include <algorithm>
#include <optional>
#include <ranges>
#include <unordered_set>
#include <vector>

#include "base.h"
#include "exceptions.h"
#include "ir/allocation_utils.h"
#include "ir/interface_nodes.h"
#include "ir/internal_nodes.h"
#include "ir/utils.h"
#include "linked_hash_map.h"
#include "multidevice/utils.h"

namespace nvfuser {

bool isTvContiguous(const TensorView* tv) {
  // Reduction and broadcast axis do not have a contiguity value.
  return std::all_of(
      tv->getContiguity().begin(),
      tv->getContiguity().end(),
      [](std::optional<bool> c) { return c.value_or(true); });
}

IterDomain* projectShardedAllocationToLogical(
    TensorView* tv,
    IterDomain* allocation_id) {
  if (allocation_id == nullptr) {
    return nullptr;
  }

  std::vector<Expr*> exprs = DependencyCheck::getAllExprsBetween(
      {tv->getLogicalDomain().begin(), tv->getLogicalDomain().end()},
      {allocation_id});

  IterDomain* logical_id = allocation_id;
  for (Expr* expr : exprs | std::views::reverse) {
    NVF_ERROR(
        isValidDeviceSplit(expr), "invalid device split: ", expr->toString());
    logical_id = expr->as<Split>()->in();
  }
  return logical_id;
}

IterDomain* projectLogicalToShardedAllocation(
    TensorView* tv,
    IterDomain* logical_id) {
  if (logical_id == nullptr) {
    return nullptr;
  }

  std::vector<Expr*> exprs = DependencyCheck::getAllExprsBetween(
      {logical_id},
      {tv->getMaybeAllocationDomain().begin(),
       tv->getMaybeAllocationDomain().end()});
  IterDomain* allocation_id = logical_id;
  for (auto expr : exprs) {
    NVF_ERROR(
        isValidDeviceSplit(expr), "invalid device split: ", expr->toString());
    allocation_id = expr->as<Split>()->inner();
  }
  return allocation_id;
}

void shardAllocationAsLoop(
    TensorView* tv,
    const std::unordered_set<ParallelType>& parallel_types) {
  LinkedHashMap<IterDomain*, std::optional<bool>> allocation_to_contiguity;
  for (const auto&& [id, contiguity] :
       zip(tv->getMaybeAllocationDomain(), tv->getContiguity())) {
    allocation_to_contiguity.pushBack(id, contiguity);
  }

  auto loop_ids_to_replicate =
      tv->getLoopDomain() | std::views::filter([&](IterDomain* id) {
        return parallel_types.count(id->getParallelType()) > 0;
      });

  // Allocation domain should be a permutation of logical domain at this point.
  std::vector<Expr*> transforms = DependencyCheck::getAllExprsBetween(
      {tv->getMaybeAllocationDomain().begin(),
       tv->getMaybeAllocationDomain().end()},
      {loop_ids_to_replicate.begin(), loop_ids_to_replicate.end()});

  for (auto* e : transforms) {
    auto* split = dynamic_cast<Split*>(e);
    NVF_ERROR(
        split != nullptr,
        "Expected all transform exprs to be a split between allocation and "
        "loop domain during sharding propagation.");
    const auto [contiguity, split_i] =
        allocation_to_contiguity.erase(split->in());
    auto [outer_contiguity, inner_contiguity] = splitContiguity(contiguity);
    allocation_to_contiguity.insert(split_i, split->outer(), outer_contiguity);
    allocation_to_contiguity.insert(split_i, split->inner(), inner_contiguity);
  }

  std::vector<IterDomain*> new_allocation_domain;
  std::vector<std::optional<bool>> new_contiguity;
  {
    new_allocation_domain.reserve(allocation_to_contiguity.size());
    new_contiguity.reserve(allocation_to_contiguity.size());

    for (auto [id, contiguity] : allocation_to_contiguity) {
      new_allocation_domain.push_back(id);
      new_contiguity.push_back(contiguity);
    }
  }
  tv->setAllocationDomain(new_allocation_domain, new_contiguity);
}

} // namespace nvfuser
