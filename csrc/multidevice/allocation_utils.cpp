// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <multidevice/allocation_utils.h>

#include <algorithm>
#include <optional>
#include <ranges>
#include <vector>

#include <exceptions.h>
#include <ir/interface_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/utils.h>
#include <multidevice/utils.h>

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

} // namespace nvfuser
