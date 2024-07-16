// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/indexing_utils.h>
#include <id_model/predicate_indexing.h>

namespace nvfuser {

std::vector<IterDomain*> getPredicateDomains(
    TensorView* consumer_tv,
    const Expr* expr) {
  // Logical domains should be the domains to predicate as they define
  // the logical shape of a tensor. However, in the case of rfactored
  // reductions, rfactor splits may not be divisible, thus root
  // domains need to be predicated. Note that the non-divisible split
  // info does not seem to cover non-divisible reduction rfactor
  // splits.
  std::vector<IterDomain*> predicate_domains = consumer_tv->hasReduction()
      ? consumer_tv->getMaybeRootDomain()
      : consumer_tv->getLogicalDomain();

  // Broadcast domains should not need to be predicated
  predicate_domains.erase(
      std::remove_if(
          predicate_domains.begin(),
          predicate_domains.end(),
          [](IterDomain* id) -> bool { return id->isBroadcast(); }),
      predicate_domains.end());

  // If this is an expr initializing a buffer for a reduction, the
  // reduction domains do not need to be predicated. In fact, if it's
  // a Local tensor, no predicate is necessary at all
  if (lower_utils::isReductionInitExpr(expr)) {
    VERBOSE() << "Reduction init expr: " << expr->toString();
    if (consumer_tv->getMemoryType() == MemoryType::Local) {
      return {};
    } else {
      predicate_domains.erase(
          std::remove_if(
              predicate_domains.begin(),
              predicate_domains.end(),
              [](IterDomain* id) -> bool { return id->isReduction(); }),
          predicate_domains.end());
    }
  }

  return predicate_domains;
}

} // namespace nvfuser
