// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <non_divisible_split.h>

namespace nvfuser {

void NonDivisibleSplitInfo::build(Fusion* fusion) {
  const auto vals = fusion->usedMathVals();
  auto tvs = ir_utils::filterByType<TensorView>(vals);

  // Find all non-divisible splits
  for (auto tv : tvs) {
    if (tv->isFusionInput()) {
      continue;
    }
    const std::vector<Val*> domain_vals(
        tv->getLeafDomain().begin(), tv->getLeafDomain().end());
    current_tv_ = tv;
    clearReachability();
    traverseTo(fusion, domain_vals);
    current_tv_ = nullptr;
  }

  if (GpuLower::current() != nullptr) {
    removeRedundancy();
  }
}

void NonDivisibleSplitInfo::handle(Split* split) {
  if (split->in()->isBroadcast()) {
    return;
  }

  // Indicates if this split is going to be either predicated or
  // validated at run time
  bool is_protected = false;

  if (isReachableFromInnerDomains(split->in())) {
    // check if this split may be non-divisible
    auto maybe_non_divisible_extent = getMaybeNonDivisibleExtent(split);
    if (maybe_non_divisible_extent) {
      // If the outputs are vectorized, predication isn't
      // sufficient, it must be divisible.
      TORCH_INTERNAL_ASSERT(
          split->outer()->getParallelType() != ParallelType::Vectorize);
      if (split->inner()->getParallelType() == ParallelType::Vectorize) {
        splits_to_validate_.insert(split);
      } else {
        // Not proven to be a divisible split
        auto gpu_lower = GpuLower::current();
        TORCH_INTERNAL_ASSERT(gpu_lower != nullptr);

        // If we know this split must be divisible, it's either validated as
        // above, exact matches to a case matching the above, or exact matches
        // to a transformation from view which must be divisible.
        if (gpu_lower->divisibleSplitSet().find(split) ==
            gpu_lower->divisibleSplitSet().end()) {
          splits_to_predicate_[current_tv_].push_back(split);
        }
      }

      is_protected = true;
    }
  }

  propagateReachability(split, is_protected);
}

bool NonDivisibleSplitInfo::isReachableFromInnerDomains(IterDomain* id) const {
  return inner_domains_.find(id) != inner_domains_.end();
}

void NonDivisibleSplitInfo::clearReachability() {
  inner_domains_.clear();
}

void NonDivisibleSplitInfo::propagateReachability(
    Split* split,
    bool is_protected) {
  // Propagate down the reachability information. Descendants of the
  // inner domain must be tracked.
  inner_domains_.insert(split->inner());

  // If this split itself is reachable, propagate the reachability to
  // the outer output as well. However, if this split is protected,
  // i.e., either predicated or validated, any potential effect by
  // descendants of the outer domain is taken care by the predicate or
  // run-time check of this split, so checking outer descendants isn't
  // required.
  if (isReachableFromInnerDomains(split->in()) && !is_protected) {
    inner_domains_.insert(split->outer());
  }
}

Val* NonDivisibleSplitInfo::getMaybeNonDivisibleExtent(Split* split) const {
  c10::optional<int64_t> in_extent;
  if (split->in()->extent()->isConstInt()) {
    in_extent = split->in()->extent()->evaluateInt();
  }

  c10::optional<int64_t> factor;
  if (split->factor()->isConstInt()) {
    factor = split->factor()->evaluateInt();
  }

  if (in_extent.has_value() && factor.has_value() &&
      in_extent.value() % factor.value() == 0) {
    return nullptr;
  }

  // even if the extent size is unknown, if the factor is known to
  // be 1, it's always divisible
  if (factor.has_value() && factor.value() == 1) {
    return nullptr;
  }

  auto ceildiv_dom = split->innerSplit() ? split->outer() : split->inner();
  return ceildiv_dom->extent();
}

void NonDivisibleSplitInfo::handle(Merge* merge) {
  propagateReachability(merge);
}

void NonDivisibleSplitInfo::propagateReachability(Merge* merge) {
  // Inner input index never exceeds its extent as it's computed as an
  // remainder. Outer may do.
  if (isReachableFromInnerDomains(merge->outer())) {
    inner_domains_.insert(merge->out());
  }
}

void NonDivisibleSplitInfo::removeRedundancy() {
  auto gpu_lower = GpuLower::current();
  TORCH_INTERNAL_ASSERT(gpu_lower != nullptr);

  std::unordered_set<IterDomain*> split_to_validate_outer;
  for (auto it = splits_to_validate_.begin();
       it != splits_to_validate_.end();) {
    auto outer_concrete = gpu_lower->caMap()->getConcreteMappedID(
        (*it)->outer(), IdMappingMode::EXACT);
    auto new_domain = split_to_validate_outer.insert(outer_concrete).second;
    if (!new_domain) {
      it = splits_to_validate_.erase(it);
    } else {
      ++it;
    }
  }

  // If validated by runtime checks, no need to predicate
  for (auto& kv : splits_to_predicate_) {
    auto& splits = kv.second;
    for (auto it = splits.begin(); it != splits.end();) {
      // If the outer domain is mapped with the outer domain of any
      // validated domain, it is safe to omit the predicate for the
      // split.
      Split* split_to_predicate = *it;
      if (std::any_of(
              splits_to_validate_.begin(),
              splits_to_validate_.end(),
              [&](Split* split_to_validate) {
                return gpu_lower->caMap()->areMapped(
                    split_to_validate->outer(),
                    split_to_predicate->outer(),
                    IdMappingMode::EXACT);
              })) {
        it = splits.erase(it);
      } else {
        ++it;
      }
    }
  }
}

} // namespace nvfuser
