// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/analysis/non_divisible_split.h>
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <expr_simplifier.h>
#include <ir/iostream.h>
#include <ir/utils.h>

namespace nvfuser {

NonDivisibleSplitInfo::NonDivisibleSplitInfo(Fusion* fusion) {
  const auto vals = fusion->usedMathVals();
  auto tvs = ir_utils::filterByType<TensorView>(vals);

  // Find all non-divisible splits
  for (auto tv : tvs) {
    if (tv->isFusionInput()) {
      continue;
    }
    const std::vector<Val*> domain_vals(
        tv->getLoopDomain().begin(), tv->getLoopDomain().end());
    current_tv_ = tv;
    clearReachability();
    traverseTo(domain_vals);
    current_tv_ = nullptr;
  }

  if (GpuLower::current() != nullptr) {
    removeRedundancy();
    addValidations();
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
      NVF_ERROR(split->outer()->getParallelType() != ParallelType::Vectorize);
      if (split->inner()->getParallelType() == ParallelType::Vectorize) {
        splits_to_validate_.insert(split);
      } else {
        // Not proven to be a divisible split
        auto gpu_lower = GpuLower::current();
        NVF_ERROR(gpu_lower != nullptr);

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
  std::optional<int64_t> in_extent;
  if (split->in()->extent()->isConstInt()) {
    in_extent = split->in()->extent()->evaluate().as<int64_t>();
  }

  std::optional<int64_t> factor;
  if (split->factor()->isConstInt()) {
    factor = split->factor()->evaluate().as<int64_t>();
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
  NVF_ERROR(gpu_lower != nullptr);

  std::unordered_set<IterDomain*> split_to_validate_outer;
  for (auto it = splits_to_validate_.begin();
       it != splits_to_validate_.end();) {
    auto outer_concrete = gpu_lower->info().caMap()->getConcreteMappedID(
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
                return gpu_lower->info().caMap()->areMapped(
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

void NonDivisibleSplitInfo::addValidations() {
  for (auto split : splits_to_validate_) {
    auto extent = split->in()->extent();
    auto factor = split->factor();
    auto is_divisible = SimplifyingIrBuilder::eqExpr(
        SimplifyingIrBuilder::modExpr(extent, factor),
        extent->fusion()->zeroVal());
    NVFUSER_LOWER_VALIDATE(
        is_divisible, "Non-divisible split detected: ", split);
  }
}

bool NonDivisibleSplitInfo::hasPredicate(TensorView* tv) const {
  auto it = splitsToPredicate().find(tv);
  return it != splitsToPredicate().end() && !(it->second.empty());
}

NonDivisiblePredicateInfo::NonDivisiblePredicateInfo(Fusion* fusion) {
  auto gpu_lower = GpuLower::current();
  NVF_ERROR(gpu_lower != nullptr, "GpuLower is requred");
  NVF_ERROR(gpu_lower->isTensorIndexerEnabled(), "TensorIndexer is required");

  const auto& tensor_indexer = gpu_lower->tensorIndexer();

  for (auto tv : fusion->allTvs()) {
    if (tv->isFusionInput()) {
      continue;
    }

    auto def = tv->definition();
    if (def == nullptr) {
      continue;
    }

    const auto path = tensor_indexer.getPredicateIndexingPath(tv, def);

    ids_to_predicate_.emplace(
        tv,
        getNonDivisibleSplitsToPredicate(
            tensor_indexer.traversalGraph(), path));
  }
}

namespace {

// Consider create a proper dispatcher for ExprPath (i.e., ValGroup + Direction)
class IndexingPathAnalysis : public OptInDispatch {
 public:
  IndexingPathAnalysis(
      const ValGraph& graph,
      const ValGraphBFS::ExprPath& indexing_path)
      : graph_(graph) {
    // Keep track of unsafe IDs whose indices may not be in the valid
    // range. When the extent of an unsafe ID is used in the index
    // propagation, that ID needs to be predicated.
    for (const auto& [expr_g, dir] : indexing_path) {
      current_direction_ = dir;
      dispatch(expr_g->front());
      current_direction_ = Direction::Undefined;
    }
  }

  void handle(Split* split) {
    const auto in_group = graph_.toGroup(split->in());
    const auto inner_group = graph_.toGroup(split->inner());
    const auto outer_group = graph_.toGroup(split->outer());

    if (currentDirection() == Direction::Forward) {
      // In the case of propagating through a forward split, if the
      // input is unsafe. The output is going to be unsafe, unless
      // the input and inner output are mapped. See
      // IdGraphIndexCompute::handle(Split*).
      if (unsafe_groups_.contains(in_group)) {
        if (in_group == inner_group) {
          unsafe_groups_.emplace(inner_group);
        } else {
          unsafe_groups_.emplace(outer_group);
        }
      }
    } else {
      // As we are using the AlmostExact graph, the input ID should
      // never be grouped together with any of output IDs since the
      // BFS traversal should never pick up such expressions.
      NVF_ERROR(in_group != inner_group);
      NVF_ERROR(in_group != outer_group);

      // In the case of backward traversal, if the inner ID is
      // unsafe, it can no longer be kept unsafe but needs to be
      // predicated
      if (unsafe_groups_.contains(inner_group)) {
        groups_to_predicate_.emplace_back(inner_group);
      }

      // Check if split->in is unsafe. It's unsafe when this split
      // is not divisible. It's also unsafe when the outer output is
      // unsafe. The inner output does not matter as it is
      // predicated if it's unsafe.

      auto gpu_lower = GpuLower::current();
      NVF_ERROR(gpu_lower != nullptr);

      bool is_divisible_split = gpu_lower->divisibleSplitSet().find(split) !=
          gpu_lower->divisibleSplitSet().end();

      if (!is_divisible_split) {
        auto extent = split->in()->extent();
        auto factor = split->factor();
        if (extent->isConstScalar() && factor->isConstScalar() &&
            (extent->evaluate().as<int64_t>() %
                 factor->evaluate().as<int64_t>() ==
             0)) {
          is_divisible_split = true;
        }
      }

      if (unsafe_groups_.contains(outer_group) || !is_divisible_split) {
        unsafe_groups_.emplace(in_group);
      }
    }
  }

  void handle(Merge* merge) {
    const auto inner_group = graph_.toGroup(merge->inner());
    const auto outer_group = graph_.toGroup(merge->outer());
    const auto out_group = graph_.toGroup(merge->out());

    if (currentDirection() == Direction::Forward) {
      // Similar to the backward split, these groups should not be the same
      NVF_ERROR(out_group != inner_group);
      NVF_ERROR(out_group != outer_group);

      // In the case of propagating through a forward merge, if the
      // inner is unsafe, it needs to be predicated as the inner
      // output of the backward split case. The output of the merge
      // is unsafe if the outer input is unsafe.
      if (unsafe_groups_.contains(inner_group)) {
        groups_to_predicate_.emplace_back(inner_group);
      }
      if (unsafe_groups_.contains(outer_group)) {
        unsafe_groups_.emplace(out_group);
      }
    } else {
      // In the case of propagating through a backward merge, if the
      // merge output is unsafe, the outer input becomes unsafe,
      // unless the inner input is mapped with the output.
      if (unsafe_groups_.contains(out_group)) {
        if (out_group == inner_group) {
          unsafe_groups_.emplace(inner_group);
        } else {
          unsafe_groups_.emplace(outer_group);
        }
      }
    }
  }

  // Just forwards the unsafe property
  void handle(Resize* resize) {
    const auto expr_g = graph_.toGroup(resize);
    const auto inputs =
        getInputsOfExprGroup(graph_, expr_g, currentDirection());
    const auto outputs =
        getOutputsOfExprGroup(graph_, expr_g, currentDirection());

    NVF_ERROR_EQ(inputs.size(), 1);
    NVF_ERROR_EQ(outputs.size(), 1);

    if (unsafe_groups_.contains(inputs.at(0))) {
      unsafe_groups_.emplace(outputs.at(0));
    }
  }

  // Should be uncommon. Just predicate if unsafe
  void handle(Swizzle* swizzle) {
    const auto expr_g = graph_.toGroup(swizzle);
    const auto inputs =
        getInputsOfExprGroup(graph_, expr_g, currentDirection());
    const auto outputs =
        getOutputsOfExprGroup(graph_, expr_g, currentDirection());

    for (const auto& inp : inputs) {
      if (unsafe_groups_.contains(inp)) {
        groups_to_predicate_.emplace_back(inp);
      }
    }
  }

  // Should be uncommon. Just predicate if unsafe
  void handle(Swizzle2D* swizzle) {
    const auto expr_g = graph_.toGroup(swizzle);
    const auto inputs =
        getInputsOfExprGroup(graph_, expr_g, currentDirection());
    const auto outputs =
        getOutputsOfExprGroup(graph_, expr_g, currentDirection());

    for (const auto& inp : inputs) {
      if (unsafe_groups_.contains(inp)) {
        groups_to_predicate_.emplace_back(inp);
      }
    }
  }

  const std::vector<ValGroup>& groupsToPredicate() const {
    return groups_to_predicate_;
  }

 private:
  Direction currentDirection() const {
    NVF_ERROR(current_direction_ != Direction::Undefined);
    return current_direction_;
  }

 private:
  const ValGraph& graph_;

  // IDs whose indices are not guaranteed to be within the valid range
  std::unordered_set<ValGroup> unsafe_groups_;
  // IDs that need to be predicated
  std::vector<ValGroup> groups_to_predicate_;

  Direction current_direction_ = Direction::Undefined;
};

} // namespace

std::vector<ValGroup> NonDivisiblePredicateInfo::
    getNonDivisibleSplitsToPredicate(
        const ValGraph& graph,
        const ValGraphBFS::ExprPath& indexing_path) {
  return IndexingPathAnalysis(graph, indexing_path).groupsToPredicate();
}

bool NonDivisiblePredicateInfo::hasPredicate(TensorView* tv) const {
  auto it = idsToPredicate().find(tv);
  return it != idsToPredicate().end() && !(it->second.empty());
}

} // namespace nvfuser
