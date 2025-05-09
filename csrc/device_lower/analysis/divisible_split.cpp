// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/analysis/divisible_split.h>
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <ir/utils.h>

#include <unordered_set>

namespace nvfuser {

std::unordered_set<Split*> getAllDivisibleSplits(Fusion* fusion) {
  ComputeAtMap ca_map(fusion);
  return getAllDivisibleSplits(fusion, &ca_map);
}

std::unordered_set<Split*> getAllDivisibleSplits(
    Fusion* fusion,
    const ComputeAtMap* ca_map) {
  std::unordered_set<Split*> all_divisible_splits;

  auto all_tvs = fusion->allTvs();
  // Find all tensor views with a view like rfactor. Splits used in view
  // transformations must be divisible by definition.
  for (auto tv : all_tvs) {
    auto logical_dom = tv->getLogicalDomain();
    // Not view if there's no rfactor axis
    if (!tv->domain()->hasViewLikeRFactor()) {
      continue;
    }

    // Take the view transformations and add all the splits. Those splits are
    // the only divisible splits.
    auto view_exprs =
        StmtSort::getExprsTo({logical_dom.begin(), logical_dom.end()});
    auto split_exprs = ir_utils::filterByType<Split>(view_exprs);
    all_divisible_splits.insert(split_exprs.begin(), split_exprs.end());
  }

  // Vectorized dimensions are enforced to be a result of divisible splits.
  // Gather vectorized splits.
  for (auto tv : all_tvs) {
    auto vec_id_it = std::find_if(
        tv->getLoopDomain().begin(),
        tv->getLoopDomain().end(),
        [](IterDomain* id) {
          return id->getParallelType() == ParallelType::Vectorize;
        });

    if (vec_id_it == tv->getLoopDomain().end()) {
      continue;
    }

    // We could have a case technically like:
    // [8, 2] where we do:
    // split(0, 2)
    // merge(1)
    // so it ends up as [4, 4]
    // split(0, 2) must be divisible, but for now we're not going to capture
    // cases like this. Just look for direct split's producing a vectorize
    // dimension.
    auto vec_id = *vec_id_it;
    if (vec_id->definition() != nullptr && vec_id->definition()->isA<Split>()) {
      all_divisible_splits.emplace(vec_id->definition()->as<Split>());
    }
  }

  // If there's no view like splits, there's nothing to find
  if (all_divisible_splits.empty()) {
    return all_divisible_splits;
  }

  // Track the concrete id in the exact map of the outer output of the split
  // expressions. This is how we'll check if there are matching splits. This
  // also gets rid of any splits that already match (for processing).
  std::unordered_map<IterDomain*, Expr*> outer_concrete_id_to_expr;

  for (auto split : all_divisible_splits) {
    outer_concrete_id_to_expr[ca_map->getConcreteMappedID(
        split->outer(), IdMappingMode::EXACT)] = split;
  }

  std::unordered_set<Expr*> visited(
      all_divisible_splits.begin(), all_divisible_splits.end());

  // Find splits that match what we already have:
  for (auto entry : outer_concrete_id_to_expr) {
    auto concrete_id = entry.first;
    auto original_view_split = entry.second;

    const auto& exact_mapped_ids =
        ca_map->idGraph().exactNodes().getDisjointSetOf(concrete_id).vector();
    for (auto other_id : exact_mapped_ids) {
      if (other_id->definition() == nullptr) {
        continue;
      }

      if (!visited.emplace(other_id->definition()).second) {
        // Already visited
        continue;
      }

      if (IterDomainGraph::exprsMap(
              original_view_split,
              other_id->definition(),
              false,
              ca_map->idGraph().exactNodes())) {
        all_divisible_splits.emplace(other_id->definition()->as<Split>());
      }
    }
  }

  // Expand with ExactGraph if available
  if (GpuLower::hasCurrent() && GpuLower::current()->hasIdModel()) {
    const auto& exact_graph =
        GpuLower::current()->idModel().idGraph(IdMappingMode::EXACT);
    std::unordered_set<Split*> additional_splits;
    for (const auto& split : all_divisible_splits) {
      const auto& split_group = exact_graph.toGroup(split);
      for (const auto& additional_expr : *split_group) {
        NVF_ERROR(additional_expr != nullptr);
        auto additional_split = dynamic_cast<Split*>(additional_expr);
        NVF_ERROR(
            additional_split != nullptr,
            "Unexpected to have a non-split expr: ",
            additional_expr->toString());
        if (!all_divisible_splits.contains(additional_split)) {
          additional_splits.insert(additional_split);
        }
      }
    }
    all_divisible_splits.insert(
        additional_splits.begin(), additional_splits.end());
  }

  return all_divisible_splits;
}

} // namespace nvfuser
