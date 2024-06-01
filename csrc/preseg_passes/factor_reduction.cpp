// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/factor_reduction.h>

#include <id_model/id_model.h>
#include <scheduler/utils.h>
#include <unordered_set>
#include <vector>

#include <ir/utils.h>

namespace nvfuser::preseg_passes {

void FactorReductionPass::runPass(Fusion* fusion) {
  // Persistent schedule expects all reductions to have same axes.
  // Factor common reduction axes into separate reduction operations
  // to create better fusions.

  // Common reduction iterDomains for reference TensorView
  std::unordered_set<IterDomain*> id_subset;
  // All TensorViews with common reduction axes
  std::vector<TensorView*> tv_subset;

  std::vector<TensorView*> reduction_tvs =
      scheduler_utils::getReductionTvs(fusion);
  if (reduction_tvs.size() <= 1) {
    return;
  }

  FusionGuard fg(fusion);
  IdModel id_model(fusion, /*build_graphs=*/false, /*allow_self_mapping=*/true);
  id_model.buildExactGraph();
  ValGraph exact_graph = id_model.idGraph(IdMappingMode::EXACT);
  const DisjointSets<Val*>& val_sets = exact_graph.disjointValSets();

  for (TensorView* tv : reduction_tvs) {
    const std::vector<IterDomain*>& tv_root_domain = tv->getRootDomain();

    // Initialize reference subset if empty
    if (tv_subset.empty()) {
      std::copy_if(
          tv_root_domain.begin(),
          tv_root_domain.end(),
          std::inserter(id_subset, id_subset.begin()),
          [](IterDomain* id) { return id->isReduction(); });
      tv_subset.push_back(tv);
      continue;
    }

    // Collect reduction ids for this TensorView
    std::vector<IterDomain*> reduction_ids;
    std::copy_if(
        tv_root_domain.begin(),
        tv_root_domain.end(),
        std::back_inserter(reduction_ids),
        [](IterDomain* id) { return id->isReduction(); });

    // Get intersection from reference subset and this TensorView
    //  * Keep reference id if any of this TensorView's reduction ids are
    //    mapped via Exact IdGraph.
    std::unordered_set<IterDomain*> intersection;
    std::copy_if(
        id_subset.begin(),
        id_subset.end(),
        std::inserter(intersection, intersection.begin()),
        [&](IterDomain* subset_id) {
          return std::any_of(
              reduction_ids.begin(), reduction_ids.end(), [&](IterDomain* id) {
                return val_sets.permissiveAreMapped(subset_id, id);
              });
        });

    // Update subsets if this TensorView has any common reduction axes
    if (!intersection.empty()) {
      id_subset.swap(intersection);
      tv_subset.push_back(tv);
    }
  }

  // All reduction iterDomains in last TensorView do not match any in id_subset
  // rfactor common subset of reduction axes for TensorView's in subset
  if (tv_subset.size() == 1) {
    return;
  }

  // Map common reduction iterDomains to integer axes
  for (TensorView* tv : tv_subset) {
    const std::vector<IterDomain*>& tv_root_domain = tv->getRootDomain();

    // Get reduction indices to factor from current TensorView
    //  * Scan through reference ids
    //  * Find corresponding match for this TensorView
    //  * Return position for reduction id in this TensorView
    std::vector<int64_t> rfactor_indices;
    rfactor_indices.reserve(id_subset.size());
    std::transform(
        id_subset.begin(),
        id_subset.end(),
        std::back_inserter(rfactor_indices),
        [&](IterDomain* subset_id) {
          auto iter = std::find_if(
              tv_root_domain.begin(),
              tv_root_domain.end(),
              [&](IterDomain* id) {
                return val_sets.permissiveAreMapped(subset_id, id);
              });
          return std::distance(tv_root_domain.begin(), iter);
        });

    size_t num_reduction_ids = std::count_if(
        tv_root_domain.begin(), tv_root_domain.end(), [](IterDomain* id) {
          return id->isReduction();
        });

    // Skip if all ids are used for this TensorView
    if (rfactor_indices.size() < num_reduction_ids) {
      // Separate common reduction axes
      // TODO replace pseudo-like rfactor operation
      // tv->rFactor(rfactor_indices);
    }
  }
}

} // namespace nvfuser::preseg_passes
