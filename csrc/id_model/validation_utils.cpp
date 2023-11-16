// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <compute_at_map.h>
#include <id_model/to_string.h>
#include <id_model/validation_utils.h>

#include <sstream>

namespace nvfuser {

void IdModelValidator::checkExactGraphEquivalence(const ValGraph& exact_graph) {
  // Empty graph
  if (exact_graph.disjointValSets().disjointSets().empty()) {
    return;
  }

  auto all_exprs = exact_graph.disjointExprSets().getAllElements();
  if (std::find_if(all_exprs.begin(), all_exprs.end(), [](Expr* expr) {
        return expr->isA<Swizzle2D>();
      }) != all_exprs.end()) {
    // Ignoring a fusion with swizzle
    return;
  }

  Fusion* fusion = exact_graph.disjointValSets()
                       .disjointSets()
                       .at(0)
                       ->vector()
                       .at(0)
                       ->fusion();
  ComputeAtMap ca_map(fusion);

  DisjointSets<IterDomain*>& ca_map_exact_sets = ca_map.id_graph_.exact_nodes_;

  // Propgate mappings through expressions in ComputeAtMap. Since we
  // want to traverse and update ca_map_exact_sets, once  updated, the
  // traversal of the ID groups cannot continue and needs to be
  // restarted. The algorithm seems terriblly inefficient, but
  // shuldn't matter as this is just for transitory validations
  bool updated = true;
  while (updated) {
    updated = false;
    for (const auto& set : ca_map_exact_sets.disjointSets()) {
      auto uses = ca_map.uniqueExactUses(set->vector().front());
      auto use_count = uses.size();
      // Note that it should be fine to continue updating the map with
      // the loop below as it should only modify output domain groups
      for (size_t i = 0; i < use_count; ++i) {
        auto use_i = uses.at(i);
        for (size_t j = i + 1; j < use_count; ++j) {
          auto use_j = uses.at(j);
          if (!IterDomainGraph::exprsMap(
                  use_i, use_j, true, ca_map_exact_sets)) {
            continue;
          }
          auto num_outputs = use_i->outputs().size();
          NVF_ERROR(use_j->outputs().size() == num_outputs);
          for (size_t output_i = 0; output_i < num_outputs; ++output_i) {
            auto out_i = use_i->output(output_i)->as<IterDomain>();
            auto out_j = use_j->output(output_i)->as<IterDomain>();
            if (!ca_map_exact_sets.strictAreMapped(out_i, out_j)) {
              ca_map_exact_sets.mapEntries(out_i, out_j);
              updated = true;
            }
          }
        }
      }
      // If updated, the previous sets returned by
      // ca_map_exact_sets.disjointSets() may contain stale sets
      if (updated) {
        ca_map.build(fusion);
        break;
      }
    }
  }

  const DisjointSets<Val*>& id_model_exact_sets = exact_graph.disjointValSets();

  if (id_model_exact_sets.size() != ca_map_exact_sets.size()) {
    std::stringstream ss;
    ss << "Mismatched number of groups: " << id_model_exact_sets.size() << ", "
       << ca_map_exact_sets.size() << "\n";

    ss << "IdModel exact sets:\n";
    for (const auto& id_set : id_model_exact_sets.disjointSets()) {
      ss << "\t" << nvfuser::toString(id_set->vector()) << "\n";
    }

    ss << "ComputeAtMap exact sets:\n";
    for (const auto& id_set : ca_map_exact_sets.disjointSets()) {
      ss << "\t" << nvfuser::toString(id_set->vector()) << "\n";
    }

    NVF_ERROR(false, ss.str());
  }

  for (const auto& id_model_id_set : id_model_exact_sets.disjointSets()) {
    NVF_ERROR(!id_model_id_set->empty());
    NVF_ERROR(
        ca_map_exact_sets.mappingExists(
            id_model_id_set->front()->as<IterDomain>()),
        "Not found in ComputeAtMap: ",
        id_model_id_set->front()->toString());

    const auto& ca_map_id_set = ca_map_exact_sets.getDisjointSetOf(
        id_model_id_set->front()->as<IterDomain>());

    std::unordered_set<Val*> ca_map_id_set_cast;
    std::copy(
        ca_map_id_set.begin(),
        ca_map_id_set.end(),
        std::inserter(ca_map_id_set_cast, ca_map_id_set_cast.end()));

    NVF_ERROR(
        id_model_id_set->set() == ca_map_id_set_cast,
        "Mismatched ID set: ",
        nvfuser::toString(id_model_id_set->vector()),
        ", ",
        nvfuser::toString(ca_map_id_set.vector()));
  }
}

} // namespace nvfuser
