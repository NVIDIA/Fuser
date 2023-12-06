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

void IdModelValidator::fullyPropagateMappings(
    Fusion* fusion,
    ComputeAtMap& ca_map,
    DisjointSets<IterDomain*>& id_sets) {
  // This algorithm seems terriblly inefficient but shuldn't matter as
  // this is just for transitory validations
  while (true) {
    // Grab all pairs of domains to map
    std::vector<std::pair<IterDomain*, IterDomain*>> ids_to_map;
    for (const auto& set : id_sets.disjointSets()) {
      // Propagate both forward and backward
      for (bool is_forward : {true, false}) {
        // Grab all use exprs of this ID set
        std::vector<Expr*> all_exprs;
        for (auto id : *set) {
          // In the case of forward propagation, the uses() exprs may
          // not be actually used for IterDomain. Make sure to pick
          // only those whose outputs are in the map
          if (is_forward) {
            for (auto use : id->uses()) {
              if (std::all_of(
                      use->outputs().begin(),
                      use->outputs().end(),
                      [&](Val* output) {
                        return output->isA<IterDomain>() &&
                            id_sets.mappingExists(output->as<IterDomain>());
                      })) {
                all_exprs.push_back(use);
              }
            }
          } else {
            all_exprs.push_back(id->definition());
          }
        }

        // Look at all combinatorial pairs of the uses of
        // definitions. If they are mapped, i.e., their input or
        // output domains are mapped and the expr
        // properties are equivalent, map the outputs or inputs as well
        auto count = all_exprs.size();
        for (size_t i = 0; i < count; ++i) {
          auto expr_i = all_exprs.at(i);
          for (size_t j = i + 1; j < count; ++j) {
            auto expr_j = all_exprs.at(j);
            if (!IterDomainGraph::exprsMap(
                    expr_i, expr_j, is_forward, id_sets)) {
              continue;
            }
            const auto& prop_target_i =
                is_forward ? expr_i->outputs() : expr_i->inputs();
            const auto& prop_target_j =
                is_forward ? expr_j->outputs() : expr_j->inputs();
            auto num_target = prop_target_i.size();
            NVF_ERROR(num_target == prop_target_j.size());
            for (size_t target_i = 0; target_i < num_target; ++target_i) {
              auto id_i = prop_target_i.at(target_i)->as<IterDomain>();
              auto id_j = prop_target_j.at(target_i)->as<IterDomain>();
              if (!id_sets.strictAreMapped(id_i, id_j)) {
                // Don't actually map them yet as it would invalidate
                // the loop over id_sets
                ids_to_map.emplace_back(id_i, id_j);
              }
            }
          }
        }
      }
    }

    // No additional domains to map. Nothing to do further
    if (ids_to_map.empty()) {
      return;
    }

    for (const auto& [id1, id2] : ids_to_map) {
      id_sets.mapEntries(id1, id2);
    }
  }
}

namespace {

void compareDisjointSets(
    const DisjointSets<IterDomain*>& ca_map_sets,
    const DisjointSets<Val*>& id_model_sets) {
  if (id_model_sets.size() != ca_map_sets.size()) {
    std::stringstream ss;
    ss << "Mismatched number of groups: " << id_model_sets.size() << ", "
       << ca_map_sets.size() << "\n";

    ss << "IdModel sets:\n";
    for (const auto& id_set : id_model_sets.disjointSets()) {
      ss << "\t" << nvfuser::toString(id_set->vector()) << "\n";
    }

    ss << "ComputeAtMap sets:\n";
    for (const auto& id_set : ca_map_sets.disjointSets()) {
      ss << "\t" << nvfuser::toString(id_set->vector()) << "\n";
    }

    NVF_ERROR(false, ss.str());
  }

  for (const auto& id_model_id_set : id_model_sets.disjointSets()) {
    NVF_ERROR(!id_model_id_set->empty());
    NVF_ERROR(
        ca_map_sets.mappingExists(id_model_id_set->front()->as<IterDomain>()),
        "Not found in ComputeAtMap: ",
        id_model_id_set->front()->toString());

    const auto& ca_map_id_set = ca_map_sets.getDisjointSetOf(
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

} // namespace

void IdModelValidator::checkExactGraphEquivalence(const IdModel& id_model) {
  const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);

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

  // IdModel propagates mappings forward and backward more
  // consistently, which is not the case with ComputeAt. To compare
  // the two mappings, augment the ComputeAt mappings with the same
  // propagation. This might potentially hide some subtle differences
  // between the two mappings, but I think this is still a reasonable
  // way to validate IdModel
  fullyPropagateMappings(fusion, ca_map, ca_map_exact_sets);

  const DisjointSets<Val*>& id_model_exact_sets = exact_graph.disjointValSets();

  // Similarly, update the almost exact CA map for the comparison
  compareDisjointSets(ca_map_exact_sets, id_model_exact_sets);

  fullyPropagateMappings(fusion, ca_map, ca_map.id_graph_.almost_exact_nodes_);

  compareDisjointSets(
      ca_map.id_graph_.almost_exact_nodes_,
      id_model.idGraph(IdMappingMode::ALMOSTEXACT).disjointValSets());
}

} // namespace nvfuser
