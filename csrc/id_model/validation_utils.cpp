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

void IdModelValidator::checkExactMapEquivalence(const IdGraph& exact_graph) {
  auto all_exprs = exact_graph.disjointExprSets().getAllElements();
  if (std::find_if(all_exprs.begin(), all_exprs.end(), [](Expr* expr) {
        return expr->isA<Swizzle2D>();
      }) != all_exprs.end()) {
    std::cerr << "Ignoring a fusion with swizzle\n";
    return;
  }

  if (exact_graph.disjointIdSets().disjointSets().empty()) {
    return;
  }

  Fusion* fusion = exact_graph.disjointIdSets()
                       .disjointSets()
                       .at(0)
                       ->vector()
                       .at(0)
                       ->fusion();
  ComputeAtMap ca_map(fusion);

  const DisjointSets<IterDomain*>& exact_sets = exact_graph.disjointIdSets();
  DisjointSets<IterDomain*>& ca_map_exact_sets = ca_map.id_graph_.exact_nodes_;

  if (getenv("VERBOSE")) {
    std::stringstream ss;
    ss << "Initial computeatmap exact sets:\n";
    for (const auto& id_set : ca_map_exact_sets.disjointSets()) {
      ss << "\t" << nvfuser::toString(id_set->vector()) << "\n";
    }
    std::cerr << ss.str();
  }

  // Since we want to traverse and update ca_map_exact_sets, once
  // updated, the traversal of the ID groups cannot continue and needs
  // to be restarted.
  bool updated = true;
  while (updated) {
    if (getenv("VERBOSE")) {
      std::cerr << "While loop\n";
    }
    updated = false;
    for (const auto& set : ca_map_exact_sets.disjointSets()) {
      if (getenv("VERBOSE")) {
        std::cerr << "set: " << nvfuser::toString(set) << std::endl;
      }
      auto uses = ca_map.uniqueExactUses(set->vector().front());
      auto use_count = uses.size();
      // Note that it should be fine to continue updating the map with
      // the loop below as it should only modify output domain groups
      for (size_t i = 0; i < use_count; ++i) {
        auto use_i = uses.at(i);
        if (getenv("VERBOSE")) {
          std::cerr << "use_i: " << use_i->toString();
        }
        for (size_t j = i + 1; j < use_count; ++j) {
          auto use_j = uses.at(j);
          if (getenv("VERBOSE")) {
            std::cerr << "use_i: " << use_i->toString()
                      << ", use_j: " << use_j->toString();
          }
          if (!IterDomainGraph::exprsMap(
                  use_i, use_j, true, ca_map_exact_sets)) {
            continue;
          }
          if (getenv("VERBOSE")) {
            std::cerr << "Mapped exprs: " << use_i->toString() << "\n"
                      << use_j->toString();
          }
          auto num_outputs = use_i->outputs().size();
          NVF_ERROR(use_j->outputs().size() == num_outputs);
          for (size_t output_i = 0; output_i < num_outputs; ++output_i) {
            auto out_i = use_i->output(output_i)->as<IterDomain>();
            auto out_j = use_j->output(output_i)->as<IterDomain>();
            if (!ca_map_exact_sets.strictAreMapped(out_i, out_j)) {
              if (getenv("VERBOSE")) {
                std::cerr << "Mapping " << out_i->toString() << " and "
                          << out_j->toString() << std::endl;
              }
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

  if (exact_sets.size() != ca_map_exact_sets.size()) {
    std::stringstream ss;
    ss << "Mismatched number of groups: " << exact_sets.size() << ", "
       << ca_map_exact_sets.size() << "\n";

    ss << "IdModel exact sets:\n";
    for (const auto& id_set : exact_sets.disjointSets()) {
      ss << "\t" << nvfuser::toString(id_set->vector()) << "\n";
    }

    ss << "ComputeAtMap exact sets:\n";
    for (const auto& id_set : ca_map_exact_sets.disjointSets()) {
      ss << "\t" << nvfuser::toString(id_set->vector()) << "\n";
    }

    NVF_ERROR(false, ss.str());
  }

  for (const auto& id_set : exact_sets.disjointSets()) {
    NVF_ERROR(!id_set->empty());
    NVF_ERROR(
        ca_map_exact_sets.mappingExists(id_set->front()),
        "Not found in ComputeAtMap: ",
        id_set->front()->toString());

    const auto& ca_map_id_set =
        ca_map_exact_sets.getDisjointSetOf(id_set->front());

    NVF_ERROR(
        id_set->set() == ca_map_id_set.set(),
        "Mismatched ID set: ",
        nvfuser::toString(id_set->vector()),
        ", ",
        nvfuser::toString(ca_map_id_set.vector()));
  }
}

} // namespace nvfuser
