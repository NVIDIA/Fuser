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
#include <ir/utils.h>
#include <utils.h>
#include <val_graph.h>

#include <sstream>

namespace nvfuser {

namespace {

bool exprsMap(
    Expr* first,
    Expr* second,
    bool forward,
    const DisjointSets<IterDomain*>& id_map) {
  if (first == nullptr || second == nullptr) {
    return false;
  }

  if (typeid(*first) != typeid(*second)) {
    return false;
  }

  NVF_ERROR(
      first->isA<Merge>() || first->isA<Split>() || first->isA<Resize>(),
      "Merge, split and resize are the only expressions supported through rfactor operations in compute at map, but found:\n",
      first->toString());

  auto first_ids = ir_utils::filterByType<IterDomain>(
                       forward ? first->inputs() : first->outputs())
                       .vector();

  auto second_ids = ir_utils::filterByType<IterDomain>(
                        forward ? second->inputs() : second->outputs())
                        .vector();

  NVF_ERROR(
      first_ids.size() == second_ids.size(),
      "Expected number of ",
      (forward ? "inputs" : "outputs"),
      " to match for\n",
      first->toString(),
      second->toString());

  {
    std::vector<std::pair<IterDomain*, IterDomain*>> zipped_ids;

    std::transform(
        first_ids.begin(),
        first_ids.end(),
        second_ids.begin(),
        std::back_inserter(zipped_ids),
        [](IterDomain* first, IterDomain* second) {
          return std::make_pair(first, second);
        });

    if (std::any_of(
            zipped_ids.begin(),
            zipped_ids.end(),
            [&](std::pair<IterDomain*, IterDomain*> id_pair) {
              return !id_map.strictAreMapped(id_pair.first, id_pair.second);
            })) {
      return false;
    }
  }

  if (first->isA<Merge>() && !forward) {
    if (!ValGraph::mapMergeBackward<IterDomain>(
            first->as<Merge>(), second->as<Merge>(), id_map)) {
      return false;
    }
  }

  if (first->isA<Split>()) {
    auto first_split = first->as<Split>();
    auto second_split = second->as<Split>();
    if (!first_split->factor()->sameAs(second_split->factor()) ||
        first_split->innerSplit() != second_split->innerSplit() ||
        !first_split->startOffset()->sameAs(second_split->startOffset()) ||
        !first_split->stopOffset()->sameAs(second_split->stopOffset())) {
      return false;
    }
  }

  if (first->isA<Resize>()) {
    auto first_resize = first->as<Resize>();
    auto second_resize = second->as<Resize>();
    if (!first_resize->leftExpand()->sameAs(second_resize->leftExpand()) ||
        !first_resize->rightExpand()->sameAs(second_resize->rightExpand())) {
      return false;
    }
  }

  return true;
}

} // namespace

IdModelValidator::IdModelValidator(Fusion* fusion) : ca_map_(fusion) {
  for (auto tv : ir_utils::allTvs(fusion)) {
    for (auto id : ir_utils::allIDsOf(tv)) {
      if (id->definition() && id->definition()->isA<Swizzle2D>()) {
        has_swizzle_ = true;
        break;
      }
    }
  }
}

void IdModelValidator::fullyPropagateMappings(
    DisjointSets<IterDomain*>& id_sets) {
  // This algorithm seems terriblly inefficient but shouldn't matter as
  // this is just for transitory validations
  while (true) {
    // Grab all pairs of domains to map
    std::vector<std::pair<IterDomain*, IterDomain*>> ids_to_map;
    for (const auto& set : id_sets.disjointSets()) {
      // Propagate both forward and backward
      for (bool is_forward : {true, false}) {
        // Grab all use/definition exprs of this ID set
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
            if (id->definition()) {
              all_exprs.push_back(id->definition());
            }
          }
        }

        // Look at all combinatorial pairs of the uses of
        // definitions. If they are mapped, i.e., their input or
        // output domains are mapped and the expr
        // properties are equivalent, map the outputs or inputs as
        // well
#if 0
        if (!is_forward) {
          std::cerr << "Expr outputs: ";
          for (auto expr: all_exprs) {
            std::cerr << expr->output(0)->name() << " ";
          }
          std::cerr << std::endl;
        }
#endif
        auto count = all_exprs.size();
        for (size_t i = 0; i < count; ++i) {
          auto expr_i = all_exprs.at(i);
          for (size_t j = i + 1; j < count; ++j) {
            auto expr_j = all_exprs.at(j);
            bool debug =
                ((expr_i->output(0)->name() == 92 &&
                  expr_j->output(0)->name() == 39) ||
                 (expr_i->output(0)->name() == 39 &&
                  expr_j->output(0)->name() == 92));
            debug = debug && !is_forward && expr_i->isA<Merge>();
            debug = false;
            if (debug) {
              std::cerr << "Considering " << expr_i->toString()
                        << expr_j->toString() << "forward: " << is_forward
                        << std::endl;
            }
            if (!exprsMap(expr_i, expr_j, is_forward, id_sets)) {
              if (debug) {
                std::cerr << "not mapped\n";
              }
              continue;
            }
            if (debug) {
              std::cerr << "Mapped\n";
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

void IdModelValidator::checkExactGraphEquivalence(const ValGraph& exact_graph) {
  if (has_swizzle_) {
    // Ignoring a fusion with swizzle
    return;
  }

  // Empty graph
  if (exact_graph.disjointValSets().disjointSets().empty()) {
    return;
  }

  DisjointSets<IterDomain*> ca_map_sets = ca_map_.id_graph_.exact_nodes_;

  // IdModel propagates mappings forward and backward more
  // consistently, which is not the case with ComputeAt. To compare
  // the two mappings, augment the ComputeAt mappings with the same
  // propagation. This might potentially hide some subtle differences
  // between the two mappings, but I think this is still a reasonable
  // way to validate IdModel
  fullyPropagateMappings(ca_map_sets);

  compareDisjointSets(ca_map_sets, exact_graph.disjointValSets());
}

void IdModelValidator::checkAlmostExactGraphEquivalence(
    const ValGraph& almost_exact_graph) {
  if (has_swizzle_) {
    // Ignoring a fusion with swizzle
    return;
  }

  // Empty graph
  if (almost_exact_graph.disjointValSets().disjointSets().empty()) {
    return;
  }

  DisjointSets<IterDomain*> ca_map_sets = ca_map_.id_graph_.almost_exact_nodes_;

  fullyPropagateMappings(ca_map_sets);

  compareDisjointSets(ca_map_sets, almost_exact_graph.disjointValSets());
}

void IdModelValidator::checkPermissiveGraphEquivalence(
    const ValGraph& permissive_graph) {
  if (has_swizzle_) {
    // Ignoring a fusion with swizzle
    return;
  }

  // Empty graph
  if (permissive_graph.disjointValSets().disjointSets().empty()) {
    return;
  }

  DisjointSets<IterDomain*> ca_map_sets = ca_map_.id_graph_.permissive_nodes_;

  fullyPropagateMappings(ca_map_sets);

  compareDisjointSets(ca_map_sets, permissive_graph.disjointValSets());
}

} // namespace nvfuser
