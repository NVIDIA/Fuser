// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/id_model.h>
#include <id_model/to_string.h>
#include <id_model/validation_utils.h>

#include <device_lower/analysis/trivial_broadcast.h>
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <disjoint_set.h>
#include <ir/utils.h>
#include <root_domain_map.h>
#include <transform_iter.h>

#include <memory>
#include <tuple>
#include <typeinfo>
#include <utility>

namespace nvfuser {

namespace {

// Map through loop swizzles, as input/output IterDomains are exact, only the
// order they're traversed differs.
void mapThroughLoopSwizzles(ValGraph& graph) {
  std::vector<Swizzle2D*> all_swizzles;

  for (const auto& expr_set :
       std::as_const(graph).disjointExprSets().disjointSets()) {
    auto swizzles_in_expr_set = ir_utils::filterByType<Swizzle2D>(
        expr_set->vector().begin(), expr_set->vector().end());
    all_swizzles.insert(
        all_swizzles.end(),
        swizzles_in_expr_set.begin(),
        swizzles_in_expr_set.end());
  }

  for (auto swizzle : all_swizzles) {
    if (swizzle->swizzleMode() == SwizzleMode::Loop) {
      graph.mapVals(swizzle->inX(), swizzle->outX());
      graph.mapVals(swizzle->inY(), swizzle->outY());
    }
  }
}

} // namespace

void IdModel::assertNoSelfMapping() {
  NVF_ERROR(
      !hasSelfMapping(),
      "Unsupported domain mapping detected in ",
      std::get<0>(*self_mapping_info_)->toString(),
      ". ",
      std::get<3>(*self_mapping_info_),
      " domains, ",
      std::get<1>(*self_mapping_info_)->toString(),
      " and ",
      std::get<2>(*self_mapping_info_)->toString(),
      ", are mapped with each other.");
}

IdModel::IdModel(
    const std::vector<Expr*>& exprs,
    const std::vector<TensorView*>& additional_tvs,
    bool allow_self_mapping) {
  build(exprs, additional_tvs);

  if (!allow_self_mapping) {
    assertNoSelfMapping();
  }
}

IdModel::IdModel(Fusion* fusion, bool allow_self_mapping, bool validate) {
  std::vector<TensorView*> inputs_and_outputs;
  {
    auto inp_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    inputs_and_outputs.insert(
        inputs_and_outputs.begin(), inp_tvs.begin(), inp_tvs.end());
  }
  {
    auto out_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
    inputs_and_outputs.insert(
        inputs_and_outputs.end(), out_tvs.begin(), out_tvs.end());
  }

  build(fusion->exprs(), inputs_and_outputs, validate);

  if (!allow_self_mapping) {
    assertNoSelfMapping();
  }
}

const ValGraph& IdModel::idGraph(IdMappingMode mode) const {
  auto graph_it = id_graphs_.find(mode);
  NVF_ERROR(
      graph_it != id_graphs_.end(),
      "Failed to find an IdGraph with the ",
      mode,
      " mode");
  return graph_it->second;
}

ValGraph& IdModel::idGraph(IdMappingMode mode) {
  auto graph_it = id_graphs_.find(mode);
  NVF_ERROR(graph_it != id_graphs_.end());
  return graph_it->second;
}

namespace {

// Returns the first pair of id's in ids detected to match each other on the
// exact ID graph. TODO: what this is really looking for is if
// there's any overlapping between the iter domains in the provided set.
//
// i.e. if we have:
// tv0 = arange(6).reshape({3, 2})
// tv1 = tv0[3, 2].t()
// tv2 = tv0[3, 2].reshape({2, 3})
// tv3 = tv1 + tv2
//
// Then we can see this overlap in the tv3 expression as:
//
// tv0 = { {0, 1, 2},
//         {3, 4, 5} }
//
// tv1 = { {0, 3},
//         {1, 4},
//         {2, 5} }
//
// tv2 = { {0, 1},
//         {2, 3},
//         {4, 5} }
//
// The elements in tv1 {3, 1, 4, 2}, map respectively to the elements in tv2
// {1, 2, 3, 4}. The reason this is so important is it means that generating
// tv3 is no longer a trivially parallelizable problem (if we include the dag
// all the way to tv0). So tv0's axes cannot be inlined across both the tv0
// and tv1 path. This breaks some assumptions we have today in schedulers that
// will assume tv2 can be trivially inlined/parallelized. Instead we'd need to
// take into consideration the effective communication going on here, so that
// we pull multiple values of tv0 to compute tv3.
//
// Note, however, that the above example is not detectable at this
// moment as the self mapping is partial through reshape. The analysis
// below would need to be extended to consider producer and consumers
// of domains as well rather than just root, rfactor and leaf domains.
std::optional<std::pair<IterDomain*, IterDomain*>> detectMappablePair(
    const std::vector<IterDomain*>& ids,
    const IdModel& id_graph,
    IdMappingMode mode) {
  for (auto id1 : ids) {
    for (auto id2 : ids) {
      if (id1 == id2) {
        continue;
      }
      if (id_graph.idGraph(mode).disjointValSets().permissiveAreMapped(
              id1, id2)) {
        return std::make_pair(id1, id2);
      }
    }
  }

  return std::nullopt;
}

// It is assumed that for any tensor represented by a list of domains,
// those domains should never be mapped with each other. It may be
// possible to lift this assumption, but it's unclear if it could
// matter in practice.
std::optional<std::tuple<TensorView*, IterDomain*, IterDomain*, std::string>>
findFirstSelfMapping(
    const std::vector<TensorView*>& all_tvs,
    const IdModel& id_graph) {
  for (auto tv : all_tvs) {
    // For each tensor, make sure root, rfactor and leaf domains
    // should not include domains that are mapped with another domain
    // in the same set of domains. This may be overly conservative,
    // and it maybe enough to check the root domains.

    // Root domains
    auto self_mappped_root_pair =
        detectMappablePair(tv->getRootDomain(), id_graph, IdMappingMode::EXACT);
    if (self_mappped_root_pair.has_value()) {
      return std::make_tuple(
          tv,
          self_mappped_root_pair->first,
          self_mappped_root_pair->second,
          "Root");
    }

    // Rfactor domains
    if (tv->hasRFactor()) {
      auto self_mappped_rf_pair = detectMappablePair(
          tv->getRFactorDomain(), id_graph, IdMappingMode::EXACT);
      if (self_mappped_rf_pair.has_value()) {
        return std::make_tuple(
            tv,
            self_mappped_rf_pair->first,
            self_mappped_rf_pair->second,
            "RFactor");
      }
    }

    // Leaf domains
    // TODO: Exact map isn't quite right here, it should be based on the index
    // map. However, it should also be impossible for index map to generate a
    // case like this.
    auto self_mappped_leaf_pair = detectMappablePair(
        tv->domain()->leaf(), id_graph, IdMappingMode::EXACT);
    if (self_mappped_leaf_pair.has_value()) {
      return std::make_tuple(
          tv,
          self_mappped_leaf_pair->first,
          self_mappped_leaf_pair->second,
          "Leaf");
    }
  }
  return std::nullopt;
}

} // namespace

void IdModel::buildIterDomainDefinitionsAndUses(
    const std::vector<TensorView*>& all_tvs) {
  for (const auto tv : all_tvs) {
    VectorOfUniqueEntries<IterDomain*> root_domain_ids{
        tv->getRootDomain().begin(), tv->getRootDomain().end()};

    std::vector<IterDomain*> all_ids = ir_utils::allIDsOf(tv);

    // Check if this domain is a consumer of a view-like operation
    const bool view_like_domain = tv->domain()->hasViewLikeRFactor();

    for (auto id : all_ids) {
      // Check if this id is a view like rfactor id
      if (view_like_domain && id->isRFactorProduct()) {
        // If the tensor domain is a view like domain, and the iteration
        // domain is marked as an rfactor product and is in the rfactor
        // domain, it's a view like rfactor iteration domain
        const auto& rfactor_domain = tv->domain()->maybeRFactor();
        if (std::find(rfactor_domain.begin(), rfactor_domain.end(), id) !=
            rfactor_domain.end()) {
          view_rfactor_ids_.emplace(id);
        }
      }

      if (id_definitions_.find(id) == id_definitions_.end()) {
        id_definitions_.emplace(id, VectorOfUniqueEntries<Expr*>{});
      }

      if (id_uses_.find(id) == id_uses_.end()) {
        id_uses_.emplace(id, VectorOfUniqueEntries<Expr*>{});
      }

      Expr* def = id->definition();

      if (def == nullptr || root_domain_ids.has(id)) {
        continue;
      }

      id_definitions_[id].pushBack(def);

      auto inp_ids = ir_utils::filterByType<IterDomain>(def->inputs());
      for (auto inp_id : inp_ids) {
        id_uses_[inp_id].pushBack(def);
      }
    }
  }
}

std::string IdModel::toString() const {
  std::stringstream ss;
  ss << "IterDomainGraphs { \n";
  // Only print initialized graphs
  for (auto mode : kIdMappingModes) {
    auto graph_it = id_graphs_.find(mode);
    if (graph_it == id_graphs_.end()) {
      continue;
    }

    // graph may be empty, but then just print it as an empty graph,
    // which might be useful for debugging
    ss << "  IdGraph " << mode << "{ \n";
    ss << "  Disjoint Ids:\n"
       << idGroupsString(idGraph(mode), 2)
       << "\n  Disjoint Expression groups:\n"
       << exprGroupsString(idGraph(mode), 2) << std::endl;
    ss << "   } IdGraph\n" << std::endl;
  }
  ss << " } IterDomainGraphs\n" << std::endl;
  return ss.str();
}

ValGraph IdModel::initializeIdGraph(bool propagate_through_exprs) {
  ValGraph id_graph(propagate_through_exprs);

  for (const auto& [id, defs] : id_definitions_) {
    auto uses_it = id_uses_.find(id);
    NVF_ERROR(
        uses_it != id_uses_.end(),
        "Failed to initialize id: ",
        id->toString(),
        " as it's missing a definition entry.");
    id_graph.initializeVal(id, defs, uses_it->second);
  }

  return id_graph;
}

void IdModel::buildExactGraph(const std::vector<Expr*>& exprs) {
  for (auto expr : exprs) {
    TensorView* c_tv = ir_utils::getTvOutput(expr);

    auto all_tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());

    // Map siblings, as all other tv output domains must match the first tv
    // outputs domain.
    std::deque<TensorView*> other_tv_outputs(
        all_tv_outputs.begin(), all_tv_outputs.end());
    other_tv_outputs.pop_front();

    for (auto other_tv_output : other_tv_outputs) {
      // Sibling tv's must be exactly mapped with eachother so simply zip
      // their leaf iter domains.

      NVF_ERROR(
          other_tv_output->getRootDomain().size() ==
              c_tv->getRootDomain().size(),
          "Multiple outputs with mismatched TV domains is not supported.");

      for (auto domain_i : c10::irange(c_tv->getRootDomain().size())) {
        auto c_id = c_tv->getRootDomain()[domain_i];
        auto o_id = other_tv_output->getRootDomain()[domain_i];
        idGraph(IdMappingMode::EXACT).mapVals(o_id, c_id);
      }
    }

    // Map producer-consumer relationships based on the root domain map
    auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    for (auto p_tv : tv_inputs) {
      // For exact mapings do not map any broadcast dimensions to
      // non-broadcast dimensions. Prevent any broadcasted axes being mapped
      // to non-broadcasted axes.
      auto exact_c2p_root_map = PairwiseRootDomainMap(p_tv, c_tv)
                                    .mapBroadcast(false)
                                    .mapConsumerToProducer();

      for (auto c_id : getSortedKeys(exact_c2p_root_map, Statement::lessThan)) {
        auto p_id = exact_c2p_root_map.at(c_id);
        idGraph(IdMappingMode::EXACT).mapVals(c_id, p_id);
      }
    }

    // TODO: Revisit if we really should map domains in the exact map
    mapThroughLoopSwizzles(idGraph(IdMappingMode::EXACT));
  }
}

namespace {

// Checks if the expression is a trivial operation where an input is simply an
// output of the transformation. Returns the mapped iter domains if found.
std::vector<std::vector<Val*>> getTriviallyMappedIds(Expr* expr) {
  std::vector<std::vector<Val*>> mapped_ids;
  if (auto merge = dynamic_cast<Merge*>(expr)) {
    if (merge->inner()->extent()->isOneInt()) {
      mapped_ids.push_back({merge->outer(), merge->out()});
    }
    if (merge->outer()->extent()->isOneInt()) {
      mapped_ids.push_back({merge->inner(), merge->out()});
    }
  } else if (auto split = dynamic_cast<Split*>(expr)) {
    if (split->factor()->isOneInt() && split->startOffset()->isZeroInt() &&
        split->stopOffset()->isZeroInt()) {
      if (split->innerSplit()) {
        mapped_ids.push_back({split->in(), split->outer()});
      } else {
        mapped_ids.push_back({split->in(), split->inner()});
      }
    }
  } else if (auto swizzle = dynamic_cast<Swizzle2D*>(expr)) {
    if (swizzle->swizzleType() == Swizzle2DType::NoSwizzle ||
        swizzle->swizzleMode() == SwizzleMode::NoSwizzle) {
      mapped_ids.push_back({swizzle->inX(), swizzle->outX()});
      mapped_ids.push_back({swizzle->inY(), swizzle->outY()});
    }
  }
  return mapped_ids;
}

} // namespace

void IdModel::buildAlmostExactMap() {
  // Build almost exact map by forwarding through broadcast axes
  idGraph(IdMappingMode::ALMOSTEXACT) = idGraph(IdMappingMode::EXACT);

  auto& almost_exact_graph = idGraph(IdMappingMode::ALMOSTEXACT);

  // Maps iter domain pairs returned by calling that return mappings from
  // isTrivialExpr on every expression in the graph.

  // Don't traverse the graph and at the same time add more mappings
  // as the traversal would be invalidated
  std::vector<std::pair<Val*, Val*>> ids_to_map;

  for (const auto& expr_group :
       almost_exact_graph.disjointExprSets().disjointSets()) {
    for (auto expr : *expr_group) {
      // If not trivial continue
      auto mapped_ids = getTriviallyMappedIds(expr);
      if (mapped_ids.empty()) {
        continue;
      }

      // Map through trivial expressions
      for (auto mapped_id_group : mapped_ids) {
        for (auto id : mapped_id_group) {
          // almost_exact_graph.mapVals(mapped_id_group.front(), id);
          ids_to_map.emplace_back(mapped_id_group.front(), id);
        }
      }
    }
  }

  for (const auto& [id1, id2] : ids_to_map) {
    almost_exact_graph.mapVals(id1, id2);
  }
}

void IdModel::build(
    const std::vector<Expr*>& exprs,
    const std::vector<TensorView*>& additional_tvs,
    bool validate) {
  // Initialize the required sets as if a permissive relationship is never
  // found, then querying an empty permissive map will fail later.
  // Initialize disjoint sets
  for (auto mode : kIdMappingModes) {
    id_graphs_[mode] = ValGraph();
  }

  std::vector<Expr*> tv_exprs;

  std::copy_if(
      exprs.begin(), exprs.end(), std::back_inserter(tv_exprs), [](Expr* expr) {
        NVF_ERROR(expr != nullptr);
        return ir_utils::isTvOp(expr);
      });

  auto all_tvs = ir_utils::allTvsOfExprs(tv_exprs);

  for (auto additional_tv : additional_tvs) {
    all_tvs.pushBack(additional_tv);
  }

  if (all_tvs.empty()) {
    return;
  }

  std::unique_ptr<IdModelValidator> validator;

  // A ComputeAtMap will be built inside the constructor of
  // IdModelValidator, which may fail for some fusions that are not
  // supported currently (but work with IdModel). Make sure the
  // validator is only created when it is indeed requested
  if (validate) {
    validator = std::make_unique<IdModelValidator>(all_tvs.front()->fusion());
  }

  FusionGuard fg(all_tvs.front()->fusion());
  // Add uses and definitions to all iter domains.
  buildIterDomainDefinitionsAndUses(all_tvs.vector());

  // Initialize the maps with all the IterDomains used in the provded
  // expressions.
  idGraph(IdMappingMode::EXACT) = initializeIdGraph();

  buildExactGraph(tv_exprs);
  if (validate) {
    validator->checkExactGraphEquivalence(idGraph(IdMappingMode::EXACT));
  }

  buildAlmostExactMap();
  if (validate) {
    validator->checkAlmostExactGraphEquivalence(
        idGraph(IdMappingMode::ALMOSTEXACT));
  }

  // Make sure there's no self mapping in TensorView's during lowering
  // that would invalidate lowering assumptions.
  self_mapping_info_ = findFirstSelfMapping(all_tvs.vector(), *this);
}

} // namespace nvfuser
