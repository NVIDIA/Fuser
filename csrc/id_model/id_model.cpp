// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/id_model.h>
#include <id_model/to_string.h>
#include <id_model/transform_replay.h>
#include <id_model/validation_utils.h>

#include <device_lower/analysis/trivial_broadcast.h>
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <disjoint_set.h>
#include <ir/utils.h>
#include <root_domain_map.h>
#include <transform_iter.h>
#include <val_graph_visitor.h>

#include <memory>
#include <tuple>
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
  const ValGraph& exact_graph = idGraph(IdMappingMode::EXACT);
  for (TensorView* tv : tvs_) {
    std::optional<SelfMapping> self_mapping = hasSelfMapping(tv, exact_graph);
    NVF_CHECK(
        !self_mapping.has_value(),
        "Unsupported domain mapping detected in ",
        tv,
        ". ",
        self_mapping->where,
        " domains, ",
        self_mapping->id1,
        " and ",
        self_mapping->id2,
        ", are mapped with each other.");
  }
}

IdModel::IdModel(
    const std::vector<Expr*>& exprs,
    const std::vector<TensorView*>& additional_tvs,
    bool build_graphs,
    bool allow_self_mapping) {
  std::copy_if(
      exprs.begin(),
      exprs.end(),
      std::back_inserter(tv_exprs_),
      [](Expr* expr) {
        NVF_ERROR(expr != nullptr);
        return ir_utils::isTvOp(expr);
      });

  auto all_tvs = ir_utils::allTvsOfExprs(tv_exprs_);
  all_tvs.pushBack(additional_tvs.begin(), additional_tvs.end());

  tvs_ = all_tvs.vector();

  // Add uses and definitions to all iter domains.
  buildIterDomainDefinitionsAndUses();

  if (build_graphs) {
    buildAllGraphs();
  }
}

IdModel::IdModel(
    Fusion* fusion,
    bool build_graphs,
    bool allow_self_mapping,
    bool validate)
    : allow_self_mapping_(allow_self_mapping), validate_(validate) {
  auto all_exprs = fusion->exprs();
  std::copy_if(
      all_exprs.begin(),
      all_exprs.end(),
      std::back_inserter(tv_exprs_),
      [](Expr* expr) {
        NVF_ERROR(expr != nullptr);
        return ir_utils::isTvOp(expr);
      });

  auto all_tvs = ir_utils::allTvsOfExprs(tv_exprs_);

  {
    auto inp_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    all_tvs.pushBack(inp_tvs.begin(), inp_tvs.end());
  }
  {
    auto out_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
    all_tvs.pushBack(out_tvs.begin(), out_tvs.end());
  }

  tvs_ = all_tvs.vector();

  // Add uses and definitions to all iter domains.
  buildIterDomainDefinitionsAndUses();

  if (build_graphs) {
    buildAllGraphs();
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
  NVF_ERROR(
      graph_it != id_graphs_.end(),
      "Failed to find an IdGraph with the ",
      mode,
      " mode");
  return graph_it->second;
}

void IdModel::buildIterDomainDefinitionsAndUses() {
  for (const auto tv : tvs_) {
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

  // To deterministically initialize the graph, the order of adding
  // domains must be deterministic. Here, we sort all IDs by their
  // names.

  std::vector<IterDomain*> all_ids;
  all_ids.reserve(id_definitions_.size());
  for (const auto& [id, defs] : id_definitions_) {
    all_ids.push_back(id);
  }

  std::sort(
      all_ids.begin(), all_ids.end(), [](IterDomain* id1, IterDomain* id2) {
        return id1->name() < id2->name();
      });

  for (auto id : all_ids) {
    auto uses_it = id_uses_.find(id);
    NVF_ERROR(
        uses_it != id_uses_.end(),
        "Failed to initialize id: ",
        id->toString(),
        " as it's missing a definition entry.");
    id_graph.initializeVal(id, id_definitions_.at(id), uses_it->second);
  }

  return id_graph;
}

void IdModel::buildExactGraph() {
  // Initialize the maps with all the IterDomains used in the provded
  // expressions.
  NVF_ERROR(
      id_graphs_.emplace(IdMappingMode::EXACT, initializeIdGraph()).second);

  for (auto expr : tv_exprs_) {
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

  idGraph(IdMappingMode::EXACT).validateConsistency();
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

void IdModel::buildAlmostExactGraph() {
  // Make sure the exact graph is already built
  maybeBuildGraph(IdMappingMode::EXACT);

  // Build almost exact map by forwarding through broadcast axes
  NVF_ERROR(
      id_graphs_
          .emplace(IdMappingMode::ALMOSTEXACT, idGraph(IdMappingMode::EXACT))
          .second);

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

  almost_exact_graph.validateConsistency();
}

void IdModel::buildPermissiveGraph() {
  // Make sure the exact graph is already built
  maybeBuildGraph(IdMappingMode::EXACT);

  // Use the exact map as the starting map rather than the
  // almost-exact map. Almost exact is useful for index hoisting but
  // not necessary for permissive and loop maps
  NVF_ERROR(
      id_graphs_
          .emplace(IdMappingMode::PERMISSIVE, idGraph(IdMappingMode::EXACT))
          .second);

  for (auto expr : tv_exprs_) {
    // Multiple outputs are already mapped, we can ignore all but the first
    // consumer given they have to be replayed in the same exact way
    TensorView* c_tv = ir_utils::getTvOutput(expr);

    auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());

    for (auto p_tv : tv_inputs) {
      ForwardingInfo permissive_forwarding(p_tv, c_tv);
      for (auto entry : permissive_forwarding.producer_forwarding_map) {
        idGraph(IdMappingMode::PERMISSIVE).mapVals(entry.first, entry.second);
      }

      if (permissive_graph_map_compliment_ids_) {
        for (const auto& entry :
             permissive_forwarding.producer_compliment_map) {
          for (auto entry_2 : entry.second) {
            idGraph(IdMappingMode::PERMISSIVE).mapVals(entry.first, entry_2);
          }
        }
      }

      for (auto entry : permissive_forwarding.consumer_forwarding_map) {
        idGraph(IdMappingMode::PERMISSIVE).mapVals(entry.first, entry.second);
      }

      if (permissive_graph_map_compliment_ids_) {
        for (const auto& entry :
             permissive_forwarding.consumer_compliment_map) {
          for (auto entry_2 : entry.second) {
            idGraph(IdMappingMode::PERMISSIVE).mapVals(entry.first, entry_2);
          }
        }
      }

      auto permissive_c2p_root_map =
          PairwiseRootDomainMap(p_tv, c_tv).mapBroadcast(true);

      for (auto entry : permissive_c2p_root_map.mapConsumerToProducer()) {
        idGraph(IdMappingMode::PERMISSIVE).mapVals(entry.first, entry.second);
      }
    }
  }

  idGraph(IdMappingMode::PERMISSIVE).validateConsistency();
}

namespace {

// Returns the root producer iteration domains that are resolved by provided
// consumer
std::vector<std::pair<IterDomain*, IterDomain*>> resolvedRootBroadcasts(
    TensorView* producer,
    TensorView* consumer) {
  auto p2c_map = PairwiseRootDomainMap(producer, consumer)
                     .mapBroadcast(true)
                     .mapProducerToConsumer();

  std::vector<std::pair<IterDomain*, IterDomain*>> resolved_bcast_domains;
  for (const auto& [p_id, c_id] : p2c_map) {
    // Look for a broadcast producer and non-broadcast consumer

    // Ignore non-broadcast producer and broadcast consumer dims
    if (!p_id->isBroadcast() || c_id->isBroadcast()) {
      continue;
    }

    if (c_id->isReduction()) {
      // This should only happen with expanded broadcast
      // domains. Otherwise, squeeze should be used
      NVF_ERROR(
          p_id->hasExpandedExtent(), "Unexpected domain: ", c_id->toString());
      continue;
    }

    resolved_bcast_domains.emplace_back(p_id, c_id);
  }
  return resolved_bcast_domains;
}

} // namespace

// Grab inlining relationships
StatefulInliningInfo buildStatefulInliningInfo(
    const std::vector<Expr*>& exprs,
    const ValGraph& exact_graph,
    const ValGraph& permissive_graph) {
  StatefulInliningInfo info;
  for (auto expr : exprs) {
    for (auto producer_tv :
         ir_utils::filterByType<TensorView>(expr->inputs())) {
      const auto& producer_root = producer_tv->getMaybeRFactorDomain();
      const auto& producer_domain = producer_tv->domain()->leaf();

      // Grab all iteration domains in producer that its compute at iter domains
      // depend on.
      auto ca_dep_vals = DependencyCheck::getAllValsBetween(
          {producer_root.begin(), producer_root.end()},
          {producer_domain.begin(),
           producer_domain.begin() + producer_tv->getComputeAtPosition()});
      auto ca_deps_filter = ir_utils::filterByType<IterDomain>(ca_dep_vals);
      VectorOfUniqueEntries<IterDomain*> all_producer_ca_deps(
          ca_deps_filter.begin(), ca_deps_filter.end());

      info.ordered_p_ca_ids.pushBack(all_producer_ca_deps);

      // Gather info on and producer-consumer
      // mappings of CA domains and broadcast resolution
      for (auto consumer_tv :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        auto all_producer_ids = ir_utils::allIDsOf(producer_tv);
        auto all_consumer_ids = ir_utils::allIDsOf(consumer_tv);

        auto p2c_permissive_map = permissive_graph.buildMapBetween(
            all_producer_ids, all_consumer_ids);

        for (const auto& [p_id, c_ids] : p2c_permissive_map) {
          if (!c_ids.empty() &&
              all_producer_ca_deps.has(p_id->as<IterDomain>())) {
            info.p2c_ca_permissive_maps[p_id->as<IterDomain>()].pushBack(c_ids);
          }
        }

        const std::vector<std::pair<IterDomain*, IterDomain*>>
            resolved_bcast_domains =
                resolvedRootBroadcasts(producer_tv, consumer_tv);

        for (const auto& [p_root_id, c_root_id] : resolved_bcast_domains) {
          info.p2c_root_broadcast_resolution_map[p_root_id].pushBack(c_root_id);
        }
      }
    }
  }
  return info;
}

void IdModel::initializeLoopGraph(const StatefulInliningInfo& info) {
  // In the case of the Loop graph, we do not propagate mappings but
  // explicitly set which domains to map based on the permissive graph
  // and the CA positions.
  NVF_ERROR(
      id_graphs_.emplace(IdMappingMode::LOOP, initializeIdGraph(false)).second);

  // Make sure this is called in a deterministic order. Build all inlined
  // relationships in loop graph.
  for (IterDomain* p_id : info.ordered_p_ca_ids) {
    auto entry_it = info.p2c_ca_permissive_maps.find(p_id);
    if (entry_it != info.p2c_ca_permissive_maps.end()) {
      const VectorOfUniqueEntries<Val*>& c_ids = entry_it->second;
      for (Val* c_id : c_ids) {
        idGraph(IdMappingMode::LOOP).mapVals(p_id, c_id);
      }
    }
  }
}

void IdModel::buildLoopGraph() {
  // Make sure the depedent graphs are already built
  maybeBuildGraph(IdMappingMode::EXACT);
  maybeBuildGraph(IdMappingMode::PERMISSIVE);

  const StatefulInliningInfo inlining_info = buildStatefulInliningInfo(
      tv_exprs_,
      idGraph(IdMappingMode::EXACT),
      idGraph(IdMappingMode::PERMISSIVE));

  initializeLoopGraph(inlining_info);

  loop_promotion_map_ = buildLoopPromotionMap(inlining_info);

  idGraph(IdMappingMode::LOOP).validateConsistency();
}

std::unordered_map<ValGroup, IterDomain*> IdModel::buildLoopPromotionMap(
    const StatefulInliningInfo& inlining_info) {
  // Make an intersection of the exact and loop map. This will group together
  // entries in each loop group that are exact with each other. This provides a
  // better graph to do promotion and replays.
  //
  // It's tempting to use the intersection of the almost exact and loop, but we
  // need to model broadcast promotion, and if we have two tensors like:
  //
  // T1[i0, b1] = T0[i0]
  // T2[i0, b2] = T0[i0]
  // Then resolution of:
  // T4 = T1[i0, b1] + T3[i0, i1]
  // T6 = T2[i0, b2] + T5[i0, i2]
  //
  // Then merge(0, 1) with all tensors except for T0
  //
  // The almost exact map will map i0, i0*b1, and i0*b2 together, but b1 and b2
  // are being resolved to i1 and i2 respectively. So we want to have separate
  // entries so we can have an easy to process promotion map.
  //
  // Loop is a permissive like map, it could have many entries, use the exact
  // map as the one we iterate on to reduce complexity as it hopefully has
  // smaller groups and this algorithm scales with the number of groups *
  // (number of entries in groups ^ 2)
  //
  // iel stands for Intersection of the Exact and Loop graphs.
  ValGraph iel_graph = buildIntersection(
      idGraph(IdMappingMode::EXACT), idGraph(IdMappingMode::LOOP), false);

  // Step 1: Build a map of the IEL groups of root broadcast domains
  // to resolving domains.
  std::unordered_map<ValGroup, IterDomain*> iel_promotion_map =
      buildInlineRootResolutionMap(iel_graph, inlining_info);

  // Step 2: Propagate the root promotions to intermediate and leaf groups.
  // At this point, the promotion may not be final as the analysis is
  // localized to IEL groups. The map is used in the next step to
  // build mappings of the loop groups.
  propagatePromotionsInIELGraph(iel_graph, iel_promotion_map);

  // This is not a right map to return but just a placeholder since
  // the loop promotion map is not yet completely merged. It will be
  // replaced by a proper map.
  return iel_promotion_map;
}

std::unordered_map<ValGroup, IterDomain*> IdModel::buildInlineRootResolutionMap(
    const ValGraph& iel_graph,
    const StatefulInliningInfo& info) {
  std::unordered_map<ValGroup, IterDomain*> iel_promotion_map;

  // This should probably work just on terminating inputs, as we shouldn't be
  // able to modify a broadcast domain between root and rfactor which would be
  // required to resolve a non input broadcast domain. But for now leaving it as
  // traversal on all broadcast groups.
  //

  // We first visit all broadcast root domains. If a broadcast is
  // resovled, see if it's promoted. Note that a domain be resolved to
  // a domain that may not be loop mapped, yet it can still be
  // promoted. In other words, there can be a domain that is exactly
  // mapped with the resolving domain *and* is mapped with the
  // broadcast domain by the loop map. The algorihm here is:
  //
  // 1. For a broadcast domain, find the domain that the broadcast is
  //    resolved to.
  // 2. If the resolving domain is also loop-mapped with the
  //    broadcast, that is the promotion domain, but the resolving
  //    domain may not be loop mapped as mentioned above. Instead,
  //    find all loop-mapped domains with the broadcast domain and
  //    pick one that is exactly mapped with the resolving domain
  //
  // Note again this process is only done for root domains. Once we
  // find promotion relationships for root domains, we propagate the
  // mappings to derived domains
  for (const ValGroup& iel_group : iel_graph.disjointValSets().disjointSets()) {
    NVF_ERROR(!iel_group->empty());

    IterDomain* iel_group_id = iel_group->front()->as<IterDomain>();

    if (!iel_group_id->isBroadcast()) {
      continue;
    }

    // Collect all the exact groups of the resolutions of the broadcast id's
    ValGroups resolved_exact_groups;
    for (Val* bcast_id : *iel_group) {
      if (auto p2c_root_broadcast_resolution_map_it =
              info.p2c_root_broadcast_resolution_map.find(
                  bcast_id->as<IterDomain>());
          p2c_root_broadcast_resolution_map_it !=
          info.p2c_root_broadcast_resolution_map.end()) {
        resolved_exact_groups.pushBack(
            idGraph(IdMappingMode::EXACT)
                .toGroups(p2c_root_broadcast_resolution_map_it->second));
      }
    }

    if (resolved_exact_groups.empty()) {
      // No resolution
      continue;
    }

    // resolved_exact_groups is a list of IDs that resolves the
    // broadcast. We only care those that are also in the same loop
    // group, and there must be just one or none. Otherwise, the
    // resolution is ambiguous.

    // Collect all the exact groups in the loop set containing this iel_group
    const ValGroup& loop_group =
        idGraph(IdMappingMode::LOOP).toGroup(iel_group_id);
    ValGroups loop_covered_exact_groups =
        idGraph(IdMappingMode::EXACT).toGroups(*loop_group);

    // The intersection of the exact groups that the broadcast domains can be
    // broadcasted to, and those that exist within the same loop groop are is
    // the promotion needed for this iel_group. The promotion should
    // be none or unique.
    ValGroups loop_exact_resolved_intersection =
        resolved_exact_groups.computeIntersect(loop_covered_exact_groups);

    if (loop_exact_resolved_intersection.empty()) {
      // No promotion
      continue;
    }

    if (loop_exact_resolved_intersection.size() > 1) {
      // Ambiguous promotion. This should not happen.
      std::stringstream err_msg;
      err_msg
          << "Invalid multiple broadcast resolution within shared loops detected, group:\n  "
          << iel_group->toString() << "\nIs being broadcasted to:";
      for (const ValGroup& entry : loop_exact_resolved_intersection) {
        err_msg << "\n  " << entry->toString();
      }
      NVF_ERROR(false, err_msg.str());
    }

    const ValGroup& exact_resolution_group =
        loop_exact_resolved_intersection.front();

    // Within the loop group, find the IDs that the broadcast IDs are
    // resolved to
    VectorOfUniqueEntries<Val*> resolved_ids =
        exact_resolution_group->computeIntersect(*loop_group);

    NVF_ERROR(!resolved_ids.empty());

    // All the IDs in resolved_ids are mapped with both of the exact
    // and loop graphs, so any of them can be used as an IEL promotion
    // ID. Just to make it extra clear, look for corresponding
    // groups in the IEL graph and make sure there's only one such group.
    ValGroups promoted_iel_groups = iel_graph.toGroups(resolved_ids);

    NVF_ERROR(!promoted_iel_groups.empty());

    if (promoted_iel_groups.size() > 1) {
      std::stringstream err_msg;
      err_msg
          << "Invalid multiple broadcast resolution within shared loops detected, group:\n  "
          << iel_group->toString() << "\nIs being broadcasted to:";
      for (const ValGroup& entry : promoted_iel_groups) {
        err_msg << "\n  " << entry->toString();
      }
      NVF_ERROR(false, err_msg.str());
    }

    iel_promotion_map[iel_group] =
        promoted_iel_groups.front()->front()->as<IterDomain>();
  }

  return iel_promotion_map;
}

void IdModel::buildAllGraphs() {
  if (tvs_.empty()) {
    return;
  }

  std::unique_ptr<IdModelValidator> validator;

  Fusion* fusion = tvs_.front()->fusion();

  // A ComputeAtMap will be built inside the constructor of
  // IdModelValidator, which may fail for some fusions that are not
  // supported currently (but work with IdModel). Make sure the
  // validator is only created when it is indeed requested
  if (validate_) {
    validator = std::make_unique<IdModelValidator>(fusion, allow_self_mapping_);
  }

  FusionGuard fg(fusion);

  buildExactGraph();
  if (validate_) {
    validator->checkExactGraphEquivalence(idGraph(IdMappingMode::EXACT));
  }

  // Make sure there's no self mapping in the Exact graph as that
  // would invalidate lowering assumptions.
  if (!allow_self_mapping_) {
    assertNoSelfMapping();
  }

  buildAlmostExactGraph();
  if (validate_) {
    validator->checkAlmostExactGraphEquivalence(
        idGraph(IdMappingMode::ALMOSTEXACT));
  }

  buildPermissiveGraph();
  // Validation is not implemented when compliment mapping is enabled
  if (!permissive_graph_map_compliment_ids_ && validate_) {
    validator->checkPermissiveGraphEquivalence(
        idGraph(IdMappingMode::PERMISSIVE));
  }

  buildLoopGraph();
}

void IdModel::buildGraph(IdMappingMode mode) {
  switch (mode) {
    case IdMappingMode::EXACT:
      buildExactGraph();
      break;
    case IdMappingMode::ALMOSTEXACT:
      buildAlmostExactGraph();
      break;
    case IdMappingMode::PERMISSIVE:
      buildPermissiveGraph();
      break;
    case IdMappingMode::LOOP:
      buildLoopGraph();
      break;
    default:
      NVF_ERROR(false, "Unsupported mode: ", mode);
  }
}

void IdModel::maybeBuildGraph(IdMappingMode mode) {
  if (id_graphs_.find(mode) != id_graphs_.end()) {
    return;
  } else {
    buildGraph(mode);
  }
}

ValGraph IdModel::buildIntersection(
    const ValGraph& graph0,
    const ValGraph& graph1,
    bool propagate_exprs) {
  ValGraph intersection = initializeIdGraph(propagate_exprs);
  for (const ValGroup& group0 : graph0.disjointValSets().disjointSets()) {
    auto set_size = group0->size();
    for (auto id0_i : c10::irange(set_size)) {
      Val* id0 = group0->vector()[id0_i];
      for (auto id1_i = id0_i; id1_i < set_size; id1_i++) {
        Val* id1 = group0->vector()[id1_i];
        // id0 and id1 map in group0. If they also map in the group1,
        // add the mapping to the intersection.
        if (graph1.disjointValSets().strictAreMapped(id0, id1)) {
          intersection.mapVals(id0, id1);
        }
      }
    }
  }
  return intersection;
}

namespace {

// Check if there's an equivalent expression as iel_expr that uses
// maybe_promoted_inputs. This is used to avoid redundantly replaying
// expressions.
// NOTE: This is currently overly conservative and some
// opportunities for reuse are lost, althought it doesn't affect
// the correctness of the analysis.
Expr* findMatchingExpr(
    const ExprGroup& iel_expr,
    const ValGraph& iel_graph,
    const std::vector<IterDomain*>& maybe_promoted_inputs) {
  // If any of domains in maybe_promoted_inputs is not found in
  // iel_graph, it means the domain is just replayed and by definition
  // has no mapping with any existing domain, which means there's no
  // matching expr.
  if (std::any_of(
          maybe_promoted_inputs.begin(),
          maybe_promoted_inputs.end(),
          [&](IterDomain* maybe_promoted_input) -> bool {
            return !iel_graph.hasGroup(maybe_promoted_input);
          })) {
    return nullptr;
  }

  // Grab all eligible uses of the promoted inputs.
  // Note that any eligible matching expr should be a use of all
  // inputs in maybe_promoted_input_uses, no matter it's promoted or
  // not. So it isn't necessary to look at all of
  // maybe_promoted_input_uses but just need to grab one.
  NVF_ERROR(!maybe_promoted_inputs.empty());
  ExprGroups maybe_promoted_input_uses =
      iel_graph.getUses(iel_graph.toGroup(maybe_promoted_inputs.front()));

  if (maybe_promoted_input_uses.empty()) {
    return nullptr;
  }

  // Look for exprs that have inputs that are mapped in the IEL
  // graph with the (promoted) inputs of iel_expr.
  for (const ExprGroup& maybe_promoted_input_use_group :
       maybe_promoted_input_uses) {
    NVF_ERROR(!maybe_promoted_input_use_group->empty());
    // maybe_promoted_inputs may include non-promoted inputs as
    // well, so maybe_promoted_input_uses may include the original
    // iel_expr itself. Since there must at least be a promoted input,
    // iel_expr itself should not be an expr group we are looking for.
    if (iel_expr == maybe_promoted_input_use_group) {
      continue;
    }
    Expr* maybe_promoted_input_use = maybe_promoted_input_use_group->front();
    if (!iel_expr->front()->sameOp(maybe_promoted_input_use)) {
      continue;
    }
    // Check if all inputs are mapped
    NVF_ERROR(
        maybe_promoted_inputs.size() ==
        maybe_promoted_input_use->inputs().size());
    bool all_inputs_match = true;
    for (const auto inp_i : c10::irange(maybe_promoted_inputs.size())) {
      // Here, new promoted ids are not added to iel_graph, so
      // once promoted, this should not return true anymore. Also,
      // strictAreMapped doesn't work as promoted domains are not
      // in the graph
      all_inputs_match = all_inputs_match &&
          iel_graph.disjointValSets().permissiveAreMapped(
              maybe_promoted_inputs[inp_i],
              maybe_promoted_input_use->inputs().at(inp_i));
    }
    if (!all_inputs_match) {
      continue;
    }

    return maybe_promoted_input_use;
  }

  return nullptr;
}

} // namespace

void IdModel::propagatePromotionsInIELGraph(
    const ValGraph& iel_graph,
    std::unordered_map<ValGroup, IterDomain*>& iel_promotion_map) {
  // In order to make this traversal work, the traversal order must be
  // topologically sorted.
  ValGraphStmtSort iel_stmt_sort(iel_graph);

  for (const ExprGroup& iel_expr : iel_stmt_sort.exprs()) {
    NVF_ERROR(!iel_expr->empty());
    const std::vector<ValGroup> iel_inp_groups =
        iel_graph.inputGroups(iel_expr);

    // Check if any inputs need promotion indicating this expr group needs to
    // be replayed with promoted inputs
    bool an_input_was_promoted = false;
    std::vector<IterDomain*> maybe_promoted_inputs;
    maybe_promoted_inputs.reserve(iel_inp_groups.size());

    for (const ValGroup& iel_inp_group : iel_inp_groups) {
      // Assumed all inputs are IterDomains
      NVF_ERROR(iel_inp_group->front()->isA<IterDomain>());

      // Propagate IEL promotions when available.
      if (auto inp_promo_it = iel_promotion_map.find(iel_inp_group);
          inp_promo_it != iel_promotion_map.end()) {
        maybe_promoted_inputs.push_back(inp_promo_it->second);
        an_input_was_promoted = true;
        continue;
      }

      // No promotion found. Just use the non-promoted domain
      maybe_promoted_inputs.push_back(iel_inp_group->front()->as<IterDomain>());
    }

    if (!an_input_was_promoted) {
      // No inputs need promotion so just continue
      continue;
    }

    Expr* promoted_expr =
        findMatchingExpr(iel_expr, iel_graph, maybe_promoted_inputs);

    bool replayed = false;

    if (!promoted_expr) {
      promoted_expr = addReplayAs(maybe_promoted_inputs, iel_expr->front());
      replayed = true;
    }

    // Mark outputs as having a promoted iter domain
    std::vector<ValGroup> out_groups = iel_graph.outputGroups(iel_expr);
    NVF_ERROR(promoted_expr->outputs().size() == out_groups.size());
    NVF_ERROR(
        ir_utils::filterByType<IterDomain>(promoted_expr->outputs()).size() ==
            out_groups.size(),
        "Unexpected non IterDomain outputs found: ",
        promoted_expr->toString());

    for (const auto i : c10::irange(out_groups.size())) {
      // Promote if necessary, if the output is already in the same exact map
      // it doesn't need a promotion.
      if (idGraph(IdMappingMode::EXACT)
              .disjointValSets()
              .strictAreMapped(
                  promoted_expr->output(i), out_groups[i]->front())) {
        continue;
      }
      iel_promotion_map[out_groups[i]] =
          promoted_expr->output(i)->as<IterDomain>();
      // Explicitly map loop map since expr propagation doesn't happen
      if (replayed) {
        idGraph(IdMappingMode::LOOP)
            .mapVals(iel_expr->front()->output(i), promoted_expr->output(i));
      }
    }
  }
}

// Replay Expr but with the inputs provided.
Expr* IdModel::addReplayAs(std::vector<IterDomain*> new_inputs, Expr* expr) {
  // Figure out which graphs are already initialized to make sure we add the new
  // expression to them.
  std::vector<IdMappingMode> initialized_modes;
  for (auto mode : kIdMappingModes) {
    auto graph_it = id_graphs_.find(mode);
    if (graph_it == id_graphs_.end()) {
      continue;
    }

    auto& graph = graph_it->second;
    if (graph.disjointValSets().disjointSetMap().empty()) {
      continue;
    }

    initialized_modes.push_back(mode);
  }

  // Replace the provided inputs with IterType::Iteration domains as
  // reduction domains cannot be merged with non-reduction domains.
  if (std::any_of(
          new_inputs.begin(),
          new_inputs.end(),
          [](IterDomain* id) { return id->isReduction(); }) &&
      std::any_of(new_inputs.begin(), new_inputs.end(), [](IterDomain* id) {
        return !id->isReduction();
      })) {
    // Inputs have mismatched type, replace new_inputs
    auto tmp_inputs = new_inputs;
    for (const auto i : c10::irange(new_inputs.size())) {
      new_inputs.at(i) = IterDomainBuilder(tmp_inputs.at(i))
                             .iter_type(IterType::Iteration)
                             .build();
      id_definitions_[new_inputs.at(i)];
      id_uses_[new_inputs.at(i)];
      for (auto mode : initialized_modes) {
        idGraph(mode).initializeVal(new_inputs.at(i), {}, {});
        idGraph(mode).mapVals(new_inputs.at(i), tmp_inputs.at(i));
      }
    }
  }

  const std::vector<IterDomain*> orig_input_ids =
      ir_utils::filterByType<IterDomain>(expr->inputs()).vector();

  // Sanity check of the original inputs
  {
    NVF_ERROR(
        new_inputs.size() == orig_input_ids.size(),
        "Invalid number of inputs: ",
        new_inputs.size(),
        " does not match number of iter domain inputs for ",
        expr->toString());

    for (auto mode : initialized_modes) {
      for (auto inp : orig_input_ids) {
        NVF_ERROR(
            idGraph(mode).hasGroup(inp),
            "All inputs for replay need to be initialized in all graphs, ",
            inp->toString(),
            " was not found in mode: ",
            mode);
      }
    }
  }

  // Create the new expression with provided inputs
  auto replay = ReplayTransform::replayAs(new_inputs, expr);

  for (auto out_id : ir_utils::filterByType<IterDomain>(replay->outputs())) {
    id_definitions_[out_id].pushBack(replay);
    // out_id is a new IterDomain with no use expr yet. Initialize its
    // use mapping with an empty set
    NVF_ERROR(id_uses_.emplace(out_id, VectorOfUniqueEntries<Expr*>{}).second);
  }

  // Add the expression to the uses of the inputs
  for (auto inp_id : ir_utils::filterByType<IterDomain>(replay->inputs())) {
    // inp_id should not be a new domain, so just make sure it has a
    // def mapping.
    NVF_ERROR(id_definitions_.find(inp_id) != id_definitions_.end());
    id_uses_[inp_id].pushBack(replay);
  }

  // Initialize output iter domains in the graphs
  for (auto mode : initialized_modes) {
    auto& graph = idGraph(mode);

    // Initialize output ids in map. The replay expr will be
    // registered as a definition by registerExpr
    for (auto out_id : ir_utils::filterByType<IterDomain>(replay->outputs())) {
      idGraph(mode).initializeVal(out_id, {}, {});
    }

    idGraph(mode).registerExpr(replay);

    // Propagate through all the uses of the iter domain groups of the inputs
    // with the new expression.
    // Gather all use expressions from inputs
    VectorOfUniqueEntries<Expr*> representative_uses;
    for (IterDomain* inp : new_inputs) {
      for (const ExprGroup& use_group : graph.getUses(graph.toGroup(inp))) {
        NVF_ERROR(!use_group->empty());
        representative_uses.pushBack(use_group->front());
      }
    }

    for (auto rep_use : representative_uses) {
      graph.maybeMapThroughExprs(rep_use, replay, true);
    }
  }

  return replay;
}

} // namespace nvfuser
