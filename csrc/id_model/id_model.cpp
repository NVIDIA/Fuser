// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/id_model.h>
#include <id_model/loop_promotion.h>
#include <id_model/to_string.h>
#include <id_model/transform_replay.h>
#include <id_model/utils.h>
#include <id_model/validation_utils.h>

#include <device_lower/analysis/trivial_broadcast.h>
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <disjoint_set.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <logical_domain_map.h>
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

void IdModel::assertNoSelfMapping(const ValGraph& graph) const {
  for (TensorView* tv : tvs_) {
    std::optional<SelfMapping> self_mapping = hasSelfMapping(tv, graph);
    if (self_mapping.has_value()) {
      tv->fusion()->print();
    }
    NVF_CHECK(
        !self_mapping.has_value(),
        "Unsupported domain mapping detected in ",
        tv->toString(),
        ". ",
        self_mapping->where,
        " domains, ",
        self_mapping->id1->toString(),
        " and ",
        self_mapping->id2->toString(),
        ", are mapped with each other.");
  }
}

IdModel::IdModel(
    const std::vector<Expr*>& exprs,
    const std::vector<TensorView*>& additional_tvs,
    bool build_graphs,
    bool allow_self_mapping,
    LoopPromotionMapBuilderCallback* loop_promotion_map_builder_callback)
    : allow_self_mapping_(allow_self_mapping),
      loop_promotion_map_builder_callback_(
          loop_promotion_map_builder_callback) {
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

  NVF_ERROR(!tvs_.empty(), "No tensor to build IdModel for");

  fusion_ = tvs_.front()->fusion();

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
    bool validate,
    LoopPromotionMapBuilderCallback* loop_promotion_map_builder_callback)
    : fusion_(fusion),
      allow_self_mapping_(allow_self_mapping),
      validate_(validate),
      loop_promotion_map_builder_callback_(
          loop_promotion_map_builder_callback) {
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
    std::vector<IterDomain*> all_ids = tv->domain()->allIDs();

    // Check if this domain is a consumer of a view-like operation
    const bool view_like_domain = tv->domain()->hasViewLikeRFactor();

    for (auto id : all_ids) {
      // Check if this id is a view like rfactor id
      if (view_like_domain && id->isRFactorProduct()) {
        // If the tensor domain is a view like domain, and the iteration
        // domain is marked as an rfactor product and is in the rfactor
        // domain, it's a view like rfactor iteration domain
        const auto& logical_domain = tv->domain()->logical();
        if (std::find(logical_domain.begin(), logical_domain.end(), id) !=
            logical_domain.end()) {
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

      if (def == nullptr) {
        continue;
      }

      // If any of the inputs is not included in the all ID set, do
      // not include the definition in the model. Note that it is
      // possible that some are included but not all since a single ID
      // may be used by multiple exprs.
      if (std::any_of(
              def->inputs().begin(), def->inputs().end(), [&](Val* inp) {
                return std::find(
                           all_ids.begin(),
                           all_ids.end(),
                           inp->as<IterDomain>()) == all_ids.end();
              })) {
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

ValGraph IdModel::initializeIdGraph(bool propagate_through_exprs) const {
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

namespace {
// In Exact and AlmostExact graphs, for all IDs of a group that have
// static extents, they should be equal.
void checkStaticExtentGroups(const ValGraph& graph) {
  std::stringstream err_msg;
  for (const ValGroup& group : graph.disjointValSets().disjointSets()) {
    std::optional<int64_t> known_static_extent;
    for (const auto val : *group) {
      auto id = val->as<IterDomain>();
      if (!id->extent()->isConstScalar()) {
        continue;
      }

      auto extent_int = id->extent()->evaluate().as<int64_t>();
      if (known_static_extent.has_value()) {
        if (known_static_extent.value() != extent_int) {
          err_msg << "Different static extents found in an ID group: "
                  << known_static_extent.value() << " and " << extent_int
                  << " in {" << toDelimitedString(group->vector()) << "}\n";
          break;
        }
      } else {
        known_static_extent = extent_int;
      }
    }
  }

  NVF_ERROR(err_msg.str().empty(), err_msg.str());
}
} // namespace

ValGraph& IdModel::buildExactGraph() {
  // Initialize the maps with all the IterDomains used in the provded
  // expressions.
  NVF_ERROR(
      id_graphs_.emplace(IdMappingMode::EXACT, initializeIdGraph()).second);

  auto& graph = idGraph(IdMappingMode::EXACT);

  for (auto expr : tv_exprs_) {
    TensorView* c_tv = ir_utils::getTvOutput(expr);

    NVF_ERROR(
        c_tv != nullptr,
        "Expected to have a TensorView output: ",
        expr->toString());

    auto all_tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());

    // Map siblings, as all other tv output domains must match the first tv
    // outputs domain.
    std::deque<TensorView*> other_tv_outputs(
        all_tv_outputs.begin(), all_tv_outputs.end());
    other_tv_outputs.pop_front();

    // Map producer-consumer relationships based on the root domain map
    auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    for (auto p_tv : tv_inputs) {
      // For exact mapings do not map any broadcast dimensions to
      // non-broadcast dimensions. Prevent any broadcasted axes being mapped
      // to non-broadcasted axes.
      auto exact_c2p_logical_map = PairwiseLogicalDomainMap(p_tv, c_tv)
                                       .mapBroadcast(false)
                                       .mapConsumerToProducer();

      for (auto c_id :
           getSortedKeys(exact_c2p_logical_map, Statement::lessThan)) {
        auto p_id = exact_c2p_logical_map.at(c_id);
        graph.mapVals(c_id, p_id);
      }
    }

    if (ir_utils::hasUniformSiblings(expr)) {
      for (auto other_tv_output : other_tv_outputs) {
        NVF_ERROR(
            other_tv_output->getMaybeRootDomain().size() ==
                c_tv->getMaybeRootDomain().size(),
            "Multiple outputs with mismatched TV domains is not supported.");

        for (auto domain_i : arange(c_tv->getMaybeRootDomain().size())) {
          auto c_id = c_tv->getMaybeRootDomain()[domain_i];
          auto o_id = other_tv_output->getMaybeRootDomain()[domain_i];
          graph.mapVals(o_id, c_id);
        }
      }
    } else {
      for (auto p_tv : tv_inputs) {
        for (auto c_tv : other_tv_outputs) {
          auto exact_c2p_root_map = PairwiseLogicalDomainMap(p_tv, c_tv)
                                        .mapBroadcast(false)
                                        .mapConsumerToProducer();

          for (auto c_id :
               getSortedKeys(exact_c2p_root_map, Statement::lessThan)) {
            auto p_id = exact_c2p_root_map.at(c_id);
            graph.mapVals(c_id, p_id);
          }
        }
      }
    }

    // TODO: Revisit if we really should map domains in the exact map
    mapThroughLoopSwizzles(graph);
  }

  // Map additional exact mappings if registered. Only map those that
  // appear in this IdModel and when they are the same (per sameAs).
  if (!tv_exprs_.empty()) {
    Fusion* fusion = tv_exprs_.front()->fusion();
    if (fusion->hasRegisteredExactMappings()) {
      DisjointSets<IterDomain*> additional_mappings =
          fusion->registeredExactMappings();
      for (const auto& disjoint_set : additional_mappings.disjointSets()) {
        IterDomain* registerd_id = nullptr;
        for (auto id : *disjoint_set) {
          if (!graph.hasGroup(id)) {
            continue;
          }

          if (registerd_id == nullptr) {
            registerd_id = id;
          } else {
            graph.mapVals(registerd_id, id);
          }
        }
      }
    }
  }

  graph.validateConsistency();

  // Make sure there's no self mapping in the Exact graph as that
  // would invalidate lowering assumptions.
  if (!allow_self_mapping_) {
    assertNoSelfMapping(graph);
  }

  if (isOptionEnabled(EnableOption::IdModelExtraValidation)) {
    checkStaticExtentGroups(graph);
  }

  return graph;
}

namespace {

// Checks if the expression is a trivial operation where an input is simply an
// output of the transformation. Returns the mapped iter domains if found.
std::vector<std::vector<Val*>> getTriviallyMappedIds(Expr* expr) {
  std::vector<std::vector<Val*>> mapped_ids;
  if (auto merge = dynamic_cast<Merge*>(expr)) {
    // Note that broacast IDs may have extents larger than 1, thus
    // merge->inner()->isBroadcast() is not a sufficient condition to
    // check. Merging a non-broadcast ID with such a broadcast ID
    // result in a non-broadcast ID with extent multiplied by the
    // broadcast extent.
    if (merge->inner()->extent()->isOneInt()) {
      mapped_ids.push_back({merge->outer(), merge->out()});
    }
    if (merge->outer()->extent()->isOneInt()) {
      mapped_ids.push_back({merge->inner(), merge->out()});
    }
  } else if (auto split = dynamic_cast<Split*>(expr)) {
    if (split->factor()->isOneInt()) {
      if (split->innerSplit()) {
        mapped_ids.push_back({split->in(), split->outer()});
      } else {
        mapped_ids.push_back({split->in(), split->inner()});
      }
    } else {
      // Rare, but don't want to deal with zero-dim IDs.
      // If the input ID is a size-one ID (not necessarily broadcast,
      // e.g., may be reduction) and the factor is not one, mapping
      // the input and the size-one output can be inconvenient for
      // predicate indexing. See
      // PredicateIndexingTest.NonTrivialSizeOneDomain for a concrete
      // example.
      if (!split->in()->extent()->isZeroInt() &&
          !split->in()->extent()->isOneInt()) {
        // Even when the factor is not known to be 1, as long as the
        // input and output have the same extent, they should be
        // mapped. This happens, for example, split 32 by 32 -> 1, 32.
        if (split->outer()->extent()->sameAs(split->in()->extent())) {
          // In and outer have the same extent. They must be non-one and
          // the inner must be one, or they must be one.
          NVF_ERROR(
              split->inner()->extent()->isOneInt() ||
                  split->outer()->extent()->isOneInt(),
              "Unexpected split: ",
              split->toString());
          mapped_ids.push_back({split->in(), split->outer()});
        }
        if (split->inner()->extent()->sameAs(split->in()->extent())) {
          NVF_ERROR(
              split->inner()->extent()->isOneInt() ||
                  split->outer()->extent()->isOneInt(),
              "Unexpected split: ",
              split->toString());
          mapped_ids.push_back({split->in(), split->inner()});
        }
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

ValGraph& IdModel::buildAlmostExactGraph() {
  // Make sure the exact graph is already built
  maybeBuildGraph(IdMappingMode::EXACT);

  // Build almost exact map by forwarding through broadcast axes
  NVF_ERROR(
      id_graphs_
          .emplace(IdMappingMode::ALMOSTEXACT, idGraph(IdMappingMode::EXACT))
          .second);

  auto& almost_exact_graph = idGraph(IdMappingMode::ALMOSTEXACT);

  // Even when EXACT has no self mapping, there was a case ALMOSTEXACT
  // had self mapping (see issue #3919). ALMOSTEXACT is used in
  // indexing, which assumes the graph has no self mapping. To avoid
  // self mapping, mark each of the root, logical and loop domains of
  // all tensors unmappable
  for (TensorView* tv : tvs_) {
    if (tv->hasRoot()) {
      almost_exact_graph.setUnmappable(
          {tv->getRootDomain().begin(), tv->getRootDomain().end()});
    }
    almost_exact_graph.setUnmappable(
        {tv->getLogicalDomain().begin(), tv->getLogicalDomain().end()});
    almost_exact_graph.setUnmappable(
        {tv->getLoopDomain().begin(), tv->getLoopDomain().end()});
  }

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
          ids_to_map.emplace_back(mapped_id_group.front(), id);
        }
      }
    }
  }

  for (const auto& [id1, id2] : ids_to_map) {
    almost_exact_graph.mapVals(id1, id2);
  }

  almost_exact_graph.validateConsistency();

  if (!allow_self_mapping_) {
    assertNoSelfMapping(almost_exact_graph);
  }

  if (isOptionEnabled(EnableOption::IdModelExtraValidation)) {
    checkStaticExtentGroups(almost_exact_graph);
  }

  return almost_exact_graph;
}

ValGraph& IdModel::buildBroadcastGraph() {
  // Make sure the exact graph is already built
  maybeBuildGraph(IdMappingMode::EXACT);

  // Use the exact graph as the starting map rather than the
  // almost-exact graph. Almost exact is useful for index hoisting but
  // not necessary otherwise
  NVF_ERROR(
      id_graphs_
          .emplace(IdMappingMode::BROADCAST, idGraph(IdMappingMode::EXACT))
          .second);

  auto& graph = idGraph(IdMappingMode::BROADCAST);

  for (auto expr : tv_exprs_) {
    for (TensorView* c_tv :
         ir_utils::filterByType<TensorView>(expr->outputs())) {
      auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());

      for (auto p_tv : tv_inputs) {
        auto permissive_c2p_logical_map =
            PairwiseLogicalDomainMap(p_tv, c_tv).mapBroadcast(true);

        for (auto entry : permissive_c2p_logical_map.mapConsumerToProducer()) {
          graph.mapVals(entry.first, entry.second);
        }
      }

      // If all outputs are uniformly mapped, only the first consumer
      // needs to be examined
      if (ir_utils::hasUniformSiblings(expr)) {
        break;
      }
    }
  }

  graph.validateConsistency();

  return graph;
}

ValGraph& IdModel::buildPermissiveGraph() {
  maybeBuildGraph(IdMappingMode::BROADCAST);

  NVF_ERROR(
      id_graphs_
          .emplace(IdMappingMode::PERMISSIVE, idGraph(IdMappingMode::BROADCAST))
          .second);

  auto& graph = idGraph(IdMappingMode::PERMISSIVE);

  for (auto expr : tv_exprs_) {
    for (TensorView* c_tv :
         ir_utils::filterByType<TensorView>(expr->outputs())) {
      auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());

      // If the loop domain is not generated from the logial domain
      // with not extra IDs, broadcast forwarding is not
      // supported. As such, permissive mappings are not generated.
      if (!ir_utils::isLoopDomainFullyDerivedFromLogicalDomain(c_tv)) {
        continue;
      }

      for (auto p_tv : tv_inputs) {
        if (!ir_utils::isLoopDomainFullyDerivedFromLogicalDomain(p_tv)) {
          continue;
        }
        ForwardingInfo permissive_forwarding(p_tv, c_tv);
        for (auto entry : permissive_forwarding.producer_forwarding_map) {
          graph.mapVals(entry.first, entry.second);
        }

        if (permissive_graph_map_compliment_ids_) {
          for (const auto& entry :
               permissive_forwarding.producer_compliment_map) {
            for (auto entry_2 : entry.second) {
              graph.mapVals(entry.first, entry_2);
            }
          }
        }

        for (auto entry : permissive_forwarding.consumer_forwarding_map) {
          graph.mapVals(entry.first, entry.second);
        }

        if (permissive_graph_map_compliment_ids_) {
          for (const auto& entry :
               permissive_forwarding.consumer_compliment_map) {
            for (auto entry_2 : entry.second) {
              graph.mapVals(entry.first, entry_2);
            }
          }
        }
      }
      // If all outputs are uniformly mapped, only the first consumer
      // needs to be examined
      if (ir_utils::hasUniformSiblings(expr)) {
        break;
      }
    }
  }

  graph.validateConsistency();

  return graph;
}

namespace {

// Returns the root producer iteration domains that are resolved by provided
// consumer
std::vector<std::pair<IterDomain*, IterDomain*>> resolvedRootBroadcasts(
    TensorView* producer,
    TensorView* consumer) {
  auto p2c_map = PairwiseLogicalDomainMap(producer, consumer)
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
    const ValGraph& permissive_graph) {
  StatefulInliningInfo info;
  for (auto expr : exprs) {
    for (auto producer_tv :
         ir_utils::filterByType<TensorView>(expr->inputs())) {
      const auto& producer_logical = producer_tv->getLogicalDomain();
      const auto& producer_domain = producer_tv->domain()->loop();

      // Broadcast forwarding is not applied when the loop domain is
      // not fully derived from the logical domain. In that case, the
      // loop promotion analysis effectively does nothing, however, we
      // still need to make loop groups, for which ordered_p_ca_ids as
      // well as p2c_ca_permissive_maps are required. Since no
      // promotion analysis is done, only loop IDs need to be
      // considered.
      auto fully_derived =
          ir_utils::isLoopDomainFullyDerivedFromLogicalDomain(producer_tv);

      // Gather info on and producer-consumer
      // mappings of CA domains and broadcast resolution
      for (auto consumer_tv :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        // Grab all iteration domains in producer that its compute at iter
        // domains depend on.
        VectorOfUniqueEntries<IterDomain*> all_producer_ca_deps;
        if (fully_derived) {
          auto ca_dep_vals = DependencyCheck::getAllValsBetween(
              {producer_logical.begin(), producer_logical.end()},
              {producer_domain.begin(),
               producer_domain.begin() +
                   producer_tv->getComputePosition(consumer_tv)});
          auto ca_deps_filter = ir_utils::filterByType<IterDomain>(ca_dep_vals);
          all_producer_ca_deps = VectorOfUniqueEntries<IterDomain*>(
              ca_deps_filter.begin(), ca_deps_filter.end());
        } else {
          all_producer_ca_deps = VectorOfUniqueEntries<IterDomain*>(
              producer_tv->getLoopDomain().begin(),
              producer_tv->getLoopDomain().begin() +
                  producer_tv->getComputePosition(consumer_tv));
        }
        info.ordered_p_ca_ids.pushBack(all_producer_ca_deps);

        auto all_producer_ids = producer_tv->domain()->allIDs();
        auto all_consumer_ids = consumer_tv->domain()->allIDs();

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

    if (ir_utils::hasUniformSiblings(expr)) {
      auto consumer_tvs = ir_utils::filterByType<TensorView>(expr->outputs());
      if (consumer_tvs.size() > 1) {
        auto all_consumer_ids = consumer_tvs.vector().at(0)->domain()->allIDs();
        info.ordered_sibling_ids.pushBack(
            {all_consumer_ids.begin(), all_consumer_ids.end()});
        for (const auto i : arange(1, consumer_tvs.size())) {
          auto consumer_tv_i = consumer_tvs.vector().at(i);
          auto all_consumer_i_ids = consumer_tv_i->domain()->allIDs();

          auto sibling_map = permissive_graph.buildMapBetween(
              all_consumer_ids, all_consumer_i_ids);

          for (const auto& [c_id_1, c_ids] : sibling_map) {
            // Note that c_ids can have multiple domains as this graph
            // is a Permissive graph and there may be broadcast merged
            // domains
            info.sibling_maps[c_id_1->as<IterDomain>()].pushBack(c_ids);
          }
        }
      }
    }
  }

  // Gather all async operations.
  std::vector<Expr*> async_warp;
  std::copy_if(
      exprs.begin(), exprs.end(), std::back_inserter(async_warp), [](Expr* e) {
        // TODO Add support for blackwell MmaOp
        return ir_utils::isCpAsyncBulkLoad(e);
      });

  if (async_warp.size() <= 1) {
    return info;
  }

  // TODO Divide into AsyncWarps
  // Assume single AsyncWarp with same stage_slice_position
  std::vector<TensorView*> async_warp_tvs;
  std::transform(
      async_warp.begin(),
      async_warp.end(),
      std::back_inserter(async_warp_tvs),
      [](Expr* e) {
        auto output_tvs =
            ir_utils::filterByType<TensorView>(e->outputs()).vector();
        NVF_ERROR(output_tvs.size() == 1);
        return output_tvs.front();
      });
  NVF_ERROR(async_warp_tvs.size() > 1);

  std::vector<int64_t> aw_stage_slice_positions;
  std::transform(
      async_warp_tvs.begin(),
      async_warp_tvs.end(),
      std::back_inserter(aw_stage_slice_positions),
      [](TensorView* tv) {
        std::optional<int64_t> opt_stage_slice_position =
            ir_utils::getStageSlicePosition(tv);
        return opt_stage_slice_position.value_or(-1);
      });
  NVF_ERROR(aw_stage_slice_positions.size() > 1);

  NVF_ERROR(std::all_of(
      aw_stage_slice_positions.begin() + 1,
      aw_stage_slice_positions.end(),
      [&](int64_t v) { return v == aw_stage_slice_positions.front(); }));

  TensorView* async_warp_tv = async_warp_tvs.front();
  NVF_ERROR(async_warp_tv != nullptr);
  int64_t stage_slice_position = aw_stage_slice_positions.front();

  VectorOfUniqueEntries<IterDomain*> all_inline_deps(
      async_warp_tv->getLoopDomain().begin(),
      async_warp_tv->getLoopDomain().begin() + stage_slice_position);
  info.ordered_sibling_ids.pushBack(all_inline_deps);

  auto all_tv_ids = async_warp_tv->domain()->allIDs();
  for (const auto i : arange(1, async_warp_tvs.size())) {
    auto tv_i = async_warp_tvs.at(i);
    auto all_tv_i_ids = tv_i->domain()->allIDs();

    auto sibling_map =
        permissive_graph.buildMapBetween(all_tv_ids, all_tv_i_ids);
    for (const auto& [tv_id_1, tv_ids] : sibling_map) {
      // Note that tv_ids can have multiple domains as this graph
      // is a Permissive graph and there may be broadcast merged
      // domains
      if (!tv_ids.empty() && all_inline_deps.has(tv_id_1->as<IterDomain>())) {
        info.sibling_maps[tv_id_1->as<IterDomain>()].pushBack(tv_ids);
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

  // Similarly maps all sibling domains
  for (IterDomain* id : info.ordered_sibling_ids) {
    auto entry_it = info.sibling_maps.find(id);
    if (entry_it != info.sibling_maps.end()) {
      const VectorOfUniqueEntries<Val*>& sibling_ids = entry_it->second;
      for (Val* sibling_id : sibling_ids) {
        idGraph(IdMappingMode::LOOP).mapVals(id, sibling_id);
      }
    }
  }
}

ValGraph& IdModel::buildLoopGraph(bool force_full_loop_promotion_analysis) {
  // Make sure the depedent graphs are already built
  maybeBuildGraph(IdMappingMode::EXACT);
  maybeBuildGraph(IdMappingMode::PERMISSIVE);

  const StatefulInliningInfo inlining_info =
      buildStatefulInliningInfo(tv_exprs_, idGraph(IdMappingMode::PERMISSIVE));

  initializeLoopGraph(inlining_info);

  validateLoopGraphHasNoSelfMappedLeafDomains();

  loop_promotion_map_ = LoopPromotionMapBuilder::get(
      *this,
      inlining_info,
      loop_promotion_map_builder_callback_,
      force_full_loop_promotion_analysis);

  // New domains are added. Make sure there's still no self mapping in
  // the loop domains
  validateLoopGraphHasNoSelfMappedLeafDomains();

  idGraph(IdMappingMode::LOOP).validateConsistency();

  return idGraph(IdMappingMode::LOOP);
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

  buildAlmostExactGraph();

  buildPermissiveGraph();
  // Validation is not implemented when compliment mapping is enabled
  if (!permissive_graph_map_compliment_ids_ && validate_) {
    validator->checkPermissiveGraphEquivalence(
        idGraph(IdMappingMode::PERMISSIVE));
  }

  buildLoopGraph();
}

ValGraph& IdModel::buildGraph(IdMappingMode mode) {
  switch (mode) {
    case IdMappingMode::EXACT:
      return buildExactGraph();
    case IdMappingMode::ALMOSTEXACT:
      return buildAlmostExactGraph();
    case IdMappingMode::BROADCAST:
      return buildBroadcastGraph();
    case IdMappingMode::PERMISSIVE:
      return buildPermissiveGraph();
    case IdMappingMode::LOOP:
      return buildLoopGraph();
    default:
      NVF_THROW("Unsupported mode: ", mode);
  }
}

ValGraph& IdModel::maybeBuildGraph(IdMappingMode mode) {
  auto it = id_graphs_.find(mode);
  if (it != id_graphs_.end()) {
    return it->second;
  } else {
    return buildGraph(mode);
  }
}

void IdModel::removeGraph(IdMappingMode mode) {
  id_graphs_.erase(mode);
}

ValGraph IdModel::buildIntersection(
    const ValGraph& graph0,
    const ValGraph& graph1,
    bool propagate_exprs) const {
  ValGraph intersection = initializeIdGraph(propagate_exprs);
  for (const ValGroup& group0 : graph0.disjointValSets().disjointSets()) {
    auto set_size = group0->size();
    for (auto id0_i : arange(set_size)) {
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
    for (const auto i : arange(new_inputs.size())) {
      new_inputs.at(i) = IterDomainBuilder(tmp_inputs.at(i))
                             .iter_type(IterType::Iteration)
                             .build();
      id_definitions_[new_inputs.at(i)];
      id_uses_[new_inputs.at(i)];
      for (auto mode : initialized_modes) {
        idGraph(mode).initializeVal(
            new_inputs.at(i), idGraph(mode).toGroup(tmp_inputs.at(i)));
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
  NVF_ERROR(replay != nullptr, "no replay found");

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
      graph.initializeVal(out_id, {}, {});
    }

    graph.registerExpr(replay);

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

void IdModel::validateLoopGraphHasNoSelfMappedLeafDomains() const {
  for (auto tv : tvs_) {
    auto self_mappped_loop_pair =
        detectSelfMapping(tv->domain()->loop(), idGraph(IdMappingMode::LOOP));
    NVF_ERROR(
        !self_mappped_loop_pair.has_value(),
        "Detected loop domains are mapped in the loop graph. Tensor: ",
        tv->toString(),
        ". Mapped loop domains: ",
        self_mappped_loop_pair->first->toString(),
        " and ",
        self_mappped_loop_pair->second->toString());
  }
}

std::unordered_map<ValGroup, IterDomain*> updateValGroupIdMap(
    const std::unordered_map<ValGroup, IterDomain*>& stale_map,
    ValGraph& new_graph) {
  std::unordered_map<ValGroup, IterDomain*> new_map;

  for (const auto& [stale_group, mapped_id] : stale_map) {
    const ValGroups& new_groups = new_graph.toGroups(*stale_group);
    NVF_ERROR(
        new_groups.size() == 1,
        "\nUpdate map assumes that new graph is equivalent to old graph plus "
        "extra mappings.\n",
        "i.e. all mappings in new_graph should exist in the graph stale_map "
        "was produced on.\n",
        "old:",
        nvfuser::toString(stale_group),
        "new: ",
        nvfuser::toString(new_groups));
    NVF_ERROR(
        new_map.emplace(new_groups.front(), mapped_id).second,
        "Expected only a single mapping but multiple entries detected for ",
        nvfuser::toString(new_groups.front()));
  }
  return new_map;
}

// Mostly just copied from ComputeAtMap::validateAndPropagatePType
void IdModel::validateAndPropagatePType() {
  for (const ValGroup& loop_group :
       idGraph(IdMappingMode::LOOP).disjointValSets().disjointSets()) {
    ParallelType common_ptype = ParallelType::Serial;
    for (Val* id : *loop_group) {
      auto id_ptype = id->as<IterDomain>()->getParallelType();
      NVF_ERROR(
          id_ptype == common_ptype || id_ptype == ParallelType::Serial ||
              common_ptype == ParallelType::Serial,
          "Issue validating parallel type disjoint ptype is, ",
          common_ptype,
          " but found in the set the id: ",
          id->toString());
      common_ptype =
          common_ptype == ParallelType::Serial ? id_ptype : common_ptype;
    }

    for (auto id : *loop_group) {
      // Due to the broadcast forwarding, not all IDs in a loop group
      // are indeed loop domains. For example, an ID may be used in a
      // merge whose output is also in this loop group.
      bool not_a_loop_domain = false;
      for (auto expr : id->uses()) {
        if (auto merge = dynamic_cast<Merge*>(expr);
            merge != nullptr && loop_group->has(merge->out())) {
          not_a_loop_domain = true;
          break;
        }
        // This is another case of input-output mappings
        if (auto swizzle2d = dynamic_cast<Swizzle2D*>(expr);
            swizzle2d != nullptr &&
            swizzle2d->swizzleMode() == SwizzleMode::Loop) {
          not_a_loop_domain = true;
          break;
        }
      }
      if (not_a_loop_domain) {
        continue;
      }
      id->as<IterDomain>()->parallelize(common_ptype);
    }
  }
}

void IdModel::allocateLoopIndexVariables() {
  FusionGuard fg(fusion_);

  NVF_ERROR(GpuLower::hasCurrent());

  NVF_ERROR(
      hasIdGraph(IdMappingMode::LOOP),
      "getLoopIndexVariable requires Loop graph");

  // Follow the same logic as ComputeAtMap::allocateIndexVariables
  for (const ValGroup& loop_group :
       idGraph(IdMappingMode::LOOP).disjointValSets().disjointSets()) {
    auto loop_promotion_map_it = loop_promotion_map_.find(loop_group);

    // Not all loop groups actually correspond to a for-loop. Ideally,
    // non for-loop loop groups should be removed. Such loop groups do
    // not need indices and don't have loop promotion.
    if (loop_promotion_map_it == loop_promotion_map_.end()) {
      continue;
    }

    ParallelType ptype = getParallelType(loop_group);

    Val* loop_index = nullptr;

    // TODO: Cleanup needed. ir_utils::isMemoryPartitionedAcross
    // should be used, but that means we would need to consider
    // multiple outputs with different memory types, though it
    // should be uncommon in practice.
    if (shouldUseZeroIndex(loop_group, *this) ||
        isParallelTypeDeviceDim(ptype)) {
      loop_index = fusion_->zeroVal();
    } else if (isParallelTypeThread(ptype)) {
      loop_index = NamedScalar::getParallelIndex(ptype);
    }

    if (loop_index != nullptr) {
      loop_index_variable_map_[loop_group] = loop_index;
      continue;
    }

    if (GpuLower::current()->circularBufferInfo().isCircularBufferedIterDomain(
            loop_group->front()->as<IterDomain>())) {
      // Allocate index variable for each stage of the circular
      // buffered loop.
      auto indices = std::make_unique<CircularBufferIndices>();
      for (auto i :
           arange(static_cast<int>(CircularBufferLoopStage::EndOfStages))) {
        indices->emplace(
            static_cast<CircularBufferLoopStage>(i),
            IrBuilder::create<Val>(DataType::Index));
      }
      circular_buffered_loop_index_variable_map_[loop_group] =
          std::move(indices);
      continue;
    }

    // If enabled, allocate own indices. Otherwise, use the one
    // generated for ComputeAtMap for compatibility with the legacy
    // indexing
    if (GpuLower::current()->idModelOptions().loop()) {
      loop_index = IrBuilder::create<Val>(DataType::Index);
    } else {
      const auto& ca_map = GpuLower::current()->caMap();
      for (const auto& id :
           ir_utils::filterByType<IterDomain>(loop_group->vector())) {
        if (!ca_map->getIdSets(IdMappingMode::LOOP).mappingExists(id)) {
          continue;
        }
        loop_index = ca_map->getIndexVariable(id);
        break;
      }
      NVF_ERROR(
          loop_index != nullptr,
          "No existing index found for ",
          nvfuser::toString(loop_group));
    }

    NVF_ERROR(loop_index != nullptr);
    loop_index_variable_map_[loop_group] = loop_index;
  }

  return;
}

Val* IdModel::getLoopIndexVariable(
    const ValGroup& loop_group,
    CircularBufferLoopStage circular_buffer_loop_stage) const {
  NVF_ERROR(
      !loop_index_variable_map_.empty(),
      "Loop index variables not generated. IdModel::allocateIndexVariables may "
      "have not been callled.");

  // Check if this loop was modified by circular buffer pass.
  bool is_circular_buffer_iterdomain =
      GpuLower::current()->circularBufferInfo().isCircularBufferedIterDomain(
          loop_group->front()->as<IterDomain>());

  if (is_circular_buffer_iterdomain) {
    // Use dedicated circular buffer index variable if the loop is circular
    // buffer loop
    if (circular_buffer_loop_stage == CircularBufferLoopStage::NotApplicable) {
      // The circular buffered loop stages are created after the loop nest
      //  lowering phase so this function will be querried before the double
      //  buffer pass. At that point, no forloop has any circular buffer
      //  stage defined, and we just default to using the main stage index.
      circular_buffer_loop_stage = CircularBufferLoopStage::Main;
    }
    return circular_buffered_loop_index_variable_map_.at(loop_group)
        ->at(circular_buffer_loop_stage);
  } else {
    return loop_index_variable_map_.at(loop_group);
  }
}

Val* IdModel::getLoopIndexVariable(
    IterDomain* id,
    CircularBufferLoopStage circular_buffer_loop_stage) const {
  const auto& loop_group = idGraph(IdMappingMode::LOOP).toGroup(id);
  return getLoopIndexVariable(loop_group, circular_buffer_loop_stage);
}

} // namespace nvfuser
