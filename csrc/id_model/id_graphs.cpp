// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/id_graphs.h>
#include <id_model/to_string.h>
#include <id_model/transform_replay.h>
#include <id_model/utils.h>
#include <id_model/visitor.h>

#include <device_lower/analysis/trivial_broadcast.h>
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <disjoint_set.h>
#include <ir/utils.h>
#include <root_domain_map.h>
#include <transform_iter.h>

#include <tuple>
#include <typeinfo>

namespace nvfuser {

void IterDomainGraphs::assertNoSelfMapping() {
  if (hasSelfMapping()) {
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
}

IterDomainGraphs::IterDomainGraphs(
    const std::vector<Expr*>& exprs,
    const std::vector<TensorView*>& additional_tvs,
    bool allow_self_mapping) {
  build(exprs, additional_tvs);

  if (!allow_self_mapping) {
    assertNoSelfMapping();
  }
}

IterDomainGraphs::IterDomainGraphs(
    const std::vector<Expr*>& exprs,
    bool allow_self_mapping)
    : IterDomainGraphs(exprs, {}, allow_self_mapping) {}

IterDomainGraphs::IterDomainGraphs(Fusion* fusion, bool allow_self_mapping) {
  std::vector<TensorView*> inputs_and_outputs;
  {
    auto inp_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    inputs_and_outputs.insert(
        inputs_and_outputs.begin(), inp_tvs.begin(), inp_tvs.end());
  }
  {
    auto out_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
    inputs_and_outputs.insert(
        inputs_and_outputs.begin(), out_tvs.begin(), out_tvs.end());
  }

  build(fusion->exprs(), inputs_and_outputs);

  if (!allow_self_mapping) {
    assertNoSelfMapping();
  }
}

const IdGraph& IterDomainGraphs::idGraph(IdMappingMode mode) const {
  auto graph_it = id_graphs_.find(mode);
  NVF_ERROR(graph_it != id_graphs_.end());
  return graph_it->second;
}

IdGraph& IterDomainGraphs::idGraph(IdMappingMode mode) {
  auto graph_it = id_graphs_.find(mode);
  NVF_ERROR(graph_it != id_graphs_.end());
  return graph_it->second;
}

Expr* IterDomainGraphs::idUse(IterDomain* id) const {
  auto use_it = id_uses_.find(id);
  if (use_it == id_uses_.end()) {
    return nullptr;
  }
  return use_it->second.front();
}

Expr* IterDomainGraphs::idDef(IterDomain* id) const {
  auto def_it = id_definitions_.find(id);
  if (def_it == id_definitions_.end()) {
    return nullptr;
  }
  return def_it->second.front();
}

namespace {

// Returns the first pair of id's in ids detected to match eachother on the
// permissive map of the ID graph. TODO: what this is really looking for is if
// there's any overlapping between the iter domains in the provided set.
//
// i.e. if we have:
// tv0 = arange(6).view({3, 2})
// tv1 = tv0[3, 2].t()
// tv2 = tv0[3, 2].view({2, 3})
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
c10::optional<std::pair<IterDomain*, IterDomain*>> detectMappablePair(
    const std::vector<IterDomain*>& ids,
    const IterDomainGraphs& id_graph,
    IdMappingMode mode) {
  for (auto id1 : ids) {
    for (auto id2 : ids) {
      if (id1 == id2) {
        continue;
      }
      if (id_graph.idGraph(mode).disjointIdSets().permissiveAreMapped(
              id1, id2)) {
        return std::make_pair(id1, id2);
      }
    }
  }

  return {};
}

// It is assumed that for any tensor represented by a list of domains,
// those domains should never be mapped with each other. It may be
// possible to lift this assumption, but it's unclear if it could
// matter in practice.
c10::optional<std::tuple<TensorView*, IterDomain*, IterDomain*, std::string>>
findFirstSelfMapping(
    const std::vector<TensorView*>& all_tvs,
    const IterDomainGraphs& id_graph) {
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
  return c10::nullopt;
}

} // namespace

void IterDomainGraphs::buildIterDomainDefinitionsAndUses(
    const std::vector<TensorView*>& all_tvs) {
  for (auto tv : all_tvs) {
    VectorOfUniqueEntries<IterDomain*> root_domain_ids{
        tv->getRootDomain().begin(), tv->getRootDomain().end()};

    auto all_ids = ir_utils::allIDsOf(tv);

    // Check is this domain is a consumer of a view-like operation
    bool view_like_domain = tv->domain()->hasViewLikeRFactor();

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
        id_definitions_[id] = {};
      }

      if (id_uses_.find(id) == id_uses_.end()) {
        id_uses_[id] = {};
      }

      auto def = id->definition();

      if (def == nullptr || root_domain_ids.has(id)) {
        continue;
      }

      if (id_definitions_.find(id) == id_definitions_.end()) {
        id_definitions_[id] = {};
      }
      id_definitions_.at(id).pushBack(def);

      auto inp_ids = ir_utils::filterByType<IterDomain>(def->inputs());
      for (auto inp_id : inp_ids) {
        if (id_uses_.find(inp_id) == id_uses_.end()) {
          id_uses_[inp_id] = {};
        }
        id_uses_.at(inp_id).pushBack(def);
      }
    }
  }
}

std::string IterDomainGraphs::toString() const {
  // Figure out which graphs are already initialized to make sure we add the new
  // expression to them.
  std::vector<IdMappingMode> initialized_modes;
  for (auto mode : kIdMappingModes) {
    auto graph_it = id_graphs_.find(mode);
    if (graph_it == id_graphs_.end()) {
      continue;
    }

    auto& graph = graph_it->second;
    if (graph.disjointIdSets().disjointSetMap().empty()) {
      continue;
    }

    initialized_modes.push_back(mode);
  }

  std::stringstream ss;
  ss << "IterDomainGraphs { \n";
  for (auto mode : initialized_modes) {
    std::stringstream ss;
    ss << "  IdGraph " << mode << "{ \n";
    ss << "  Disjoint Ids:\n"
       << idGroupsString(idGraph(mode), 2)
       << "\n  Disjoint Expression groups:\n"
       << exprGroupsString(idGraph(mode), 2) << std::endl;
    ss << "   } IdGraph\n" << std::endl;
    return ss.str();
  }
  ss << " } IterDomainGraphs\n" << std::endl;
  return ss.str();
}

// Replay Expr but with the inputs provided.
Expr* IterDomainGraphs::addReplayAs(
    std::vector<IterDomain*> new_inputs,
    Expr* expr) {
  // Figure out which graphs are already initialized to make sure we add the new
  // expression to them.
  std::vector<IdMappingMode> initialized_modes;
  for (auto mode : kIdMappingModes) {
    auto graph_it = id_graphs_.find(mode);
    if (graph_it == id_graphs_.end()) {
      continue;
    }

    auto& graph = graph_it->second;
    if (graph.disjointIdSets().disjointSetMap().empty()) {
      continue;
    }

    initialized_modes.push_back(mode);
  }

  auto orig_inputs = ir_utils::filterByType<IterDomain>(expr->inputs());
  std::vector<IterDomain*> orig_input_ids(
      orig_inputs.begin(), orig_inputs.end());

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
    decltype(new_inputs) tmp_inputs;
    std::swap(tmp_inputs, new_inputs);
    for (auto tmp_input : tmp_inputs) {
      new_inputs.push_back(
          IterDomainBuilder(tmp_input).iter_type(IterType::Iteration).build());
      id_definitions_[new_inputs.back()];
      id_uses_[new_inputs.back()];
      for (auto mode : initialized_modes) {
        idGraph(mode).initializeId(new_inputs.back(), {}, {});
        idGraph(mode).mapIds(new_inputs.back(), tmp_input);
      }
    }
  }

  {
    NVF_ERROR(
        new_inputs.size() == orig_input_ids.size(),
        "Invalid number of inputs: ",
        new_inputs.size(),
        " does not match number of iter domain inputs for ",
        expr->toString());

    VectorOfUniqueEntries<IterDomain*> all_inputs{
        orig_input_ids.begin(), orig_input_ids.end()};

    all_inputs.pushBack(new_inputs);

    for (auto mode : initialized_modes) {
      for (auto inp : all_inputs) {
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
    id_uses_[out_id];
  }

  // Add the expression to the uses of the inputs
  for (auto inp_id : ir_utils::filterByType<IterDomain>(replay->inputs())) {
    id_definitions_[inp_id];
    id_uses_[inp_id].pushBack(replay);
  }

  // Initialize output iter domains in the graphs
  for (auto mode : initialized_modes) {
    idGraph(mode).disjointExprSets().initializeSet(replay);
    auto replay_group = idGraph(mode).toGroup(replay);

    // Initialize output ids in map
    for (auto out_id : ir_utils::filterByType<IterDomain>(replay->outputs())) {
      idGraph(mode).initializeId(out_id, {replay}, {});
    }

    // Update uses of the inputs in the graphs
    for (auto inp_id : ir_utils::filterByType<IterDomain>(replay->inputs())) {
      auto inp_group = idGraph(mode).toGroup(inp_id);
      idGraph(mode).addUniqueUses(inp_group, replay_group);
    }

    // Propagate through all the uses of the iter domain groups of the inputs
    // with the new expression.
    auto& graph = idGraph(mode);
    // Gather all use expressions from inputs
    VectorOfUniqueEntries<Expr*> representative_uses;
    for (IterDomain* inp : new_inputs) {
      auto uses_pair = graph.getUses(graph.toGroup(inp));
      if (uses_pair.second) {
        for (const ExprGroup& use_group : uses_pair.first) {
          representative_uses.pushBack(use_group->front());
        }
      }
    }

    for (auto rep_use : representative_uses) {
      graph.maybeMapThroughExprs(rep_use, replay, true);
    }
  }

  return replay;
}

// Generate a new expr with the IterDomain inputs/outputs replaced based on map.
// Replaced inputs/outputs should almost exact match with provided expr.
Expr* IterDomainGraphs::addExprWithReplacement(
    const std::unordered_map<IterDomain*, IterDomain*>& old_2_new_ids,
    Expr* old_expr) {
  // Figure out which graphs are already initialized to make sure we add the new
  // expression to them.
  std::vector<IdMappingMode> initialized_modes;
  for (auto mode : kIdMappingModes) {
    auto graph_it = id_graphs_.find(mode);
    if (graph_it == id_graphs_.end()) {
      continue;
    }

    auto& graph = graph_it->second;
    if (graph.disjointIdSets().disjointSetMap().empty()) {
      continue;
    }

    initialized_modes.push_back(mode);
  }

  // We will fill this map for every IterDomain in input and output.
  std::unordered_map<IterDomain*, IterDomain*> replacement_map = old_2_new_ids;

  // Validate replacement map. Make sure the keys are an input or output
  for (auto replacement_entry : replacement_map) {
    NVF_ERROR(
        std::find(
            old_expr->inputs().begin(),
            old_expr->inputs().end(),
            replacement_entry.first) != old_expr->inputs().end() ||
            std::find(
                old_expr->outputs().begin(),
                old_expr->outputs().end(),
                replacement_entry.first) != old_expr->outputs().end(),
        "Wanted to replace ",
        replacement_entry.first->toString(),
        " however the is not an input or output of:\n",
        old_expr->toString());
  }

  // If all inputs and or all output were replaced
  bool all_inps_replaced = true;
  bool all_outs_replaced = true;
  {
    for (auto inp_id : ir_utils::filterByType<IterDomain>(old_expr->inputs())) {
      if (replacement_map.find(inp_id) == replacement_map.end()) {
        all_inps_replaced = false;
        replacement_map[inp_id] = inp_id->cloneWithoutRFactor();
      }
    }

    for (auto out_id :
         ir_utils::filterByType<IterDomain>(old_expr->outputs())) {
      if (replacement_map.find(out_id) == replacement_map.end()) {
        all_outs_replaced = false;
        replacement_map[out_id] = out_id->cloneWithoutRFactor();
      }
    }

    NVF_ERROR(
        (all_inps_replaced || all_outs_replaced),
        "Either all the inputs or all the outputs need to be replaced when using this function.");

    for (auto mode : initialized_modes) {
      for (auto inp_or_out_id : all_inps_replaced
               ? ir_utils::filterByType<IterDomain>(old_expr->inputs())
               : ir_utils::filterByType<IterDomain>(old_expr->outputs())) {
        NVF_ERROR(
            idGraph(mode).hasGroup(inp_or_out_id),
            "Expected ",
            inp_or_out_id->toString(),
            " to be initialized in graph mode: ",
            mode);
      }
    }
  }

  // Create the new expression with provided outputs
  auto replay = ReplacementTransformCloner::clone(replacement_map, old_expr);

  // Add new output iter domains to id_definitions_/id_uses_ of IdGraphs
  for (auto out_id : ir_utils::filterByType<IterDomain>(replay->outputs())) {
    id_definitions_[out_id].pushBack(replay);
    id_uses_[out_id];
  }

  // Add new input iter domains to id_definitions_/id_uses_ of IdGraphs
  for (auto inp_id : ir_utils::filterByType<IterDomain>(replay->inputs())) {
    id_definitions_[inp_id];
    id_uses_[inp_id].pushBack(replay);
  }

  // Update all the initialized graph mappings
  for (auto mode : initialized_modes) {
    auto& graph = idGraph(mode);

    graph.disjointExprSets().initializeSet(replay);
    auto replay_group = graph.toGroup(replay);

    // Initialize any non-existant input ids, update existing ones
    for (auto inp_id : ir_utils::filterByType<IterDomain>(replay->inputs())) {
      if (!graph.disjointIdSets().mappingExists(inp_id)) {
        // inp_id is not initialized in the map, initialize it
        graph.initializeId(inp_id, {}, {replay});
      } else {
        // Update unique uses of existing input ids
        auto inp_group = graph.toGroup(inp_id);
        graph.addUniqueUses(inp_group, replay_group);
      }
    }

    // Initialize any non-existant output ids, update existing ones
    for (auto out_id : ir_utils::filterByType<IterDomain>(replay->outputs())) {
      if (!graph.disjointIdSets().mappingExists(out_id)) {
        // out_id is not initialized in the map, initialize it
        graph.initializeId(out_id, {replay}, {});
      } else {
        // out_id is already initialized, add the replay as a unique definition
        // of its group
        auto out_group = graph.toGroup(out_id);
        graph.addUniqueDefinitions(out_group, replay_group);
      }
    }

    // If the inputs were replaced we want to map through forward the newly
    // added expression. If the outputs were replaced we want to map through
    // backwards the newly added expression.

    // Forward
    VectorOfUniqueEntries<Expr*> representative_uses;
    for (auto in : ir_utils::filterByType<IterDomain>(replay->inputs())) {
      auto uses_pair = graph.getUses(graph.toGroup(in));
      if (uses_pair.second) {
        for (const ExprGroup& use_group : uses_pair.first) {
          if (use_group == replay_group) {
            continue;
          }
          representative_uses.pushBack(use_group->front());
        }
      }
    }

    for (auto rep_use : representative_uses) {
      graph.maybeMapThroughExprs(rep_use, replay, true);
    }

    // Backwards
    VectorOfUniqueEntries<Expr*> representative_defs;
    for (auto out : ir_utils::filterByType<IterDomain>(replay->outputs())) {
      auto defs_pair = graph.getDefinitions(graph.toGroup(out));
      if (defs_pair.second) {
        for (const ExprGroup& def_group : defs_pair.first) {
          if (def_group == replay_group) {
            continue;
          }
          representative_defs.pushBack(def_group->front());
        }
      }
    }

    for (auto rep_def : representative_defs) {
      graph.maybeMapThroughExprs(rep_def, replay, false);
    }
  }
  return replay;
}

// Clone provided iter domain and return the new copy. Map that copy in relevant
// maps.
IterDomain* IterDomainGraphs::cloneIterDomain(IterDomain* id) {
  // Figure out which graphs are already initialized to make sure we add the new
  // expression to them.
  std::vector<IdMappingMode> initialized_modes;
  for (auto mode : kIdMappingModes) {
    auto graph_it = id_graphs_.find(mode);
    if (graph_it == id_graphs_.end()) {
      continue;
    }

    auto& graph = graph_it->second;
    if (graph.disjointIdSets().disjointSetMap().empty()) {
      continue;
    }

    initialized_modes.push_back(mode);
  }

  auto id_copy = id->cloneWithoutRFactor();

  id_uses_[id_copy] = {};
  id_definitions_[id_copy] = {};

  for (auto mode : initialized_modes) {
    idGraph(mode).initializeId(id_copy, {}, {});
    idGraph(mode).mapIds(id, id_copy);
  }

  return id_copy;
}

IdGraph IterDomainGraphs::initializeIdGraph(bool propagate_through_exprs) {
  IdGraph id_graph(propagate_through_exprs);

  for (const auto& [id, defs] : id_definitions_) {
    auto uses_it = id_uses_.find(id);
    NVF_ERROR(
        uses_it != id_uses_.end(),
        "Failed to initialize id: ",
        id->toString(),
        " as it's missing a definition entry.");
    id_graph.initializeId(id, defs, uses_it->second);
  }

  return id_graph;
}

void IterDomainGraphs::buildExactMap(const std::vector<Expr*>& exprs) {
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
        idGraph(IdMappingMode::EXACT).mapIds(o_id, c_id);
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
        idGraph(IdMappingMode::EXACT).mapIds(c_id, p_id);
      }
    }

    idGraph(IdMappingMode::EXACT).mapThroughLoopSwizzles();
  }
}

void IterDomainGraphs::buildPermissiveMap(const std::vector<Expr*>& exprs) {
  // Use the exact map as the starting map rather than the
  // almost-exact map. Almost exact is useful for index hoisting but
  // not necessary for permissive and loop maps
  idGraph(IdMappingMode::PERMISSIVE) = idGraph(IdMappingMode::EXACT);

  for (auto expr : exprs) {
    // Multiple outputs are already mapped, we can ignore all but the first
    // consumer given they have to be replayed in the same exact way
    // Multiple outputs are already mapped, we can ignore all but the first
    // consumer given they have to be replayed in the same exact way
    TensorView* c_tv = ir_utils::getTvOutput(expr);

    auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());

    for (auto p_tv : tv_inputs) {
      auto p_ids_vec = ir_utils::allIDsOf(p_tv);
      auto c_ids_vec = ir_utils::allIDsOf(c_tv);
      std::unordered_set<IterDomain*> p_ids(p_ids_vec.begin(), p_ids_vec.end());
      std::unordered_set<IterDomain*> c_ids(c_ids_vec.begin(), c_ids_vec.end());

      ForwardingInfo permissive_forwarding(p_tv, c_tv);
      for (auto entry : permissive_forwarding.producer_forwarding_map) {
        idGraph(IdMappingMode::PERMISSIVE).mapIds(entry.first, entry.second);
      }

      // TODO: Should this just get rolled up in the forwarding map now?
      for (const auto& entry : permissive_forwarding.producer_compliment_map) {
        for (auto entry_2 : entry.second) {
          idGraph(IdMappingMode::PERMISSIVE).mapIds(entry.first, entry_2);
        }
      }

      for (auto entry : permissive_forwarding.consumer_forwarding_map) {
        idGraph(IdMappingMode::PERMISSIVE).mapIds(entry.first, entry.second);
      }

      // TODO: Should this just get rolled up in the forwarding map now?
      // TODO: Why should IDs be mapped to their compliments? Is this right?
      for (const auto& entry : permissive_forwarding.consumer_compliment_map) {
        for (auto entry_2 : entry.second) {
          idGraph(IdMappingMode::PERMISSIVE).mapIds(entry.first, entry_2);
        }
      }

      auto permissive_c2p_root_map = PairwiseRootDomainMap(p_tv, c_tv);

      for (auto entry : permissive_c2p_root_map.mapConsumerToProducer()) {
        idGraph(IdMappingMode::PERMISSIVE).mapIds(entry.first, entry.second);
      }
    }
  }
  idGraph(IdMappingMode::PERMISSIVE).mapThroughLoopSwizzles();
}

void IterDomainGraphs::buildAlmostExactMap() {
  // Build almost exact map by forwarding through broadcast axes
  idGraph(IdMappingMode::ALMOSTEXACT) = idGraph(IdMappingMode::EXACT);
  idGraph(IdMappingMode::ALMOSTEXACT).mapThroughTrivialExprs();
}

// TODO: Reenable after reenabling parallel propagation.
//        propagateLoopPTypes
void IterDomainGraphs::validatePTypes(
    const std::vector<TensorView*>& all_tvs) const {
  // VectorOfUniqueEntries<IterDomain*> leaf_ids;
  // for (auto tv : all_tvs) {
  //   leaf_ids.pushBack(tv->domain()->leaf());
  // }

  // for (const auto& disjoint_set :
  //      idGraph(IdMappingMode::EXACT).disjointIdSets().disjointSets()) {
  //   for (auto id : disjoint_set->vector()) {
  //     auto id_ptype = id->getParallelType();

  //     NVF_ERROR(
  //         leaf_ids.has(id) || id_ptype == ParallelType::Serial,
  //         "Invalid parallelization of non leaf iter domain: ",
  //         id->toString());
  //   }
  // }
}

void IterDomainGraphs::propagateLoopPTypes() const {
  for (const auto& loop_disjoint_set :
       idGraph(IdMappingMode::LOOP).disjointIdSets().disjointSets()) {
    ParallelType common_ptype = ParallelType::Serial;
    for (auto id : loop_disjoint_set->vector()) {
      auto id_ptype = id->getParallelType();

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

    for (auto id : loop_disjoint_set->vector()) {
      id->parallelize(common_ptype);
    }
  }
}

namespace {
struct StatefulLoweringInfo {
  // Tracks all p2c mappings in permissive maps even those not inlined between
  // producer and consumer
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      p2c_permissive_maps;

  // All consumer ids in a deterministic order (ignores fusion->inputs())
  VectorOfUniqueEntries<IterDomain*> ordered_c_ids;

  // p2c mappings through the fusion within (including dependencies of) inlined
  // leaf domains.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      p2c_ca_permissive_maps;

  // All producer ids within (including dependencies of) inlined leaf domains,
  // used for deterministic order
  VectorOfUniqueEntries<IterDomain*> ordered_p_ca_ids;

  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      p2c_root_broadcast_resolution_map;
};

// Returns the root producer iteration domains that are resolved by provided
// consumer
std::unordered_map<IterDomain*, IterDomain*> resolvedRootBroadcasts(
    TensorView* producer,
    TensorView* consumer) {
  auto p2c_map =
      PairwiseRootDomainMap(producer, consumer).mapProducerToConsumer();

  std::unordered_map<IterDomain*, IterDomain*> resolved_bcast_map;
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

    resolved_bcast_map[p_id] = c_id;
  }
  return resolved_bcast_map;
}

StatefulLoweringInfo buildInfo(
    const std::vector<Expr*>& exprs,
    const IdGraph& exact_graph,
    const IdGraph& permissive_graph) {
  StatefulLoweringInfo info;
  // Grab inlining relationships
  for (auto expr : exprs) {
    for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
      auto producer_root = producer->getMaybeRFactorDomain();
      auto producer_domain = producer->domain()->leaf();

      // Grab all iteration domains in producer that its compute at iter domains
      // depend on.
      VectorOfUniqueEntries<IterDomain*> all_producer_ca_deps;
      {
        auto ca_dep_vals = DependencyCheck::getAllValsBetween(
            {producer_root.begin(), producer_root.end()},
            {producer_domain.begin(),
             producer_domain.begin() + producer->getComputeAtPosition()});
        auto ca_deps_filter = ir_utils::filterByType<IterDomain>(ca_dep_vals);

        all_producer_ca_deps.insert(
            ca_deps_filter.begin(), ca_deps_filter.end());
      }

      info.ordered_p_ca_ids.pushBack(all_producer_ca_deps);
      for (auto consumer :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        auto resolved_bcast_map = resolvedRootBroadcasts(producer, consumer);

        for (const auto& [p_id, c_id] : resolved_bcast_map) {
          info.p2c_root_broadcast_resolution_map[p_id].pushBack(c_id);
          for (auto other_exact_bcast : *(exact_graph.toGroup(p_id))) {
            if (p_id == other_exact_bcast) {
              continue;
            }
            if (all_producer_ca_deps.has(other_exact_bcast)) {
              // TODO-NM: Why is this here? Can be removed?
              NVF_ERROR(
                  false,
                  "Can this happen? Adding other exact: ",
                  other_exact_bcast->name(),
                  " in addition to ",
                  p_id->name(),
                  " of ",
                  producer->toString());
              info.p2c_root_broadcast_resolution_map[other_exact_bcast]
                  .pushBack(c_id);
            }
          }
        }

        auto all_producer_ids = ir_utils::allIDsOf(producer);
        auto all_consumer_ids = ir_utils::allIDsOf(consumer);
        info.ordered_c_ids.pushBack(all_consumer_ids);

        auto p2c_permissive_map = permissive_graph.buildMapBetween(
            all_producer_ids, all_consumer_ids);

        for (const auto& entry : p2c_permissive_map) {
          if (entry.second.empty()) {
            continue;
          }
          if (all_producer_ca_deps.has(entry.first)) {
            info.p2c_ca_permissive_maps[entry.first].pushBack(entry.second);
          }
          info.p2c_permissive_maps[entry.first].pushBack(entry.second);
        }

        for (const auto& entry : p2c_permissive_map) {
          if (entry.second.empty()) {
            continue;
          }
          info.p2c_permissive_maps[entry.first].pushBack(entry.second);
        }
      }
    }
  }
  return info;
}

} // namespace

void IterDomainGraphs::build(
    const std::vector<Expr*>& exprs,
    const std::vector<TensorView*>& additional_tvs) {
  // Initialize the required sets as if a permissive relationship is never
  // found, then querying an empty permissive map will fail later.
  // Initialize disjoint sets
  for (auto mode : kIdMappingModes) {
    id_graphs_[mode] = IdGraph();
  }

  std::vector<Expr*> tv_exprs;

  std::copy_if(
      exprs.begin(), exprs.end(), std::back_inserter(tv_exprs), [](Expr* expr) {
        NVF_ERROR(expr != nullptr);
        return ir_utils::isTvOp(expr);
      });

  auto all_tvs = ir_utils::allTvsOfExprs(tv_exprs);
  if (!additional_tvs.empty()) {
    std::unordered_set<TensorView*> all_added_tvs(
        all_tvs.begin(), all_tvs.end());
    for (auto additional_tv : additional_tvs) {
      if (all_added_tvs.find(additional_tv) == all_added_tvs.end()) {
        all_tvs.push_back(additional_tv);
      }
    }
  }

  if (all_tvs.empty()) {
    return;
  }

  FusionGuard fg(all_tvs.front()->fusion());
  // Add uses and definitions to all iter domains.
  buildIterDomainDefinitionsAndUses(all_tvs);

  // Initialize the maps with all the IterDomains used in the provded
  // expressions.
  idGraph(IdMappingMode::EXACT) = initializeIdGraph();

  buildExactMap(tv_exprs);
  buildAlmostExactMap();
  buildPermissiveMap(tv_exprs);

  // Permissive graph needs the trivial exprs from the almost exact graph to
  // build correctly. Once built though we can remove the trivial expressions
  // from the almost exact graph.
  idGraph(IdMappingMode::ALMOSTEXACT).removeTrivialExprs();

  // Only build loop map during lowering
  // TODO: make this configurable
  if (true || FusionGuard::getCurFusion()->isA<kir::Kernel>()) {
    validatePTypes(all_tvs);

    StatefulLoweringInfo info = buildInfo(
        tv_exprs,
        idGraph(IdMappingMode::EXACT),
        idGraph(IdMappingMode::PERMISSIVE));

    initializeLoopMap(info);

    // Initial propagation of parallel types for inlined iter domains. Each time
    // new expressions are replayed this needs to be run. The disjoint sets in
    // the loop graph can only be joined after this point.
    // propagateLoopPTypes();

    auto iel_promotion_map = buildInlinePromotions(info);
    // propagateLoopPTypes();

    // Find loops that need to be promoted because of broadcast resolution,
    // figure out what that resolution should look like, compute IDs for it if
    // necessary.
    iel_promotion_map =
        buildLoopPromotionMap(tv_exprs, info, iel_promotion_map);
    // Loop map potentialy changed changed, as we could have replayed
    // expressions. Re-propagate parallel types.
    // propagateLoopPTypes();

    // This pass still doesn't work, disable for now in case it's disruptive to
    // tests.
    /*
    // Find loops that need to be promoted because of broadcast resolution,
    // figure out what that resolution should look like, compute IDs for it if
    // necessary.
    auto leaf_id_promo_map =
        buildIndexGraph(tv_exprs, all_tvs, info, iel_promotion_map);
    // Make sure we update ptypes onto the index leaf iter domains
    propagateLoopPTypes();
    */
  }

  // Debug, make sure there's no self mapping in TensorView's during lowering
  // that would invalidate lowering assumptions.
  self_mapping_info_ = findFirstSelfMapping(all_tvs, *this);
}

VectorOfUniqueEntries<IterDomain*> IterDomainGraphs::computeTerminalLoopIds(
    const StatefulLoweringInfo info) {
  VectorOfUniqueEntries<IterDomain*> terminal_loop_ids;
  for (const IdGroup& group :
       idGraph(IdMappingMode::LOOP).disjointIdSets().disjointSets()) {
    if (group->size() == 1) {
      terminal_loop_ids.pushBack(group->front());
    }

    // Don't select producer iter domains
    for (auto loop_id : *group) {
      if (info.p2c_ca_permissive_maps.find(loop_id) !=
          info.p2c_ca_permissive_maps.end()) {
        continue;
      }

      auto uses_it = id_uses_.find(loop_id);
      if (uses_it == id_uses_.end()) {
        terminal_loop_ids.pushBack(loop_id);
        continue;
      }

      // If there's an output group that is not in the same group, then it's id
      // consumer terminal. Also if there's no output groups it's id consumer
      // terminal.
      bool all_outs_in_loop_group = uses_it->second.empty() ? false : true;
      for (auto use : uses_it->second) {
        for (auto out_id : ir_utils::filterByType<IterDomain>(use->outputs())) {
          if (group != idGraph(IdMappingMode::LOOP).toGroup(out_id)) {
            all_outs_in_loop_group = false;
          }
        }
      }

      if (!all_outs_in_loop_group) {
        terminal_loop_ids.pushBack(loop_id);
      }
    }
  }
  return terminal_loop_ids;
}

IdGraph IterDomainGraphs::buildIntersection(
    const IdGraph& graph0,
    const IdGraph& graph1,
    bool propagate_exprs) {
  auto intersection = initializeIdGraph(propagate_exprs);
  for (const auto& group0 : graph0.disjointIdSets().disjointSets()) {
    auto set_size = group0->size();
    for (auto id0_i : c10::irange(set_size)) {
      auto id0 = group0->vector()[id0_i];
      for (auto id1_i = id0_i; id1_i < set_size; id1_i++) {
        auto id1 = group0->vector()[id1_i];
        // id0 and id1 map in group0. If they also map in the group1,
        // add the mapping to the inersection.
        if (graph1.disjointIdSets().strictAreMapped(id0, id1)) {
          intersection.mapIds(id0, id1);
        }
      }
    }
  }
  return intersection;
}

void IterDomainGraphs::initializeLoopMap(StatefulLoweringInfo& info) {
  // See Indexing20 example for why we shouldn't propagate when generating loop
  // groups
  idGraph(IdMappingMode::LOOP) = initializeIdGraph(false);

  // Make sure this is called in a deterministic order. Build all inlined
  // relationships in loop graph.
  for (IterDomain* p_id : info.ordered_p_ca_ids) {
    auto entry_it = info.p2c_ca_permissive_maps.find(p_id);
    if (entry_it != info.p2c_ca_permissive_maps.end()) {
      const VectorOfUniqueEntries<IterDomain*>& c_ids = entry_it->second;
      for (IterDomain* c_id : c_ids) {
        idGraph(IdMappingMode::LOOP).mapIds(p_id, c_id);
      }
    }
  }
}

std::unordered_map<IdGroup, IterDomain*> IterDomainGraphs::
    buildInlinePromotions(StatefulLoweringInfo& info) {
  // Make an intersection of the exact and loop map. This will group together
  // entries in each loop group that are exact with each other. This provides a
  // better graph to do promotion and replays.

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

  IdGraph intersection_exact_loop_graph = buildIntersection(
      idGraph(IdMappingMode::EXACT), idGraph(IdMappingMode::LOOP), false);

  // Promotion logic is going to be on the intersection of the exact and loop
  // graph. We will generate a map on the entries of this graph so it's
  // important to not modify this graph moving forward, as that would invalidate
  // the map.
  //
  // iel stands for Intersection of the Exact and Loop graphs.
  std::unordered_map<IdGroup, IterDomain*> iel_promotion_map;

  // This should probably work just on terminating inputs, as we shouldn't be
  // able to modify a broadcast domain between root and rfactor which would be
  // required to resolve a non input broadcast domain. But for now leaving it as
  // traversal on all broadcast groups.
  //
  // TODO-NM: The ordering appears to be non-deterministic

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
  for (const IdGroup& iel_group :
       intersection_exact_loop_graph.disjointIdSets().disjointSets()) {
    NVF_ERROR(!iel_group->empty());

    if (!iel_group->front()->isBroadcast()) {
      continue;
    }

    // Collect all the exact groups of the resolutions of the broadcast id's
    IdGroups resolved_exact_groups;
    for (IterDomain* bcast_id : *iel_group) {
      if (auto p2c_root_broadcast_resolution_map_it =
              info.p2c_root_broadcast_resolution_map.find(bcast_id);
          p2c_root_broadcast_resolution_map_it !=
          info.p2c_root_broadcast_resolution_map.end()) {
        resolved_exact_groups.pushBack(
            idGraph(IdMappingMode::EXACT)
                .toGroups(p2c_root_broadcast_resolution_map_it->second));
      }
    }

    // Collect all the exact groups in the loop set containing this iel_group
    auto loop_group = idGraph(IdMappingMode::LOOP).toGroup(iel_group->front());
    auto loop_covered_exact_groups =
        idGraph(IdMappingMode::EXACT).toGroups(*loop_group);

    // The intersection of the exact groups that the broadcast domains can be
    // broadcasted to, and those that exist within the same loop groop are is
    // the promotion needed for this iel_group.
    IdGroups loop_exact_resolved_intersection =
        resolved_exact_groups.intersect(loop_covered_exact_groups);

    if (loop_exact_resolved_intersection.empty()) {
      // No resolution
      continue;
    }

    if (loop_exact_resolved_intersection.size() > 1) {
      std::stringstream err_msg;

      err_msg
          << "Invalid multiple broadcast resolution within shared loops detected, group:\n  "
          << iel_group->toString() << "\nIs being broadcasted to:";

      for (const IdGroup& entry : loop_exact_resolved_intersection) {
        err_msg << "\n  " << entry->toString();
      }
      NVF_ERROR(false, err_msg.str());
    }

    // loop_exact_resolved_intersection.size() must be 1 at this point
    IdGroup exact_resolution_group = loop_exact_resolved_intersection.front();

    VectorOfUniqueEntries<IterDomain*> resolved_ids =
        exact_resolution_group->intersect(*loop_group);
    auto promoted_iel_groups =
        intersection_exact_loop_graph.toGroups(resolved_ids);

    if (promoted_iel_groups.empty()) {
      continue;
    }

    if (promoted_iel_groups.size() > 1) {
      std::stringstream err_msg;

      err_msg
          << "Invalid multiple broadcast resolution within shared loops detected, group:\n  "
          << iel_group->toString() << "\nIs being broadcasted to:";

      for (const IdGroup& entry : promoted_iel_groups) {
        err_msg << "\n  " << entry->toString();
      }
      NVF_ERROR(false, err_msg.str());
    }

    iel_promotion_map[iel_group] = promoted_iel_groups.front()->front();
  }

  // Propagate promotion mappings from root domains to derived domains
  // by traversing IEL exprs. For each expr, if an input is promoted,
  // the output needs to be promoted too. If there's already a domain
  // that the output domain should be promoted to, create a mapping to it from
  // the promoted output domain. If not, a new domain is created by
  // replaying the expr with the promoted inputs.

  // In order to make
  // this traversal work, the traversal order must be toplogically
  // sorted.
  IdGraphStmtSort iel_stmt_sort(intersection_exact_loop_graph);

  for (const ExprGroup& iel_expr : iel_stmt_sort.exprs()) {
    NVF_ERROR(!iel_expr->empty());
    std::vector<IdGroup> input_groups =
        intersection_exact_loop_graph.inputGroups(iel_expr);

    // Check if any inputs need promotion indicating this expr group needs to
    // be replayed with promoted inputs
    std::vector<IterDomain*> promoted_inputs;
    bool an_input_was_promoted = false;

    for (const IdGroup& inp : input_groups) {
      auto inp_promo_it = iel_promotion_map.find(inp);
      if (inp_promo_it == iel_promotion_map.end()) {
        promoted_inputs.push_back(inp->front());
      } else {
        promoted_inputs.push_back(inp_promo_it->second);
        an_input_was_promoted = true;
      }
    }

    if (!an_input_was_promoted) {
      // No inputs need promotion so just continue
      continue;
    }

    IdGroups promoted_input_groups;
    for (auto inp_id : promoted_inputs) {
      if (intersection_exact_loop_graph.hasGroup(inp_id)) {
        promoted_input_groups.pushBack(
            intersection_exact_loop_graph.toGroup(inp_id));
      }
    }

    // Before replaying, check if there's already an expression like this, if so
    // use that for promotion. We would need the iel entries for non-promoted
    // inputs to match exactly to reuse the expression.
    //
    // Unfortunately this doesn't actually seem to save any replays because
    // we're not adding the replayed expression to the iel graph since we're
    // traversing the iel graph.
    //
    // TODO: Can we reduce the number of new expressions generated
    // here?
    //
    // TODO-NM: This won't work for any single-input expr, e.g.,
    // split, as there's no other non-promoted input. Can't we just
    // look at the use expr of the promoted IDGroup?
    //
    // TODO-NM: Why can't we just also use the promoted IDs and their
    // uses? E.g., test Indexing5, t3 has a merge of iS11 and bS7,
    // both of them are promoted to iS17 and iS45, respectively. Since
    // there's no promoted input, there would be no reuse, but it
    // seems perfectly fine to reuse the merge of iS17 and iS45.

    ExprGroups non_promoted_input_uses;
    for (const IdGroup& iel_group :
         promoted_input_groups.intersect(input_groups)) {
      non_promoted_input_uses.pushBack(
          intersection_exact_loop_graph.getUniqueUses(iel_group));
    }

    Expr* replay = nullptr;

    // Look for exprs that have inputs that are mapped in the IEL
    // graph with the (promoted) inputs of iel_expr. If found, no need
    // to create a new expr to produce promoted outputs
    for (const ExprGroup& iel_use_group : non_promoted_input_uses) {
      // No need to check itself
      if (iel_expr == iel_use_group) {
        continue;
      }
      if (IdGraph::transformAtributesMatch(
              iel_expr->front(), iel_use_group->front())) {
        auto use_inps =
            ir_utils::filterByType<IterDomain>(iel_use_group->front()->inputs())
                .vector();
        bool inps_match = true;
        for (auto inp_i : c10::irange(use_inps.size())) {
          inps_match = inps_match &&
              intersection_exact_loop_graph.disjointIdSets().strictAreMapped(
                  use_inps[inp_i], promoted_inputs[inp_i]);
        }
        if (inps_match) {
          replay = iel_use_group->front();
          break;
        }
      }
    }

    bool replayed = replay == nullptr;
    if (replay == nullptr) {
      replay = addReplayAs(promoted_inputs, iel_expr->front());
    }

    std::vector<IdGroup> out_groups =
        intersection_exact_loop_graph.outputGroups(iel_expr);

    // Mark outputs as having a promoted iter domain
    auto replay_out_ids =
        ir_utils::filterByType<IterDomain>(replay->outputs()).vector();
    auto ref_out_ids =
        ir_utils::filterByType<IterDomain>(iel_expr->front()->outputs())
            .vector();

    NVF_ERROR(replay_out_ids.size() == out_groups.size());

    for (auto i : c10::irange(replay_out_ids.size())) {
      iel_promotion_map[out_groups[i]] = replay_out_ids[i];
      // Explicitly map loop map since expr propagation doesn't happen
      if (replayed) {
        idGraph(IdMappingMode::LOOP).mapIds(replay_out_ids[i], ref_out_ids[i]);
      }
    }
  }

  std::stringstream ss;
  ss << "Inline promotion map\n";
  for (const auto& [iel_group, promoted_id] : iel_promotion_map) {
    ss << "\t" << nvfuser::toString(iel_group) << " -> " << promoted_id->name()
       << std::endl;
  }
  VERBOSE() << ss.str();

  return iel_promotion_map;
}

namespace {

std::unordered_map<IdGroup, IterDomain*> updateMap(
    const std::unordered_map<IdGroup, IterDomain*>& stale_map,
    IdGraph& new_graph) {
  std::unordered_map<IdGroup, IterDomain*> new_map;

  for (const auto& [stale_key, mapped_id] : stale_map) {
    const IdGroups& new_groups = new_graph.toGroups(*stale_key);
    NVF_ERROR(
        new_groups.size() == 1,
        "\nUpdate map assumes that new graph is equivalent to old graph plus extra mappings.\n",
        "i.e. all mappings in new_graph should exist in the graph stale_map was produced on.\n",
        "old:",
        nvfuser::toString(stale_key),
        "new: ",
        nvfuser::toString(new_groups));
    new_map[new_groups.front()] = mapped_id;
  }
  return new_map;
}

// Returns for each IdGroup in provided IdGraph what the input IdGroups are
// traversing on definitions. Ignoring broadcast IdGroups and resetting inputs
// at RFactor IdGroups.
std::unordered_map<IdGroup, IdGroups> computeCoveredGroups(
    const IdGraph& exact_graph,
    const std::unordered_set<IterDomain*>& view_rfactor_ids) {
  // Map from an exact iter domain group, to all the exact iter domain groups it
  // covers
  std::unordered_map<IdGroup, IdGroups> covered_ids;

  for (const IdGroup& id_group : exact_graph.disjointIdSets().disjointSets()) {
    // Initialize inputs
    if (exact_graph.getUniqueDefinitions(id_group).empty()) {
      covered_ids[id_group] = {id_group};
    }

    // Initialize rfactor groups
    if (std::any_of(id_group->begin(), id_group->end(), [&](IterDomain* id) {
          return view_rfactor_ids.find(id) != view_rfactor_ids.end();
        })) {
      covered_ids[id_group] = {id_group};
    }

    // Initialize broadcast groups to empty since broadcast domains
    // don't matter for indexing
    if (std::any_of(id_group->begin(), id_group->end(), [&](IterDomain* id) {
          return id->isBroadcast();
        })) {
      covered_ids[id_group] = {};
    }
  }

  IdGraphStmtSort exact_stmt_sort(exact_graph);

  for (const ExprGroup& exact_expr : exact_stmt_sort.exprs()) {
    std::vector<IdGroup> input_groups = exact_graph.inputGroups(exact_expr);

    IdGroups covered;
    for (const IdGroup& inp_group : input_groups) {
      covered.pushBack(covered_ids.at(inp_group));
    }

    for (const IdGroup& output_group : exact_graph.outputGroups(exact_expr)) {
      // Don't overwrite initialized cases due to rfactor markings.
      if (covered_ids.find(output_group) == covered_ids.end()) {
        covered_ids[output_group] = covered;
      }
    }
  }

  return covered_ids;
}
}; // namespace

std::unordered_map<IdGroup, IterDomain*> IterDomainGraphs::
    buildLoopPromotionMap(
        const std::vector<Expr*>& exprs,
        StatefulLoweringInfo& info,
        const std::unordered_map<IdGroup, IterDomain*>& stale_promotion_map) {
  // Non-ca domains may also need to be promoted if parent domains are
  // promoted.

  // Need to use the intersection of exact and loop map again, it needs to be
  // recomputed.
  auto intersection_exact_loop_graph = buildIntersection(
      idGraph(IdMappingMode::EXACT), idGraph(IdMappingMode::LOOP), false);

  // Update the promotion map
  auto iel_promotion_map =
      updateMap(stale_promotion_map, intersection_exact_loop_graph);

  // Map from an exact iter domain group, to all the exact iter domain groups it
  // covers; needs to be recomputed.
  std::unordered_map<IdGroup, IdGroups> exact_covered_ids =
      computeCoveredGroups(idGraph(IdMappingMode::EXACT), view_rfactor_ids_);

  // Grab terminal iter domain in the loop groups.
  VectorOfUniqueEntries<IterDomain*> terminal_loop_ids =
      computeTerminalLoopIds(info);

  // Loop promotion map is to prepare for IterDomain replays to resolve
  // non-inlined loop groups. Since these replays will modify the loop map as
  // we're iterating over the loop map, operate on a copy of the loop map, not
  // the original one.
  auto loop_graph_copy = idGraph(IdMappingMode::LOOP);

  // Build a map from loop iter domain group to a promoted iter domain (doesn't
  // have to be in the loop group) that covers all the exact groups
  // representative of the resolved transformations within the loop group. Only
  // the inlined loop groups will be covered here.
  std::unordered_map<IdGroup, IterDomain*> loop_graph_copy_promotion_map;

  // TODO: I'm uncertain if we can simply use the iel_promotion_map. Once this
  // system is in use we should test not recomputing the "concrete ids".

  for (const IdGroup& loop_group :
       loop_graph_copy.disjointIdSets().disjointSets()) {
    if (loop_group->size() == 1) {
      loop_graph_copy_promotion_map[loop_group] = loop_group->front();
      continue;
    }

    // Grab all the (potentially promoted) terminal iter domains in this group.
    // Save the exact group and the iter domain in this vector.
    std::vector<std::pair<IdGroup, IterDomain*>> exact_promoted_terminal_ids;
    for (auto loop_id : *loop_group) {
      // If not a terminal id in the group skip
      if (!terminal_loop_ids.has(loop_id)) {
        continue;
      }

      // Grab the iel entry
      const IdGroup& iel_group = intersection_exact_loop_graph.toGroup(loop_id);

      auto iel_promo_it = iel_promotion_map.find(iel_group);
      if (iel_promo_it == iel_promotion_map.end()) {
        // If this terminal ID doesn't have a promotion associated with it, save
        // the terminal ID.
        exact_promoted_terminal_ids.emplace_back(
            idGraph(IdMappingMode::EXACT).toGroup(loop_id), loop_id);
      } else {
        // If this terminal ID has a promotion, grab the promoted ID.
        exact_promoted_terminal_ids.emplace_back(
            idGraph(IdMappingMode::EXACT).toGroup(iel_promo_it->second),
            iel_promo_it->second);
      }
    }

    // All the exact groups of the iter domains in the loop group
    IdGroups exact_groups = idGraph(IdMappingMode::EXACT).toGroups(*loop_group);

    // All exact groups covered by all iter domains in this loop group
    IdGroups loop_group_covered_ids;
    for (const IdGroup& exact_group : exact_groups) {
      auto covered_it = exact_covered_ids.find(exact_group);
      NVF_ERROR(covered_it != exact_covered_ids.end());
      loop_group_covered_ids.pushBack(covered_it->second);
    }

    IterDomain* loop_promotion_id = nullptr;

    // Check if any of the candidate Iter Domains we collected cover all the
    // exact groups of loop_group_covered_ids. If so, that's the correct
    // promoted iter domain of this group.
    for (const auto& entry : exact_promoted_terminal_ids) {
      const IdGroup& terminal_id_group = entry.first;
      IterDomain* terminal_id = entry.second;
      auto covered_it = exact_covered_ids.find(terminal_id_group);
      NVF_ERROR(covered_it != exact_covered_ids.end());
      if (loop_group_covered_ids.subtract(covered_it->second).empty()) {
        loop_promotion_id = terminal_id;
        break;
      }
    }

    if (loop_promotion_id == nullptr) {
      std::stringstream err_msg;
      err_msg
          << "\n ERROR Loop promotion map build. Could not find promotion for loop group:\n  ";
      err_msg << nvfuser::toString(loop_group, 0, true);
      err_msg << "\nnone of the terminal iter domains of this group:\n  ";
      for (const auto& entry : exact_promoted_terminal_ids) {
        const IdGroup& terminal_id_group = entry.first;
        const IdGroups& covered_id_groups =
            exact_covered_ids.at(terminal_id_group);
        err_msg << "  " << nvfuser::toString(terminal_id_group, 0, true)
                << " -(covers)-> " << nvfuser::toString(covered_id_groups)
                << std::endl;
      }
      err_msg << "iter domains in this group cover all id groups:\n";
      for (const IdGroup& covered_group : loop_group_covered_ids) {
        err_msg << "  " << nvfuser::toString(covered_group, 0, true);
      }
      // NVF_ERROR(false, err_msg.str());
    } else {
      loop_graph_copy_promotion_map[loop_group] = loop_promotion_id;
    }
  }

  // At this point, most of loop groups should have correct promoted
  // IDs. However, non-inlined loop groups may miss promotion that
  // should be propagated from parent ID groups, e.g., iS50 of T2 in
  // Indexing19. Its parent ID loop group is promoted, but the loop
  // group of iS50 is not found yet.

  // Reset the promotion map for the second pass.
  // TODO: Unclear if we could simply update the iel_promotion_map from
  // buildInlinePromotions, instead of manually building it.
  iel_promotion_map.clear();

  // Need to run a replay for the loop groups that are dependent on inlined loop
  // groups, but themselves are not inlined loop groups.

  for (const ExprGroup& iel_expr :
       IdGraphStmtSort(intersection_exact_loop_graph).exprs()) {
    NVF_ERROR(!iel_expr->empty());

    std::vector<IdGroup> iel_inp_groups =
        intersection_exact_loop_graph.inputGroups(iel_expr);

    std::vector<IdGroup> iel_out_groups =
        intersection_exact_loop_graph.outputGroups(iel_expr);

    // When replaying the transformations we can't blindly apply loop promotion
    // to all iter domains within a loop group as it would replay the
    // transformations within that loop group on the promoted id of that loop
    // group.
    //
    // i.e. if we have the inlined domains from:
    // T2[i0*i1] pa(1) = T0[i0*b1]ca(1) + T1[i0*i1]ca(1)
    // The inlined loop group would be:
    //
    // i0, i1, b1, i0*i1, b0*i1
    // Then if we replayed the iel transformations they would be:
    // merge(i0, i1)
    // merge(i0, b1)
    //
    // So if we replayed them with loop promotion, then i0, i1, b1 would be
    // promoted to i0*i1, and the merges would be replayed.
    //
    // Therefore only promote i0*b1 to i0*i1, or i0*i1 to i0*i1 (i.e. don't
    // promote an input to any transformation within the loop group).
    //
    // So if we have an iel_expr make sure it's inputs and outputs are not in
    // the same loop group.

    IdGroups inp_loop_groups;
    for (const IdGroup& iel_inp_group : iel_inp_groups) {
      inp_loop_groups.pushBack(loop_graph_copy.toGroup(iel_inp_group->front()));
    }

    IdGroups out_loop_groups;
    for (const IdGroup& iel_out_group : iel_out_groups) {
      out_loop_groups.pushBack(loop_graph_copy.toGroup(iel_out_group->front()));
    }

    // The inputs should be promoted based on the loop promotion map.
    bool loop_promote_inputs =
        !inp_loop_groups.subtract(out_loop_groups).empty();

    std::vector<IterDomain*> promoted_inputs;

    bool an_input_was_promoted = false;

    // Promote inputs for replay
    for (const IdGroup& iel_inp_group : iel_inp_groups) {
      // Promote loops based on the loop promotion map. If the loop promotion
      // map should be used and has an entry we should use that promotion. This
      // happen when an iel expression is across a loop group boundary.
      // Signifying and capturing instances when we traverse across an inlined
      // loop group to a non-inlined loop group boundary (think of the iel graph
      // projected onto the loop graph).
      const IdGroup& loop_copy_group =
          loop_graph_copy.toGroup(iel_inp_group->front());
      auto inp_loop_promo_it =
          loop_graph_copy_promotion_map.find(loop_copy_group);
      if (loop_promote_inputs &&
          inp_loop_promo_it != loop_graph_copy_promotion_map.end()) {
        promoted_inputs.push_back(inp_loop_promo_it->second);
        an_input_was_promoted = true;
      } else {
        // We still could require an input promotion. We could be traversing
        // across non-inlined groups. Meaning we have inputs that were promoted
        // in an inlined loop group traversing through the non-inlined portions
        // of the iel graph.
        auto inp_promo_it = iel_promotion_map.find(iel_inp_group);
        if (inp_promo_it == iel_promotion_map.end()) {
          promoted_inputs.push_back(iel_inp_group->front());
        } else {
          promoted_inputs.push_back(inp_promo_it->second);
          an_input_was_promoted = true;
        }
      }
    }

    if (!an_input_was_promoted) {
      continue;
    }

    Expr* replay = nullptr;

    // Before replaying, check if there's already an expression like this, if so
    // use that for promotion. We're still only looking for representative iter
    // domains, so if there's already an expression that would produce something
    // representative (matching in the IEL graph) of what the new inputs would
    // generate, just promote to that expressions outputs, don't bother
    // generating a new one.
    //
    // Check all uses of the IEL map the inputs are in, and look for one that
    // would match. Grab all uses of the promoted inputs' groups in the IEL
    // map. Note that promotion should be to loop-mapped domains, so
    // the IEL graph is used rather than the exact graph
    std::vector<IdGroup> promoted_input_groups;

    ExprGroups promoted_input_uses;
    for (auto inp_id : promoted_inputs) {
      // inp_id may have been just replayed, in which case it should
      // not exist in the IEL graph. It should be just ignored as it
      // should not have any use yet.
      if (!intersection_exact_loop_graph.hasGroup(inp_id)) {
        continue;
      }
      const auto& inp_exact_group =
          intersection_exact_loop_graph.toGroup(inp_id);
      promoted_input_groups.push_back(inp_exact_group);
      promoted_input_uses.pushBack(
          intersection_exact_loop_graph.getUniqueUses(inp_exact_group));
    }

    // Check every use to see if it matches
    for (const ExprGroup& iel_use_group : promoted_input_uses) {
      NVF_ERROR(!iel_use_group->empty());
      // Check if all the attributes (including type) of the transform match
      if (!IdGraph::transformAtributesMatch(
              iel_expr->front(), iel_use_group->front())) {
        continue;
      }
      // Check if inputs all match
      if (promoted_input_groups !=
          intersection_exact_loop_graph.inputGroups(iel_use_group)) {
        continue;
      }
      // Input mapping doesn't always mean expr and output
      // mappings. Make sure the exprs are mapped, which automatically
      // means the outputs are mapped in the case of the LOOP map
      if (!idGraph(IdMappingMode::LOOP)
               .disjointExprSets()
               .permissiveAreMapped(
                   iel_expr->front(), iel_use_group->front())) {
        continue;
      }
      // This is just an extra sanity check. Make sure all exprs in
      // the use group are mapped
      NVF_ERROR(
          std::all_of(
              iel_use_group->vector().begin(),
              iel_use_group->vector().end(),
              [&](Expr* iel_use) {
                return idGraph(IdMappingMode::LOOP)
                    .disjointExprSets()
                    .permissiveAreMapped(iel_expr->front(), iel_use);
              }),
          "Not all mapped: ",
          nvfuser::toString(iel_expr),
          "\n",
          nvfuser::toString(iel_use_group));

      replay = iel_use_group->front();
      break;
    }

    bool replayed = replay == nullptr;
    if (replay == nullptr) {
      replay = addReplayAs(promoted_inputs, iel_expr->front());
    }

    auto output_groups = intersection_exact_loop_graph.outputGroups(iel_expr);

    // Match or replay, mark promotion for output groups.
    auto replay_out_ids =
        ir_utils::filterByType<IterDomain>(replay->outputs()).vector();
    auto ref_out_ids =
        ir_utils::filterByType<IterDomain>(iel_expr->front()->outputs())
            .vector();

    NVF_ERROR(replay_out_ids.size() == output_groups.size());

    for (auto i : c10::irange(replay_out_ids.size())) {
      if (!idGraph(IdMappingMode::EXACT)
               .disjointIdSets()
               .strictAreMapped(replay_out_ids[i], output_groups[i]->front())) {
        // Promote if necessary, if the output is already in the same exact map
        // it doesn't need a promotion.
        iel_promotion_map[output_groups[i]] = replay_out_ids[i];
        // Explicitly map loop map since expr propagation doesn't happen on the
        // loop map and the replayed outputs are brand new so we can map them
        // without joining disjoint loop groups (other than the new loop groups
        // the outputs of the replay are in)
        if (replayed) {
          // If we built new iter domains because we generated a new expression,
          // link the outputs in the loop graph.
          idGraph(IdMappingMode::LOOP)
              .mapIds(replay_out_ids[i], ref_out_ids[i]);
        }
      }
    }
  }

  // Update the coverage map
  exact_covered_ids =
      computeCoveredGroups(idGraph(IdMappingMode::EXACT), view_rfactor_ids_);

  // Set up the loop promotion map of loop groups to promotion
  // IDs. Note that the IEL promotion map is still incomplete in the
  // sense that:
  //
  // - Not all loop graphs have promotions set at this point.
  // - Multiple domains that are loop-mapped may have different
  //   promotions, one of which should cover the rest.
  //
  // Fill the gap, here we traverse the loop graph and for each loop
  // group we examine each IEL group. If an IEL group has a promotion,
  // we consider it as a candidate of the promotion of this loop
  // group. If not, we include a domain of the IEL group as a
  // candidate too. We also look at the inline promotion map since
  // that may also contain the promotion the loop should be associated
  // with. Once all candidates are obtained, we pick one that covers
  // all the exact domains (cf. concrete domains in ComputeAtMap)
  for (const IdGroup& loop_group :
       loop_graph_copy.disjointIdSets().disjointSets()) {
    IdGroups iel_groups = intersection_exact_loop_graph.toGroups(*loop_group);
    // All exact groups covered by all iter domains in this loop group
    IdGroups loop_group_covered_ids;
    for (const IdGroup& iel_group : iel_groups) {
      auto exact_group =
          idGraph(IdMappingMode::EXACT).toGroup(iel_group->front());
      auto covered_it = exact_covered_ids.find(exact_group);
      NVF_ERROR(
          covered_it != exact_covered_ids.end(),
          "Exact covered id not found for ",
          nvfuser::toString(exact_group));
      loop_group_covered_ids.pushBack(covered_it->second);
    }

    VectorOfUniqueEntries<IterDomain*> representative_id_candidates;

    for (const IdGroup& iel_group : iel_groups) {
      if (auto iel_promotion_map_it = iel_promotion_map.find(iel_group);
          iel_promotion_map_it != iel_promotion_map.end()) {
        IterDomain* iel_promotion_id = iel_promotion_map_it->second;
        representative_id_candidates.pushBack(iel_promotion_id);
      } else {
        representative_id_candidates.pushBack(iel_group->front());
      }
    }

    if (auto loop_graph_copy_promotion_map_it =
            loop_graph_copy_promotion_map.find(
                loop_graph_copy.toGroup(loop_group->vector().at(0)));
        loop_graph_copy_promotion_map_it !=
        loop_graph_copy_promotion_map.end()) {
      VERBOSE() << "Found in loop promotion: " << nvfuser::toString(loop_group)
                << std::endl;
      representative_id_candidates.pushBack(
          loop_graph_copy_promotion_map_it->second);
    }

    VERBOSE() << "Loop promotion candidates: " << std::endl;

    // All candidates gathered
    for (IterDomain* candidate_id : representative_id_candidates) {
      auto covered_it = exact_covered_ids.find(
          idGraph(IdMappingMode::EXACT).toGroup(candidate_id));
      NVF_ERROR(covered_it != exact_covered_ids.end());
      if (loop_group_covered_ids.subtract(covered_it->second).empty()) {
        // Found
        VERBOSE() << "Representative found: " << candidate_id->toString()
                  << std::endl;
        const IdGroup& current_loop_group =
            idGraph(IdMappingMode::LOOP).toGroup(loop_group->front());
        loop_promotion_map_.emplace(current_loop_group, candidate_id);
        break;
      }
    }
  }

  // Sanity check of the loop promotion map
  for (const IdGroup& loop_group :
       idGraph(IdMappingMode::LOOP).disjointIdSets().disjointSets()) {
    // Non-leaf loop groups are not guaranteed to have valid
    // promotions. See for example FusionRepro1713, where root domains
    // are all grouped together but there's no valid promotion.
    if (idGraph(IdMappingMode::LOOP).hasUses(loop_group)) {
      continue;
    }
    auto promotion_it = loop_promotion_map_.find(loop_group);
    NVF_ERROR(
        promotion_it != loop_promotion_map_.end(),
        "Loop promotion not found for ",
        nvfuser::toString(loop_group));
    IterDomain* promotion = promotion_it->second;
    // Make sure the promotion domain is also loop-mapped
    NVF_ERROR(
        loop_group->has(promotion),
        "Loop promotion not loop-mapped. Loop group: ",
        nvfuser::toString(loop_group),
        ". Promotion domain: ",
        promotion->name());
  }

  VERBOSE() << "Loop promotion map:" << std::endl;
  for (const auto& [iel_group, id] : iel_promotion_map) {
    VERBOSE() << nvfuser::toString(iel_group) << " -> " << id->name()
              << std::endl;
  }

  return iel_promotion_map;
}

std::unordered_map<IterDomain*, IterDomain*> IterDomainGraphs::buildIndexGraph(
    const std::vector<Expr*>& exprs,
    const std::vector<TensorView*>& all_tvs,
    StatefulLoweringInfo& info,
    std::unordered_map<IdGroup, IterDomain*> stale_promotion_map) {
  NVF_ERROR(false, "Not implemented yet.");
}

} // namespace nvfuser
