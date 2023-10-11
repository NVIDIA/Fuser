// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <test/utils.h>

#include <expr_evaluator.h>
#include <fusion.h>
#include <id_model/id_graph.h>
#include <id_model/to_string.h>
#include <inlining.h>
#include <ops/all_ops.h>
#include <transform_iter.h>

namespace nvfuser {

class IdGraphTest : public NVFuserTest {};

namespace {

auto buildIterDomainDefinitionsAndUses(
    const std::vector<TensorView*>& all_tvs) {
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<Expr*>> id_uses;
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<Expr*>> id_definitions;

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
          // view_rfactor_ids_.emplace(id);
        }
      }

      if (id_definitions.find(id) == id_definitions.end()) {
        id_definitions[id] = {};
      }

      if (id_uses.find(id) == id_uses.end()) {
        id_uses[id] = {};
      }

      auto def = id->definition();

      if (def == nullptr || root_domain_ids.has(id)) {
        continue;
      }

      if (id_definitions.find(id) == id_definitions.end()) {
        id_definitions[id] = {};
      }
      id_definitions.at(id).pushBack(def);

      auto inp_ids = ir_utils::filterByType<IterDomain>(def->inputs());
      for (auto inp_id : inp_ids) {
        if (id_uses.find(inp_id) == id_uses.end()) {
          id_uses[inp_id] = {};
        }
        id_uses.at(inp_id).pushBack(def);
      }
    }
  }

  return std::make_pair(id_uses, id_definitions);
}

IdGraph initializeIdGraph(
    bool propagate_exprs,
    const std::unordered_map<IterDomain*, VectorOfUniqueEntries<Expr*>>&
        id_uses,
    const std::unordered_map<IterDomain*, VectorOfUniqueEntries<Expr*>>&
        id_definitions) {
  IdGraph id_graph(propagate_exprs);

  for (const auto& definition_entry : id_definitions) {
    auto id = definition_entry.first;
    auto defs = definition_entry.second;
    auto uses_it = id_uses.find(id);
    NVF_ERROR(
        uses_it != id_uses.end(),
        "Failed to initialize id: ",
        id->toString(),
        " as it's missing a definition entry.");
    id_graph.initializeId(id, defs, uses_it->second);
  }

  return id_graph;
}

void buildExactMap(const std::vector<Expr*>& exprs, IdGraph& id_graph) {
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
        id_graph.mapIds(o_id, c_id);
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
        std::cerr << "Map: " << c_id->toString() << ", " << p_id->toString()
                  << std::endl;
        id_graph.mapIds(c_id, p_id);
      }
    }

    id_graph.mapThroughLoopSwizzles();
  }
}

void buildPermissiveMap(const std::vector<Expr*>& exprs, IdGraph& id_graph) {
  buildExactMap(exprs, id_graph);

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
#if 0
        std::cerr << "Permissive map 1: " << entry.first->toString()
                  << ", " << entry.second->toString()
                  << std::endl;
#endif
        id_graph.mapIds(entry.first, entry.second);
      }
#if 0
      // TODO: Should this just get rolled up in the forwarding map now?
      // TODO: Why should IDs be mapped to their compliments? Is this right?
      for (auto entry : permissive_forwarding.producer_compliment_map) {
        for (auto entry_2 : entry.second) {
          std::cerr << "Permissive map 2: " << entry.first->toString()
                    << ", " << entry_2->toString()
                    << std::endl;
          id_graph.mapIds(entry.first, entry_2);
        }
      }
#endif

      for (auto entry : permissive_forwarding.consumer_forwarding_map) {
#if 0
        std::cerr << "Permissive map 3: " << entry.first->toString()
                  << ", " << entry.second->toString()
                  << std::endl;
#endif
        id_graph.mapIds(entry.first, entry.second);
      }

#if 0
      // TODO: Should this just get rolled up in the forwarding map now?
      // TODO: Why should IDs be mapped to their compliments? Is this right?
      for (auto entry : permissive_forwarding.consumer_compliment_map) {
        for (auto entry_2 : entry.second) {
          std::cerr << "Permissive map 4: " << entry.first->toString()
                    << ", " << entry_2->toString()
                    << std::endl;
          id_graph.mapIds(entry.first, entry_2);
        }
      }
#endif
      auto permissive_c2p_root_map = PairwiseRootDomainMap(p_tv, c_tv);

      for (auto entry : permissive_c2p_root_map.mapConsumerToProducer()) {
#if 0
        std::cerr << "Permissive map 5: " << entry.first->toString()
                  << ", " << entry.second->toString()
                  << std::endl;
#endif
        id_graph.mapIds(entry.first, entry.second);
      }
    }
  }
  id_graph.mapThroughLoopSwizzles();
}

// Partially copied from IterDomainGraphs::build for testing IdGraph only
IdGraph buildExactMap(Fusion* fusion) {
  FusionGuard fg(fusion);

  auto exprs = fusion->exprs();

  std::vector<Expr*> tv_exprs;

  std::copy_if(
      exprs.begin(), exprs.end(), std::back_inserter(tv_exprs), [](Expr* expr) {
        NVF_ERROR(expr != nullptr);
        return ir_utils::isTvOp(expr);
      });

  auto all_tvs = ir_utils::allTvsOfExprs(tv_exprs);

  std::unordered_set<TensorView*> all_added_tvs(all_tvs.begin(), all_tvs.end());
  for (auto additional_tv :
       ir_utils::filterByType<TensorView>(fusion->inputs())) {
    if (all_added_tvs.insert(additional_tv).second) {
      all_tvs.push_back(additional_tv);
    }
  }
  for (auto additional_tv :
       ir_utils::filterByType<TensorView>(fusion->outputs())) {
    if (all_added_tvs.insert(additional_tv).second) {
      all_tvs.push_back(additional_tv);
    }
  }

  if (all_tvs.empty()) {
    return IdGraph();
  }

  // Add uses and definitions to all iter domains.
  auto [id_uses, id_definitions] = buildIterDomainDefinitionsAndUses(all_tvs);

  auto id_graph = initializeIdGraph(true, id_uses, id_definitions);

  buildExactMap(tv_exprs, id_graph);

  return id_graph;
}

IdGraph buildPermissiveMap(Fusion* fusion) {
  FusionGuard fg(fusion);

  auto exprs = fusion->exprs();

  std::vector<Expr*> tv_exprs;

  std::copy_if(
      exprs.begin(), exprs.end(), std::back_inserter(tv_exprs), [](Expr* expr) {
        NVF_ERROR(expr != nullptr);
        return ir_utils::isTvOp(expr);
      });

  auto all_tvs = ir_utils::allTvsOfExprs(tv_exprs);

  std::unordered_set<TensorView*> all_added_tvs(all_tvs.begin(), all_tvs.end());
  for (auto additional_tv :
       ir_utils::filterByType<TensorView>(fusion->inputs())) {
    if (all_added_tvs.insert(additional_tv).second) {
      all_tvs.push_back(additional_tv);
    }
  }
  for (auto additional_tv :
       ir_utils::filterByType<TensorView>(fusion->outputs())) {
    if (all_added_tvs.insert(additional_tv).second) {
      all_tvs.push_back(additional_tv);
    }
  }

  if (all_tvs.empty()) {
    return IdGraph();
  }

  // Add uses and definitions to all iter domains.
  auto [id_uses, id_definitions] = buildIterDomainDefinitionsAndUses(all_tvs);

  auto id_graph = initializeIdGraph(true, id_uses, id_definitions);

  buildPermissiveMap(tv_exprs, id_graph);

  return id_graph;
}

} // namespace

// Test the exact map with a multi-promotion fusion pattern. Promotion
// should not matter as the exact map is concerned
TEST_F(IdGraphTest, MultiPromotionExactMap) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [y]
  auto tv0 = makeSymbolicTensor(1);
  // [w, x, y, z]
  auto tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // y
  auto tv2 = broadcast(tv0, {true, false});
  // w, y
  auto tv3 = broadcast(tv2, {false, false, true});
  // w, y, z
  auto tv4 = broadcast(tv3, {false, true, false, false});
  // w, x, y, z
  auto tv5 = add(tv4, tv1);

  fusion.addOutput(tv5);

  tv5->merge(1)->merge(1)->merge(0)->split(0, 11);

  TransformPropagator propagator(tv5);
  MaxRootDomainInfoSpanningTree(tv5).traverse(&propagator);

  inlineAllAt(tv5, 1);

  auto exact_map = buildExactMap(&fusion);

  // Make sure the non-root IDs should not be mapped at
  // all, except for tv1 and tv5, which should be mapped with each
  // other, so their non-root domain groups should have size 2.
  for (auto tv : {tv0, tv1, tv2, tv3, tv4, tv5}) {
    for (auto id : ir_utils::allIDsOf(tv)) {
      if (std::find(
              tv->getRootDomain().begin(), tv->getRootDomain().end(), id) !=
          tv->getRootDomain().end()) {
        continue;
      }

      size_t expected_size = 0;
      if (tv->name() == 1 || tv->name() == 5) {
        expected_size = 2;
      } else {
        expected_size = 1;
      }
      const auto& idg = exact_map.toGroup(id);
      ASSERT_EQ(idg->size(), expected_size)
          << "Unexpected IdGroup size: " << toString(idg)
          << ", tensor: " << tv->toString();
    }
  }
}

// Test the permissive map with a multi-promotion fusion pattern. Promotion
// should not matter as the exact map is concerned
TEST_F(IdGraphTest, MultiPromotionPermissiveMap) {
  GTEST_SKIP();
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [y]
  auto tv0 = makeSymbolicTensor(1);
  // [w, x, y, z]
  auto tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // y
  auto tv2 = broadcast(tv0, {true, false});
  // w, y
  auto tv3 = broadcast(tv2, {false, false, true});
  // w, y, z
  auto tv4 = broadcast(tv3, {false, true, false, false});
  // w, x, y, z
  auto tv5 = add(tv4, tv1);

  fusion.addOutput(tv5);

  tv5->merge(1)->merge(1)->merge(0)->split(0, 11);

  TransformPropagator propagator(tv5);
  MaxRootDomainInfoSpanningTree(tv5).traverse(&propagator);

  inlineAllAt(tv5, 1);

  auto map = buildPermissiveMap(&fusion);

  const auto& id_sets = map.disjointIdSets().disjointSets();

  ASSERT_EQ(id_sets.size(), 6) << "Unexpected number of disjoint sets";

  for (const auto& id_set : id_sets) {
    std::unordered_set<IterDomain*> ref_set;
    // w and x are merged in the current implementation. Is it the
    // right behavior?
    if (id_set->has(tv1->getRootDomain().at(0))) { // w
      if (true) {
        ref_set = std::unordered_set<IterDomain*>(
            {tv1->getRootDomain().at(0),
             tv2->getRootDomain().at(0),
             tv3->getRootDomain().at(0),
             tv4->getRootDomain().at(0),
             tv5->getRootDomain().at(0),
             tv1->getRootDomain().at(1),
             tv4->getRootDomain().at(1),
             tv5->getRootDomain().at(1)});
      } else {
        ref_set = std::unordered_set<IterDomain*>(
            {tv1->getRootDomain().at(0),
             tv2->getRootDomain().at(0),
             tv3->getRootDomain().at(0),
             tv4->getRootDomain().at(0),
             tv5->getRootDomain().at(0)});
      }
    } else if (id_set->has(tv1->getRootDomain().at(1))) { // x
      ref_set = std::unordered_set<IterDomain*>(
          {tv1->getRootDomain().at(1),
           tv4->getRootDomain().at(1),
           tv5->getRootDomain().at(1)});
    } else if (id_set->has(tv1->getRootDomain().at(2))) { // y
      ref_set = std::unordered_set<IterDomain*>(
          {tv0->getRootDomain().at(0),
           tv1->getRootDomain().at(2),
           tv2->getRootDomain().at(1),
           tv3->getRootDomain().at(1),
           tv4->getRootDomain().at(2),
           tv5->getRootDomain().at(2)});
      // Gather all IDs produced by merge
      for (auto tv : ir_utils::allTvs(&fusion)) {
        for (auto id : ir_utils::allIDsOf(tv)) {
          if (auto merge = dynamic_cast<Merge*>(id->definition())) {
            ref_set.insert(id);
          }
        }
      }
    } else if (id_set->has(tv1->getRootDomain().at(3))) { // z
      ref_set = std::unordered_set<IterDomain*>(
          {tv1->getRootDomain().at(3),
           tv3->getRootDomain().at(2),
           tv4->getRootDomain().at(3),
           tv5->getRootDomain().at(3)});
    } else if (id_set->has(tv1->axis(0))) { // leaf outer
      for (auto tv : ir_utils::allTvs(&fusion)) {
        ref_set.insert(tv->axis(0));
      }
    } else if (id_set->has(tv1->axis(1))) { // leaf inner
      for (auto tv : ir_utils::allTvs(&fusion)) {
        ref_set.insert(tv->axis(1));
      }
    } else {
      FAIL() << "Unexpected ID set";
    }

    const auto& actual_set = id_set->set();

    if (!ref_set.empty()) {
      ASSERT_EQ(ref_set, actual_set);
    }
  }
}

} // namespace nvfuser
