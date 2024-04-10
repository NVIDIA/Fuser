// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <fusion.h>
#include <id_model/id_model.h>
#include <id_model/to_string.h>
#include <inlining.h>
#include <ops/all_ops.h>
#include <transform_iter.h>
#include <val_graph_visitor.h>

namespace nvfuser {

using IdModelTest = NVFuserTest;

TEST_F(IdModelTest, DetectSelfMapping) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 2});
  fusion.addInput(tv0);
  auto tv1 = transpose(tv0, 0, 1);
  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  EXPECT_THAT(
      [&]() { IdModel id_model(&fusion, /*build_graphs=*/true); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("!hasSelfMapping")));
}

TEST_F(IdModelTest, PerTensorSelfMapping) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x0 = makeConcreteTensor({2, 2});
  fusion.addInput(x0);
  TensorView* x1 = makeConcreteTensor({2, 2});
  fusion.addInput(x1);

  TensorView* y0 = transpose(x0, 0, 1);
  y0 = add(x0, y0);
  fusion.addOutput(y0);

  TensorView* y1 = transpose(x1, 0, 1);
  fusion.addOutput(y1);

  IdModel id_model(&fusion, /*build_graphs=*/true, /*allow_self_mapping=*/true);
  const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);
  EXPECT_TRUE(hasSelfMapping(y0, exact_graph).has_value());
  EXPECT_FALSE(hasSelfMapping(y1, exact_graph).has_value());
}

namespace {

// Get n-th parent expr traversing through the first input of each
// parent
Expr* getParentExpr(Val* val, int n) {
  for (int i = 0; i < n - 1; ++i) {
    NVF_ERROR(val->definition() != nullptr);
    val = val->definition()->input(0);
  }
  NVF_ERROR(val->definition() != nullptr);
  return val->definition();
};

IterDomain* getParentId(IterDomain* id, int n) {
  for (int i = 0; i < n; ++i) {
    NVF_ERROR(id->definition() != nullptr);
    NVF_ERROR(id->definition()->input(0)->isA<IterDomain>());
    id = id->definition()->input(0)->as<IterDomain>();
  }
  NVF_ERROR(id != nullptr);
  return id;
};

// Get the n-th descendant by traversing a sibling
IterDomain* getChildId(IterDomain* id, int n, int sibling_idx = 0) {
  for (int i = 0; i < n; ++i) {
    NVF_ERROR(!id->uses().empty());
    NVF_ERROR(id->uses().front()->output(sibling_idx)->isA<IterDomain>());
    id = id->uses().front()->output(sibling_idx)->as<IterDomain>();
  }
  NVF_ERROR(id != nullptr);
  return id;
};

template <typename ValType>
ValType* getValByName(const std::vector<ValType*>& vals, StmtNameType name) {
  if (auto it = std::find_if(
          vals.begin(),
          vals.end(),
          [&](auto val) { return val->name() == name; });
      it != vals.end()) {
    return *it;
  } else {
    return nullptr;
  }
}

IterDomain* getChildIdByName(IterDomain* id, StmtNameType name) {
  auto named_val = getValByName(ir_utils::consumerValsOf(id), name);
  NVF_ERROR(named_val != nullptr, "Cannot find a child ID named ", name);
  NVF_ERROR(named_val->isA<IterDomain>());
  return named_val->as<IterDomain>();
};

// Helper class to test IdModel
class IdModelTester : public IdModel {
 public:
  // Do not automatically build the graphs
  IdModelTester(Fusion* fusion) : IdModel(fusion, /*build_graphs=*/false) {
    // Make sure the depedent graphs are already built
    maybeBuildGraph(IdMappingMode::EXACT);
    maybeBuildGraph(IdMappingMode::PERMISSIVE);

    // Gather broadcast resolution and inlining information
    const StatefulInliningInfo inlining_info = buildStatefulInliningInfo(
        tv_exprs_,
        idGraph(IdMappingMode::EXACT),
        idGraph(IdMappingMode::PERMISSIVE));

    initializeLoopGraph(inlining_info);

    iel_graph = buildIntersection(
        idGraph(IdMappingMode::EXACT), idGraph(IdMappingMode::LOOP), false);

    s1_root_resolution_map =
        buildInlineRootResolutionMap(iel_graph, inlining_info);

    s2_iel_promotion_map = s1_root_resolution_map;

    propagatePromotionsInIELGraph(iel_graph, s2_iel_promotion_map);

    const auto s3_original_loop_promotion_map = projectIELPromotionToLoopGraph(
        iel_graph,
        s2_iel_promotion_map,
        idGraph(IdMappingMode::LOOP),
        inlining_info);

    // Make a copy for validation as idGraph(IdMappingMode::LOOP) will
    // be updated in the later steps
    s3_loop_graph = idGraph(IdMappingMode::LOOP);
    s3_loop_promotion_map =
        updateValGroupIdMap(s3_original_loop_promotion_map, s3_loop_graph);

    // Note that s4_iel_promotion_map is an empty map at this
    // point. It'll be populated with the Step-3 map
    propagatePromotionsInIELGraph(
        iel_graph,
        s4_iel_promotion_map,
        idGraph(IdMappingMode::LOOP),
        s3_original_loop_promotion_map);
  }

  void print(std::ostream& os) const {
    os << "Step 1 results:\n";
    for (const auto& [g, id] : s1_root_resolution_map) {
      os << nvfuser::toString(g) << " -> " << id->toString() << std::endl;
    }
    os << "Step 2 results:\n";
    for (const auto& [g, id] : s2_iel_promotion_map) {
      os << nvfuser::toString(g) << " -> " << id->toString() << std::endl;
    }
    os << "Step 3 results:\n";
    for (const auto& [g, id] : s3_loop_promotion_map) {
      os << nvfuser::toString(g) << " -> " << id->toString() << std::endl;
    }
    os << "Step 4 results:\n";
    for (const auto& [g, id] : s4_iel_promotion_map) {
      os << nvfuser::toString(g) << " -> " << id->toString() << std::endl;
    }
  }

  ValGraph iel_graph;
  std::unordered_map<ValGroup, IterDomain*> s1_root_resolution_map;
  std::unordered_map<ValGroup, IterDomain*> s2_iel_promotion_map;
  ValGraph s3_loop_graph;
  std::unordered_map<ValGroup, IterDomain*> s3_loop_promotion_map;
  std::unordered_map<ValGroup, IterDomain*> s4_iel_promotion_map;
};

// Test if id is resolved to an ID that is exact mapped with
// ref_id. If ref_id  is nullptr, test if root_broadcast_id has no
// resolution.
void validateIELResolution(
    IterDomain* id,
    IterDomain* ref_id,
    const IdModelTester& tester,
    const std::unordered_map<ValGroup, IterDomain*>& iel_promotion_map) {
  const auto& iel_graph = tester.iel_graph;
  const auto& exact_graph = tester.idGraph(IdMappingMode::EXACT);
  const auto& loop_graph = tester.idGraph(IdMappingMode::LOOP);

  const auto& iel_group = iel_graph.toGroup(id);
  auto iel_promotion_map_it = iel_promotion_map.find(iel_group);
  if (ref_id != nullptr) {
    ASSERT_TRUE(iel_promotion_map_it != iel_promotion_map.end())
        << "IEL promotion not found for: " << nvfuser::toString(iel_group);
    ASSERT_FALSE(ref_id->isBroadcast());
    auto promotion_id = iel_promotion_map_it->second;
    ASSERT_TRUE(
        exact_graph.disjointValSets().strictAreMapped(promotion_id, ref_id))
        << "Unexpected promotion. "
        << "Expected: " << ref_id->toString()
        << ". Actual: " << promotion_id->toString();
    ASSERT_TRUE(loop_graph.disjointValSets().strictAreMapped(id, promotion_id))
        << "Promotion of " << id->toString()
        << " not mapped in the loop graph: " << promotion_id->toString();
  } else {
    ASSERT_TRUE(iel_promotion_map_it == iel_promotion_map.end())
        << "Promotion should not exist for: " << nvfuser::toString(iel_group)
        << ", but found: " << iel_promotion_map_it->second->toString();
  }
}

// Check if each domain gets promoted to a proper domain after the
// Step 2 IEL propagation. It is assumed that the proper promotion is
// the corresponding domain in the unique consumer tensor, which is
// the case with most of the test fusions.
void checkStep2Results(Fusion* fusion, const IdModelTester& tester) {
  const auto& iel_graph = tester.iel_graph;
  const auto& iel_promotion_map = tester.s2_iel_promotion_map;

  auto getPromotedDomain = [&](IterDomain* id) -> IterDomain* {
    if (auto it = iel_promotion_map.find(iel_graph.toGroup(id));
        it != iel_promotion_map.end()) {
      return it->second;
    } else {
      return nullptr;
    }
  };

  for (auto tv : ir_utils::allTvs(fusion)) {
    // If there's no broadcast or it isn't inlined, there's no
    // promotion
    if (std::none_of(
            tv->getRootDomain().begin(),
            tv->getRootDomain().end(),
            [](auto id) { return id->isBroadcast(); }) ||
        (tv->getComputeAtPosition() == 0 &&
         tv->getMaxProducerPosition() == 0)) {
      // Make sure there's no promotion of any of the IDs of this tensor
      for (auto id : ir_utils::allIDsOf(tv)) {
        auto promoted_id = getPromotedDomain(id);
        ASSERT_EQ(promoted_id, nullptr)
            << "Expected no mapping for " << id->toString()
            << " but found to be mapped to: " << promoted_id->toString();
      }
      continue;
    }

    auto consumers = ir_utils::consumerTvsOf(tv);
    ASSERT_EQ(consumers.size(), 1) << "Assumed to have one consumer";
    TensorView* c_tv = consumers.at(0);
    const auto p2c = BestEffortReplay::replayCasP(
                         c_tv, tv, -1, PairwiseRootDomainMap(tv, c_tv))
                         .getReplay();

    for (auto p_id : ir_utils::allIDsOf(tv)) {
      // Root domains are already done at Step 1
      if (std::find(
              tv->getRootDomain().begin(), tv->getRootDomain().end(), p_id) !=
          tv->getRootDomain().end()) {
        continue;
      }

      // If no broadcast is involved, nothing should be promoted
      auto p_id_dep_vals = DependencyCheck::getAllValsBetween(
          {tv->getRootDomain().begin(), tv->getRootDomain().end()}, {p_id});
      if (std::find_if(
              p_id_dep_vals.begin(), p_id_dep_vals.end(), [](Val* dep_id) {
                return dep_id->as<IterDomain>()->isBroadcast();
              }) == p_id_dep_vals.end()) {
        auto promoted_id = getPromotedDomain(p_id);
        ASSERT_EQ(promoted_id, nullptr)
            << "Expected no mapping for " << p_id->toString()
            << " but found to be mapped to: " << promoted_id->toString();
        continue;
      }

      // p_id should be promoted to c_id
      auto c_id = p2c.at(p_id);
      validateIELResolution(p_id, c_id, tester, iel_promotion_map);
    }
  }
}

// Validate the loop promotion map at Step 3. This validation ensures
// the promotion map is exactly the same as a given reference
// map. Since the valid promotion map may not be unique, the exact
// equality is not required, however, as long as everything is done
// deterministically, the resulting map should always be the
// same. The exact equality helps ensure the determinism as well.
void checkStep3Results(
    const IdModelTester& tester,
    const std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>&
        ref_promotion_map) {
  const auto& loop_graph = tester.s3_loop_graph;
  const auto& loop_promotion_map = tester.s3_loop_promotion_map;

  for (const auto& loop_group : loop_graph.disjointValSets().disjointSets()) {
    auto promotion_it = loop_promotion_map.find(loop_group);
    ASSERT_NE(promotion_it, loop_promotion_map.end())
        << "No promotion found for: " << nvfuser::toString(loop_group);
    IterDomain* promotion_id = promotion_it->second;

    auto ref_promotion_it = std::find_if(
        ref_promotion_map.begin(),
        ref_promotion_map.end(),
        [&](const auto& ref_promotion) {
          return ref_promotion.first == loop_group->set();
        });

    // Self promotion omitted in the reference
    if (ref_promotion_it == ref_promotion_map.end()) {
      ASSERT_EQ(loop_group->size(), 1);
      ASSERT_EQ(loop_group->front(), promotion_id)
          << "Expected promotion: " << loop_group->front()->toString()
          << ". Actual: " << promotion_id->toString();
      continue;
    }

    auto ref_promotion_id = ref_promotion_it->second;
    ASSERT_EQ(promotion_id, ref_promotion_id)
        << "Expected promotion: " << ref_promotion_id->toString()
        << ". Actual: " << promotion_id->toString();

    ASSERT_EQ(loop_graph.toGroup(promotion_id), loop_group)
        << "Loop group promoted to a non-mapped domain. Loop group: "
        << nvfuser::toString(loop_group)
        << ". Promotion: " << promotion_id->toString();
  }
}

void checkStep4Results(
    const IdModelTester& tester,
    const std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>&
        ref_promotion_map) {
  const auto& iel_promotion_map = tester.s4_iel_promotion_map;

  EXPECT_EQ(iel_promotion_map.size(), ref_promotion_map.size())
      << "Mismatched Step-4 result map. "
      << "Expected to have " << ref_promotion_map.size()
      << " mappings but found " << iel_promotion_map.size();

  for (const auto& ref_promotion_pair : ref_promotion_map) {
    const auto& ref_promotion_group = ref_promotion_pair.first;
    const auto& ref_promotion_id = ref_promotion_pair.second;

    auto iel_promotion_it = std::find_if(
        iel_promotion_map.begin(),
        iel_promotion_map.end(),
        [&](const auto& iel_promotion) {
          return iel_promotion.first->set() == ref_promotion_group;
        });

    auto iel_promotion_id = iel_promotion_it->second;
    EXPECT_EQ(ref_promotion_id, iel_promotion_id)
        << "Expected promotion: " << ref_promotion_id->toString()
        << ". Actual: " << iel_promotion_id->toString();
  }
}

// Create a fusion where we're missing a valid concrete id so the compute at map
// processing will fail. We need to be able to create the concrete ID not just
// look for one. It is not yet possible to lower this fusion as the
// current indexing cannot generate correct indices. Also used in
// FusionIndeixing19 as well as Example 2 in the design doc about Loop
// Promotion Analysis.
std::unique_ptr<Fusion> createFusionWithMultipleResolutionPaths() {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({7});
  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  auto tv2 = broadcast(tv1, {false, true});

  auto tv3 = makeConcreteTensor({7, 11});
  fusion.addInput(tv3);

  auto tv4 = add(tv3, tv2);
  auto tv5 = broadcast(tv4, {false, false, true});
  // tv4[7, 11, 1]

  auto tv6 = broadcast(tv1, {false, true});

  auto tv7 = makeConcreteTensor({7, 13});
  fusion.addInput(tv7);
  auto tv8 = add(tv7, tv6);
  auto tv9 = broadcast(tv8, {false, true, false});
  // tv9[7, 1, 13]

  auto tv10 = add(tv5, tv9);
  fusion.addOutput(tv10);

  // tv10[7, 11, 13]
  tv10->merge(0)->merge(0);
  // tv10[7*11*13]
  tv10->split(0, 5)->split(0, 3);
  // tv10[7*11*13//5//3, 3, 5]

  TransformPropagatorWithCheck propagator(tv10);
  MaxRootDomainInfoSpanningTree(tv10).traverse(&propagator);

  std::vector<TensorView*> tensors_to_inline{tv1, tv2, tv4, tv6, tv8};
  for (auto tensor : tensors_to_inline) {
    tensor->inlineAt(1);
  }

  return fusion_ptr;
}

// Check the results of ValGraphStmtSort. Only the ordering of
// ExprGroups is checked for now as it's likely sufficient.
//
// ref_order: The order must be exactly the
// same as indicated by this list. While there can be different
// order that still satisfy the topologial ordering, we also need
// deterministic ordering, so the results should be always the same.
void checkSortingResults(
    const ValGraph& graph,
    const ExprGroups& sorted_expr_groups,
    const ValGroups& sorted_val_groups,
    const std::vector<Expr*>& ref_order) {
  // Make sure sorted_val_groups cover all Expr groups
  const std::unordered_set<ExprGroup>& ref_expr_group_set{
      graph.disjointExprSets().disjointSets().begin(),
      graph.disjointExprSets().disjointSets().end()};
  std::unordered_set<ExprGroup> sorted_expr_group_set{
      sorted_expr_groups.begin(), sorted_expr_groups.end()};
  ASSERT_EQ(sorted_expr_group_set, ref_expr_group_set)
      << "Mismatched ExprGroups.";

  // Make sure sorted_val_groups covers all Val groups
  const std::unordered_set<ValGroup>& ref_val_group_set{
      graph.disjointValSets().disjointSets().begin(),
      graph.disjointValSets().disjointSets().end()};
  std::unordered_set<ValGroup> sorted_val_group_set{
      sorted_val_groups.begin(), sorted_val_groups.end()};
  ASSERT_EQ(sorted_val_group_set, ref_val_group_set) << "Mismatched ValGroups.";

  // Check the ordering
  ASSERT_EQ(sorted_expr_groups.size(), ref_order.size());
  for (const auto i : c10::irange(ref_order.size())) {
    Expr* ref_expr = ref_order.at(i);
    const ExprGroup& eg = sorted_expr_groups.at(i);
    ASSERT_TRUE(eg->has(ref_expr))
        << "Mismatch detected at " << i << "-th expr group. "
        << "Expected: " << nvfuser::toString(graph.toGroup(ref_expr)) << ", "
        << ref_expr->toString() << ". Actual: " << nvfuser::toString(eg) << ", "
        << eg->front()->toString();
  }
}

} // namespace

// Sorting test with a trivial fusion
TEST_F(IdModelTest, ValGraphStmtSort1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  // No ID expr yet. checkSortingResults validates the exprssion
  // order, but since there's no expr, it just makes sure exprs() and
  // vals() return all the val and expr groups.
  {
    IdModel id_model(&fusion);
    const ValGraph& vg = id_model.idGraph(IdMappingMode::EXACT);
    ValGraphStmtSort vg_stmt_sort(vg);
    checkSortingResults(vg, vg_stmt_sort.exprs(), vg_stmt_sort.vals(), {});
  }

  // Add ID exprs. Just apply a merge-and-split pattern to all
  // tensors.
  tv2->merge(0)->split(0, 4);
  TransformPropagator propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  // The exact graph should just map all IDs of the tensors. Ther
  // ordering of the exprs should be the merge and then the split.
  {
    IdModel id_model(&fusion);

    const ValGraph& vg = id_model.idGraph(IdMappingMode::EXACT);
    ValGraphStmtSort vg_stmt_sort(vg);

    // Reference expr order: merge, split
    std::vector<Expr*> ref_order;
    ref_order.push_back(getParentExpr(tv2->axis(0), 2));
    ref_order.push_back(getParentExpr(tv2->axis(0), 1));

    checkSortingResults(
        vg, vg_stmt_sort.exprs(), vg_stmt_sort.vals(), ref_order);
  }
}

// Sorting test wth a disconnected graph
TEST_F(IdModelTest, ValGraphStmtSort2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  // Note that the two groups of tensors, {tv0, tv1} and {tv2, tv3},
  // are not connected

  for (auto tv : ir_utils::allTvs(&fusion)) {
    tv->merge(0)->split(0, 4);
  }

  // Since the two tensors are disconnected, there's no ordering
  // between the ID exprs of the two tensor groups. So, the correct
  // ordering should have the merge exprs before the split exprs, but
  // there's no order between the tv1 and tv3 exprs. For example,
  // these are all valid:
  //
  // tv1 merge -> tv3 merge -> tv1 split -> tv3 split
  // tv1 merge -> tv1 split -> tv3 merge -> tv3 split
  // tv3 merge -> tv3 split -> tv1 merge -> tv1 split
  // tv3 merge -> tv1 merge -> tv3 split -> tv1 split
  //
  // Here, the actual order returned by ValGraphStmtSort is the first
  // one. Since it should be deterministic, we check if the returned
  // expr vector is indeed ordered that way.

  IdModel id_model(&fusion);

  const ValGraph& vg = id_model.idGraph(IdMappingMode::EXACT);
  ValGraphStmtSort vg_stmt_sort(vg);

  std::vector<Expr*> ref_order;
  ref_order.push_back(getParentExpr(tv1->axis(0), 2));
  ref_order.push_back(getParentExpr(tv3->axis(0), 2));
  ref_order.push_back(getParentExpr(tv1->axis(0), 1));
  ref_order.push_back(getParentExpr(tv3->axis(0), 1));

  checkSortingResults(vg, vg_stmt_sort.exprs(), vg_stmt_sort.vals(), ref_order);
}

// Sorting with trivial ExprGroup, i.e., ExprGroup whose input and
// output are mapped as the same ValGroup. It's effectively a cyclic
// dependency and the graph is no longer a DAG.
TEST_F(IdModelTest, ValGraphStmtSort3) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  auto tv3 = makeSymbolicTensor(2);
  fusion.addInput(tv3);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  // Merge and split by one. The split input and output will be mapped.
  for (auto tv : {tv0, tv1, tv2}) {
    tv->merge(0)->split(0, 1);
  }

  // Also test an isolated trivial expr. Note that tv3 and tv4 are not
  // connected with tv0, tv1 and tv2.
  tv4->merge(0)->split(0, 1);

  IdModel id_model(&fusion);
  ValGraph vg = id_model.idGraph(IdMappingMode::EXACT);

  // Map the split-by-1 input and output
  vg.mapVals(tv2->axis(0), tv2->axis(0)->definition()->input(0));
  vg.mapVals(tv4->axis(0), tv4->axis(0)->definition()->input(0));

  ValGraphStmtSort vg_stmt_sort(vg);

  std::vector<Expr*> ref_order;
  ref_order.push_back(getParentExpr(tv2->axis(0), 2));
  ref_order.push_back(getParentExpr(tv4->axis(0), 2));
  ref_order.push_back(getParentExpr(tv2->axis(0), 1));
  ref_order.push_back(getParentExpr(tv4->axis(0), 1));

  checkSortingResults(vg, vg_stmt_sort.exprs(), vg_stmt_sort.vals(), ref_order);
}

// Sorting test with the same fusion as Indexing19
TEST_F(IdModelTest, ValGraphStmtSort4) {
  auto fusion = createFusionWithMultipleResolutionPaths();
  FusionGuard fg(fusion.get());
  auto all_tvs = ir_utils::allTvs(fusion.get());

  // Since this fusion is not supported by ComputeAtMap, the
  // validation flag must be false
  IdModel id_model(fusion.get(), false, false, false);
  id_model.buildExactGraph();
  const ValGraph& vg = id_model.idGraph(IdMappingMode::EXACT);

  ValGraphStmtSort vg_stmt_sort(vg);

  auto tv1 = getValByName(all_tvs, 1);
  auto tv2 = getValByName(all_tvs, 2);
  auto tv4 = getValByName(all_tvs, 4);
  auto tv5 = getValByName(all_tvs, 5);
  auto tv6 = getValByName(all_tvs, 6);
  auto tv8 = getValByName(all_tvs, 8);
  auto tv9 = getValByName(all_tvs, 9);
  auto tv10 = getValByName(all_tvs, 10);

  // Expected reference order:
  //
  // exprg{39}: Merge iS2 bS3
  // exprg{57}: Merge iS11 bS12
  // exprg{17}: Merge iS17 bS18
  // exprg{51 63}: Merge iS15 iS16
  // exprg{69 73}: Split iS1
  // exprg{9 25 33 45}: Merge iS20 iS21
  // exprg{41}: Split iS46
  // exprg{59}: Split iS61
  // exprg{19}: Merge iS29 iS19
  // exprg{53 65}: Split iS56
  // exprg{71 75}: Split iS71
  // exprg{11}: Merge iS23 iS22
  // exprg{27}: Merge iS35 bS10
  // exprg{35 47}: Split iS41
  // exprg{43}: Split iS47
  // exprg{61}: Split iS62
  // exprg{21}: Split iS30
  // exprg{55 67}: Split iS57
  // exprg{13}: Split iS24
  // exprg{29}: Split iS36
  // exprg{37 49}: Split iS42
  // exprg{23}: Split iS31
  // exprg{15}: Split iS25
  // exprg{31}: Split iS37

  std::vector<Expr*> ref_order;
  ref_order.push_back(getParentExpr(tv2->axis(0), 3));
  ref_order.push_back(getParentExpr(tv6->axis(0), 3));
  ref_order.push_back(getParentExpr(tv9->axis(0), 4));
  ref_order.push_back(getParentExpr(tv8->axis(0), 3));
  ref_order.push_back(getParentExpr(tv1->axis(0), 2));
  ref_order.push_back(getParentExpr(tv10->axis(0), 4));
  ref_order.push_back(getParentExpr(tv2->axis(0), 2));
  ref_order.push_back(getParentExpr(tv6->axis(0), 2));
  ref_order.push_back(getParentExpr(tv9->axis(0), 3));
  ref_order.push_back(getParentExpr(tv8->axis(0), 2));
  ref_order.push_back(getParentExpr(tv1->axis(0), 1));
  ref_order.push_back(getParentExpr(tv10->axis(0), 3));
  ref_order.push_back(getParentExpr(tv5->axis(0), 3));
  ref_order.push_back(getParentExpr(tv4->axis(0), 2));
  ref_order.push_back(getParentExpr(tv2->axis(0), 1));
  ref_order.push_back(getParentExpr(tv6->axis(0), 1));
  ref_order.push_back(getParentExpr(tv9->axis(0), 2));
  ref_order.push_back(getParentExpr(tv8->axis(0), 1));
  ref_order.push_back(getParentExpr(tv10->axis(0), 2));
  ref_order.push_back(getParentExpr(tv5->axis(0), 2));
  ref_order.push_back(getParentExpr(tv4->axis(0), 1));
  ref_order.push_back(getParentExpr(tv9->axis(0), 1));
  ref_order.push_back(getParentExpr(tv10->axis(0), 1));
  ref_order.push_back(getParentExpr(tv5->axis(0), 1));

  checkSortingResults(vg, vg_stmt_sort.exprs(), vg_stmt_sort.vals(), ref_order);
}

// Testing loop promotion with a simple broadcast pattern
TEST_F(IdModelTest, LoopPromotion1) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto t0 = makeSymbolicTensor(1);
  fusion->addInput(t0);
  auto t1 = makeSymbolicTensor(2);
  fusion->addInput(t1);
  auto t2 = broadcast(t0, {true, false});
  auto t3 = add(t2, t1);
  fusion->addOutput(t3);

  {
    IdModelTester tester(fusion.get());

    // Nothing inlined. Should be no resolution
    ASSERT_TRUE(tester.s1_root_resolution_map.empty());
  }

  t2->inlineAt(2);
  ASSERT_EQ(t2->getComputeAtPosition(), 2);

  {
    IdModelTester tester(fusion.get());

    // Check Step 1 results
    // t2 is now fully inlined. Its root broadcast domain should be
    // resoled with the corresponding domain of t3
    validateIELResolution(
        t2->getRootDomain().at(0),
        t3->getRootDomain().at(0),
        tester,
        tester.s1_root_resolution_map);

    // Check Step 2 results
    // Nothing to propagate in this fusion, so iel_promotion_map
    // should be equivalent to root_resolution_map
    ASSERT_EQ(tester.s1_root_resolution_map, tester.s2_iel_promotion_map)
        << "Unexpected IEL promotion map";

    // Check Step 3 results. See the design doc for the expected results
    std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
        s3_reference_map = {
            {std::unordered_set<Val*>{t2->axis(0), t3->axis(0)}, t3->axis(0)},
            {std::unordered_set<Val*>{t2->axis(1), t3->axis(1)}, t3->axis(1)}};

    checkStep3Results(tester, s3_reference_map);

    ASSERT_TRUE(tester.s4_iel_promotion_map.empty())
        << "No step-4 IEL promotion expected";
  }
}

// Test with a fusion with progressive broadcasting
TEST_F(IdModelTest, LoopPromotion2) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto t0 = makeSymbolicTensor(1);
  fusion->addInput(t0);
  auto t1 = makeSymbolicTensor(3);
  fusion->addInput(t1);

  auto t2 = broadcast(t0, {true, false});
  auto t3 = broadcast(t2, {true, false, false});
  auto t4 = add(t3, t1);
  fusion->addOutput(t4);

  inlineMost();

  IdModelTester tester(fusion.get());

  // Check Step 1 results
  // Validate t2 and t3 as they have root broadcast domains
  validateIELResolution(
      t2->getRootDomain().at(0),
      t4->getRootDomain().at(1),
      tester,
      tester.s1_root_resolution_map);

  validateIELResolution(
      t3->getRootDomain().at(0),
      t4->getRootDomain().at(0),
      tester,
      tester.s1_root_resolution_map);

  // Check Step 2 results
  // Nothing to propagate in this fusion, so iel_promotion_map
  // should be equivalent to root_resolution_map
  ASSERT_EQ(tester.s1_root_resolution_map, tester.s2_iel_promotion_map)
      << "Unexpected IEL promotion map";

  // Check Step 3 results. See the design doc for the expected results
  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s3_reference_map = {
          {std::unordered_set<Val*>{t2->axis(0), t3->axis(1), t4->axis(1)},
           t4->axis(1)},
          {std::unordered_set<Val*>{t2->axis(1), t3->axis(2), t4->axis(2)},
           t4->axis(2)},
          {std::unordered_set<Val*>{t3->axis(0), t4->axis(0)}, t4->axis(0)}};

  checkStep3Results(tester, s3_reference_map);

  ASSERT_TRUE(tester.s4_iel_promotion_map.empty())
      << "No step-4 IEL promotion expected";
}

// Multiple inlined and non-inlined broadcast domains
TEST_F(IdModelTest, LoopPromotion3) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = makeSymbolicTensor(4);
  fusion->addInput(tv1);

  auto tv2 = broadcast(tv0, {false, true, false, true});
  auto tv3 = add(tv2, tv1);
  fusion->addOutput(tv3);

  // tv3: [i0, i1, i2, i3] -> [i0*i1, i2*i3]
  tv3->merge(0);
  tv3->merge(1);

  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv2->inlineAt(1);

  // tv2: [i0*b1, i2*b3] ca(1)
  // tv3: [i0*i1, i2*i3]

  IdModelTester tester(fusion.get());

  // Check Step 1 results
  // The b1 broadcast domain tv2 should be resolved as it's inlined,
  // but b3 should not.
  validateIELResolution(
      tv2->getRootDomain().at(1),
      tv3->getRootDomain().at(1),
      tester,
      tester.s1_root_resolution_map);

  validateIELResolution(
      tv2->getRootDomain().at(3),
      nullptr,
      tester,
      tester.s1_root_resolution_map);

  // Check Step 2 results
  validateIELResolution(
      tv2->axis(0), tv3->axis(0), tester, tester.s2_iel_promotion_map);

  validateIELResolution(
      tv2->axis(1), nullptr, tester, tester.s2_iel_promotion_map);

  // Check Step 3 results. See the design doc for the expected results
  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s3_reference_map = {
          {std::unordered_set<Val*>{
               tv2->axis(0),
               tv2->getRootDomain().at(0),
               tv2->getRootDomain().at(1),
               tv3->axis(0),
               tv3->getRootDomain().at(0),
               tv3->getRootDomain().at(1)},
           tv3->axis(0)}};

  checkStep3Results(tester, s3_reference_map);

  ASSERT_TRUE(tester.s4_iel_promotion_map.empty())
      << "No step-4 IEL promotion expected";
}

// Test root resolution with a fusion with outer split.
// Currently invalid code will be generated.
//
// Used as Example 1 in the design doc about Loop
// Promotion Analysis.
TEST_F(IdModelTest, LoopPromotion4) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({1, 4});
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor({3, 4});
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // [i0, i1]
  tv4->merge(0);
  // [i0*i1]
  tv4->split(0, 4, false); // outer split
  // [4, i0*i1/4]

  TransformPropagator propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  for (auto tv : ir_utils::allTvs(&fusion)) {
    tv->inlineAt(-2);
  }

  IdModelTester tester(&fusion);

  // Verify all tensors with root broadcast have correct resolutions
  for (auto tv : ir_utils::allTvs(&fusion)) {
    // Skip tensors with no broadcast or non-inlined
    if (std::none_of(
            tv->getRootDomain().begin(),
            tv->getRootDomain().end(),
            [](auto id) { return id->isBroadcast(); }) ||
        tv->getComputeAtPosition() == 0) {
      continue;
    }

    switch (tv->name()) {
      case 2:
        // T2_l[ iS20{4}, iS21{( ceilDiv(( 1 * 4 ), 4) )} ] ca_pos( 1 )
        //  root domain : (bS4{1}, iS5{4})
        validateIELResolution(
            tv->getRootDomain().at(0),
            tv4->getRootDomain().at(0),
            tester,
            tester.s1_root_resolution_map);
        break;
      default:
        FAIL() << "Unexpected tensor: " << tv->toString();
    }
  }

  checkStep2Results(&fusion, tester);

  auto id10 = getChildIdByName(tv4->getRootDomain()[0], 10);
  auto id13 = getChildIdByName(tv3->getRootDomain()[0], 13);
  auto id19 = getChildIdByName(tv2->getRootDomain()[0], 19);
  auto id25 = getChildIdByName(id10, 25);
  auto id26 = getChildIdByName(id10, 26);

  // Check Step 3 results. See the design doc for the expected results
  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s3_reference_map = {// 4, 6, 8 -> 8
                          {std::unordered_set<Val*>{
                               tv2->getRootDomain().at(0),
                               tv3->getRootDomain().at(0),
                               tv4->getRootDomain().at(0)},
                           tv4->getRootDomain().at(0)},
                          // 5, 7, 9 -> 9
                          {std::unordered_set<Val*>{
                               tv2->getRootDomain().at(1),
                               tv3->getRootDomain().at(1),
                               tv4->getRootDomain().at(1)},
                           tv4->getRootDomain().at(1)},
                          // 10, 13, 19 -> 10
                          {std::unordered_set<Val*>{id10, id13, id19}, id10},
                          // 11, 14, 20, 25 -> 11
                          {std::unordered_set<Val*>{
                               tv2->axis(0), tv3->axis(0), tv4->axis(0), id25},
                           tv4->axis(0)},
                          // 21, 26 -> 26
                          {std::unordered_set<Val*>{tv2->axis(1), id26}, id26}};

  checkStep3Results(tester, s3_reference_map);

  ASSERT_EQ(id10->name(), 10);
  auto id34 = getChildIdByName(id10, 34);
  auto id35 = getChildIdByName(id10, 35);

  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s4_reference_map = {// 19 -> 10
                          {std::unordered_set<Val*>{id19}, id10},
                          // 20 -> 34
                          {std::unordered_set<Val*>{tv2->axis(0)}, id34},
                          // 21 -> 35
                          {std::unordered_set<Val*>{tv2->axis(1)}, id35}};

  checkStep4Results(tester, s4_reference_map);
}

// Test root resolution with the same fusion as Indexing1
TEST_F(IdModelTest, LoopPromotion5) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = broadcast(tv2, {true, false, false, false});
  auto tv4 = add(tv3, tv1);

  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->merge(0);
  tv4->merge(0);

  tv4->split(0, 128);
  tv4->split(0, 4);

  tv2->computeAt(tv4, 1);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::Unroll);
  tv4->axis(2)->parallelize(ParallelType::TIDx);

  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(2)->parallelize(ParallelType::TIDx);

  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  auto all_tvs = ir_utils::allTvs(&fusion);

  IdModelTester tester(&fusion);

  // Check Step 1 results
  for (auto tv : all_tvs) {
    // Skip tensors with no broadcast or non-inlined
    if (std::none_of(
            tv->getRootDomain().begin(),
            tv->getRootDomain().end(),
            [](auto id) { return id->isBroadcast(); }) ||
        tv->getComputeAtPosition() == 0) {
      continue;
    }

    switch (tv->name()) {
      case 3:
        // T3_l[ iS30{( ceilDiv(( ceilDiv(( ( ( 1 * i0 ) * i2 ) * i3 ), 128) ),
        // 4) )}, iUR31{4}, ithreadIdx.x29{128} ] ca_pos( 1 ) produce_pos( 1 )
        //  root domain : (bS10{1}, iS11{i0}, iS12{i2}, iS13{i3})
        validateIELResolution(
            tv->getRootDomain().at(0),
            tv4->getRootDomain().at(0),
            tester,
            tester.s1_root_resolution_map);
        break;
      default:
        FAIL() << "Unexpected tensor: " << tv->toString();
    }
  }

  // Check Step 2 results
  checkStep2Results(&fusion, tester);

  auto id19 = getParentId(tv4->axis(0), 3);
  ASSERT_EQ(id19->name(), 19);
  auto id20 = getParentId(tv4->axis(0), 2);
  ASSERT_EQ(id20->name(), 20);
  auto id38 = getChildIdByName(id20, 38);
  auto id39 = getChildIdByName(id20, 39);
  auto id40 = getChildIdByName(id38, 40);
  auto id41 = getChildIdByName(id38, 41);

  // Check Step 3 results. See the design doc for the expected results
  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s3_reference_map = {
          // 7, 10, 11, 25, 14, 15, 18 -> 18
          {std::unordered_set<Val*>{
               tv2->getRootDomain().at(0),
               tv3->getRootDomain().at(0),
               tv3->getRootDomain().at(1),
               getParentId(tv3->axis(0), 4),
               tv4->getRootDomain().at(0),
               tv4->getRootDomain().at(1),
               getParentId(tv4->axis(0), 4)},
           getParentId(tv4->axis(0), 4)},
          // 8, 12, 16 -> 16
          {std::unordered_set<Val*>{
               tv2->getRootDomain().at(1),
               tv3->getRootDomain().at(2),
               tv4->getRootDomain().at(2)},
           tv4->getRootDomain().at(2)},
          // 9, 13, 17 -> 17
          {std::unordered_set<Val*>{
               tv2->getRootDomain().at(2),
               tv3->getRootDomain().at(3),
               tv4->getRootDomain().at(3)},
           tv4->getRootDomain().at(3)},
          // 32, 26, 19 -> 19
          {std::unordered_set<Val*>{
               getParentId(tv2->axis(0), 3),
               getParentId(tv3->axis(0), 3),
               getParentId(tv4->axis(0), 3)},
           getParentId(tv4->axis(0), 3)},
          // 33, 27, 20 -> 20
          {std::unordered_set<Val*>{
               getParentId(tv2->axis(0), 2),
               getParentId(tv3->axis(0), 2),
               getParentId(tv4->axis(0), 2)},
           getParentId(tv4->axis(0), 2)},
          // 21, 28, 34, 38 -> 21
          {std::unordered_set<Val*>{
               getParentId(tv2->axis(0), 1),
               getParentId(tv3->axis(0), 1),
               getParentId(tv4->axis(0), 1),
               id38},
           getParentId(tv4->axis(0), 1)},
          // 29, 39 -> 29
          {std::unordered_set<Val*>{tv3->axis(2), id39}, id39},
          // 31, 41 -> 41
          {std::unordered_set<Val*>{tv3->axis(1), id41}, id41},
          // 23, 30, 36, 40 -> 23
          {std::unordered_set<Val*>{
               tv2->axis(0), tv3->axis(0), tv4->axis(0), id40},
           tv4->axis(0)},
      };

  checkStep3Results(tester, s3_reference_map);

  auto id44 = getChildIdByName(id20, 44);
  auto id45 = getChildIdByName(id20, 45);
  auto id50 = getChildIdByName(id44, 50);
  auto id51 = getChildIdByName(id44, 51);

  auto id46 = getChildIdByName(id20, 46);
  auto id47 = getChildIdByName(id20, 47);
  auto id52 = getChildIdByName(id46, 52);
  auto id53 = getChildIdByName(id46, 53);

  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s4_reference_map = {
          // 32 -> 19
          {std::unordered_set<Val*>{getParentId(tv2->axis(0), 3)}, id19},
          // 33 -> 20
          {std::unordered_set<Val*>{getParentId(tv2->axis(0), 2)}, id20},
          // 34 -> 44
          {std::unordered_set<Val*>{getParentId(tv2->axis(0), 1)}, id44},
          // 35 -> 45
          {std::unordered_set<Val*>{tv2->axis(2)}, id45},
          // 36 -> 50
          {std::unordered_set<Val*>{tv2->axis(0)}, id50},
          // 37 -> 41
          {std::unordered_set<Val*>{tv2->axis(1)}, id51},
          // 26 -> 19
          {std::unordered_set<Val*>{getParentId(tv3->axis(0), 3)}, id19},
          // 27 -> 20
          {std::unordered_set<Val*>{getParentId(tv3->axis(0), 2)}, id20},
          // 28 -> 46
          {std::unordered_set<Val*>{getParentId(tv3->axis(0), 1)}, id46},
          // 29 -> 47
          {std::unordered_set<Val*>{tv3->axis(2)}, id47},
          // 30 -> 52
          {std::unordered_set<Val*>{tv3->axis(0)}, id52},
          // 31 -> 53
          {std::unordered_set<Val*>{tv3->axis(1)}, id53}};

  checkStep4Results(tester, s4_reference_map);
}

// Test root resolution with the same fusion as Indexing19
TEST_F(IdModelTest, LoopPromotion6) {
  auto fusion = createFusionWithMultipleResolutionPaths();
  FusionGuard fg(fusion.get());
  auto all_tvs = ir_utils::allTvs(fusion.get());

  fusion->printMath();
  fusion->print();

  IdModelTester tester(fusion.get());

  auto tv1 = getValByName(all_tvs, 1);
  auto tv2 = getValByName(all_tvs, 2);
  auto tv4 = getValByName(all_tvs, 4);
  auto tv5 = getValByName(all_tvs, 5);
  auto tv6 = getValByName(all_tvs, 6);
  auto tv8 = getValByName(all_tvs, 8);
  auto tv9 = getValByName(all_tvs, 9);

  // Check Step 1 results
  for (auto tv : all_tvs) {
    // Skip tensors with no broadcast or non-inlined
    if (std::none_of(
            tv->getRootDomain().begin(),
            tv->getRootDomain().end(),
            [](auto id) { return id->isBroadcast(); }) ||
        tv->getComputeAtPosition() == 0) {
      continue;
    }

    switch (tv->name()) {
      case 2:
        // T2_l[ iS49{( ceilDiv(( ceilDiv(( 7 * 1 ), 5) ), 3) )}, iS50{3},
        // iS48{5} ] ca_pos( 1 ) produce_pos( 1 )
        //  root domain : (iS2{7}, bS3{1})
        // Resolution: Resolved by the immediate consumer (T4)
        validateIELResolution(
            tv->getRootDomain().at(1),
            tv4->getRootDomain().at(1),
            tester,
            tester.s1_root_resolution_map);
        break;
      case 5:
        // T5_l[ iS39{( ceilDiv(( ceilDiv(( ( 7 * 11 ) * 1 ), 5) ), 3) )},
        // iS40{3}, iS38{5} ] produce_pos( 1 )
        //  root domain : (iS8{7}, iS9{11}, bS10{1})
        // Resolution: T5 is not inlined to the immediate consumer,
        // T10. Resolution is done with the other path from T1, such
        // as T8 or T9.
        validateIELResolution(
            tv->getRootDomain().at(2),
            tv9->getRootDomain().at(2),
            tester,
            tester.s1_root_resolution_map);
        break;
      case 6:
        // T6_l[ iS64{( ceilDiv(( ceilDiv(( 7 * 1 ), 5) ), 3) )}, iS65{3},
        // iS63{5} ] ca_pos( 1 ) produce_pos( 1 )
        //  root domain : (iS11{7}, bS12{1})
        // Resolution: Resolved by the immediate consumer (T8)
        validateIELResolution(
            tv->getRootDomain().at(1),
            tv8->getRootDomain().at(1),
            tester,
            tester.s1_root_resolution_map);
        break;
      case 9:
        // T9_l[ iS33{( ceilDiv(( ceilDiv(( ( 7 * 1 ) * 13 ), 5) ), 3) )},
        // iS34{3}, iS32{5} ] produce_pos( 1 )
        //  root domain : (iS17{7}, bS18{1}, iS19{13})
        // Resolution: T9 is not inlined to the immediate consumer,
        // T10. Resolution is done with the other path from T1, such
        // as T4 or T5
        validateIELResolution(
            tv->getRootDomain().at(1),
            tv5->getRootDomain().at(1),
            tester,
            tester.s1_root_resolution_map);
        break;
      default:
        FAIL() << "Unexpected tensor: " << tv->toString();
    }
  }

  checkStep2Results(fusion.get(), tester);

  // 83 -> 89, 90
  // 89 -> 93, 94
  auto id83 = getChildIdByName(tv9->getRootDomain().at(2), 83);
  auto id89 = getChildIdByName(id83, 89);
  auto id90 = getChildIdByName(id83, 90);
  auto id93 = getChildIdByName(id89, 93);
  auto id94 = getChildIdByName(id89, 94);

  // 84 -> 91, 92
  // 91 -> 95, 96
  auto id84 = getChildIdByName(tv9->getRootDomain().at(2), 84);
  auto id91 = getChildIdByName(id84, 91);
  auto id92 = getChildIdByName(id84, 92);
  auto id95 = getChildIdByName(id91, 95);
  auto id96 = getChildIdByName(id91, 96);

  // 35 -> 79, 80
  // 79 -> 85, 86
  auto id35 = getChildIdByName(tv5->getRootDomain().at(0), 35);
  auto id79 = getChildIdByName(id35, 79);
  auto id80 = getChildIdByName(id35, 80);
  auto id85 = getChildIdByName(id79, 85);
  auto id86 = getChildIdByName(id79, 86);

  // 56 -> 81, 82
  // 81 -> 87, 88
  auto id56 = getChildIdByName(tv8->getRootDomain().at(0), 56);
  auto id81 = getChildIdByName(id56, 81);
  auto id82 = getChildIdByName(id56, 82);
  auto id87 = getChildIdByName(id81, 87);
  auto id88 = getChildIdByName(id81, 88);

  // Check Step 3 results. See the design doc for the expected results
  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s3_reference_map = {
          // 1 2 3 6 7 8 9 10 11 12 15 16 17 18 19 29 30 35 36 41 46 56 61
          // 83 84 -> 84
          {std::unordered_set<Val*>{
               tv1->getRootDomain().at(0),
               tv2->getRootDomain().at(0),
               tv2->getRootDomain().at(1),
               getChildId(tv2->getRootDomain().at(0), 1),
               tv4->getRootDomain().at(0),
               tv4->getRootDomain().at(1),
               getChildId(tv4->getRootDomain().at(0), 1),
               tv5->getRootDomain().at(0),
               tv5->getRootDomain().at(1),
               tv5->getRootDomain().at(2),
               getChildId(tv5->getRootDomain().at(0), 1),
               getChildId(tv5->getRootDomain().at(2), 1),
               tv6->getRootDomain().at(0),
               tv6->getRootDomain().at(1),
               getChildId(tv6->getRootDomain().at(0), 1),
               tv8->getRootDomain().at(0),
               tv8->getRootDomain().at(1),
               getChildId(tv8->getRootDomain().at(0), 1),
               tv9->getRootDomain().at(0),
               tv9->getRootDomain().at(1),
               tv9->getRootDomain().at(2),
               getChildId(tv9->getRootDomain().at(0), 1),
               getChildId(tv9->getRootDomain().at(0), 2),
               id83,
               id84},
           id84},
          // 31 37 42 47 57 62 71 79 81 89 91 -> 91
          {std::unordered_set<Val*>{
               getChildId(tv1->getRootDomain().at(0), 1),
               getChildId(tv2->getRootDomain().at(0), 2),
               getChildId(tv4->getRootDomain().at(0), 2),
               getChildId(tv5->getRootDomain().at(0), 3),
               getChildId(tv6->getRootDomain().at(0), 2),
               getChildId(tv8->getRootDomain().at(0), 2),
               getChildId(tv9->getRootDomain().at(0), 3),
               id79,
               id81,
               id89,
               id91},
           id91},
          // 33 39 44 49 59 64 73 85 87 93 95 -> 95
          {std::unordered_set<Val*>{
               tv1->axis(0),
               tv2->axis(0),
               tv4->axis(0),
               tv5->axis(0),
               tv6->axis(0),
               tv8->axis(0),
               tv9->axis(0),
               id85,
               id87,
               id93,
               id95},
           id95},
          // 48 80 -> 80
          {std::unordered_set<Val*>{tv2->axis(2), id80}, id80},
          // 50 86 -> 86
          {std::unordered_set<Val*>{tv2->axis(1), id86}, id86},
          // 40 96 -> 96
          {std::unordered_set<Val*>{tv5->axis(1), id96}, id96},
          // 63 82 -> 82
          {std::unordered_set<Val*>{tv6->axis(2), id82}, id82},
          // 65 88 -> 88
          {std::unordered_set<Val*>{tv6->axis(1), id88}, id88},
          // 34 94 -> 94
          {std::unordered_set<Val*>{tv9->axis(1), id94}, id94},
          // 38 92 -> 92
          {std::unordered_set<Val*>{tv5->axis(2), id92}, id92},
          // 32 90 -> 90
          {std::unordered_set<Val*>{tv9->axis(2), id90}, id90},
      };

  checkStep3Results(tester, s3_reference_map);

  // For tv1
  auto id102 = getChildIdByName(id84, 102);
  auto id103 = getChildIdByName(id84, 103);
  auto id117 = getChildIdByName(id102, 117);
  auto id118 = getChildIdByName(id102, 118);

  // For tv2
  auto id106 = getChildIdByName(id84, 106);
  auto id107 = getChildIdByName(id84, 107);
  auto id121 = getChildIdByName(id106, 121);
  auto id122 = getChildIdByName(id106, 122);

  // For tv6
  auto id110 = getChildIdByName(id84, 110);
  auto id111 = getChildIdByName(id84, 111);
  auto id125 = getChildIdByName(id110, 125);
  auto id126 = getChildIdByName(id110, 126);

  // For tv4
  auto id119 = getChildIdByName(id84, 119);
  auto id120 = getChildIdByName(id84, 120);
  auto id137 = getChildIdByName(id119, 137);
  auto id138 = getChildIdByName(id119, 138);

  // For tv5
  auto id135 = getChildIdByName(id84, 135);
  auto id136 = getChildIdByName(id84, 136);
  auto id143 = getChildIdByName(id135, 143);
  auto id144 = getChildIdByName(id135, 144);

  // For tv8
  auto id115 = getChildIdByName(id84, 115);
  auto id116 = getChildIdByName(id84, 116);
  auto id133 = getChildIdByName(id115, 133);
  auto id134 = getChildIdByName(id115, 134);

  // For tv9
  auto id129 = getChildIdByName(id84, 129);
  auto id130 = getChildIdByName(id84, 130);
  auto id139 = getChildIdByName(id129, 139);
  auto id140 = getChildIdByName(id129, 140);

  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s4_reference_map = {
          // tv1: 71 -> 102
          {std::unordered_set<Val*>{getParentId(tv1->axis(0), 1)}, id102},
          // tv1: 72 -> 103
          {std::unordered_set<Val*>{tv1->axis(2)}, id103},
          // tv1: 73 -> 117
          {std::unordered_set<Val*>{tv1->axis(0)}, id117},
          // tv1: 74 -> 118
          {std::unordered_set<Val*>{tv1->axis(1)}, id118},
          // tv2: 47 -> 106
          {std::unordered_set<Val*>{getParentId(tv2->axis(0), 1)}, id106},
          // tv2: 48 -> 107
          {std::unordered_set<Val*>{tv2->axis(2)}, id107},
          // tv2: 49 -> 121
          {std::unordered_set<Val*>{tv2->axis(0)}, id121},
          // tv2: 50 -> 122
          {std::unordered_set<Val*>{tv2->axis(1)}, id122},
          // tv4: 42 -> 119
          {std::unordered_set<Val*>{getParentId(tv4->axis(0), 1)}, id119},
          // tv4: 43 -> 120
          {std::unordered_set<Val*>{tv4->axis(2)}, id120},
          // tv4: 44 -> 137
          {std::unordered_set<Val*>{tv4->axis(0)}, id137},
          // tv4: 45 -> 138
          {std::unordered_set<Val*>{tv4->axis(1)}, id138},
          // tv5: 37 -> 135
          {std::unordered_set<Val*>{getParentId(tv5->axis(0), 1)}, id135},
          // tv5: 38 -> 136
          {std::unordered_set<Val*>{tv5->axis(2)}, id136},
          // tv5: 39 -> 143
          {std::unordered_set<Val*>{tv5->axis(0)}, id143},
          // tv5: 40 -> 144
          {std::unordered_set<Val*>{tv5->axis(1)}, id144},
          // tv6: 62 -> 110
          {std::unordered_set<Val*>{getParentId(tv6->axis(0), 1)}, id110},
          // tv6: 63 -> 111
          {std::unordered_set<Val*>{tv6->axis(2)}, id111},
          // tv6: 64 -> 125
          {std::unordered_set<Val*>{tv6->axis(0)}, id125},
          // tv6: 65 -> 126
          {std::unordered_set<Val*>{tv6->axis(1)}, id126},
          // tv8: 57 -> 115
          {std::unordered_set<Val*>{getParentId(tv8->axis(0), 1)}, id115},
          // tv8: 58 -> 116
          {std::unordered_set<Val*>{tv8->axis(2)}, id116},
          // tv8: 59 -> 133
          {std::unordered_set<Val*>{tv8->axis(0)}, id133},
          // tv8: 60 -> 134
          {std::unordered_set<Val*>{tv8->axis(1)}, id134},
          // tv9: 31 -> 129
          {std::unordered_set<Val*>{getParentId(tv9->axis(0), 1)}, id129},
          // tv9: 32 -> 130
          {std::unordered_set<Val*>{tv9->axis(2)}, id130},
          // tv9: 33 -> 139
          {std::unordered_set<Val*>{tv9->axis(0)}, id139},
          // tv9: 34 -> 140
          {std::unordered_set<Val*>{tv9->axis(1)}, id140}};

  checkStep4Results(tester, s4_reference_map);
}

// Same fusion as NvFuserTest.FusionInlineBroadcastIndexing0
TEST_F(IdModelTest, LoopPromotion7) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = set(tv0);
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->split(0, 32);

  TransformPropagatorWithCheck propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv2->inlineAt(1);
  tv3->inlineAt(1);

  tv2->split(-1, 8);

  auto all_tvs = ir_utils::allTvs(&fusion);

  fusion.printMath();
  fusion.print();

  IdModelTester tester(&fusion);

  // Verify all tensors with root broadcast have correct resolutions
  for (auto tv : all_tvs) {
    // Skip tensors with no broadcast or non-inlined
    if (std::none_of(
            tv->getRootDomain().begin(),
            tv->getRootDomain().end(),
            [](auto id) { return id->isBroadcast(); }) ||
        tv->getComputeAtPosition() == 0) {
      continue;
    }

    switch (tv->name()) {
      case 3:
        // T3_l[ iS15{( ceilDiv(( 1 * i0 ), 32) )}, iS16{32} ] ca_pos( 1 )
        // produce_pos( 1 ) root domain : (bS4{1}, iS5{i0})
        validateIELResolution(
            tv->getRootDomain().at(0),
            tv4->getRootDomain().at(0),
            tester,
            tester.s1_root_resolution_map);
        break;
      default:
        FAIL() << "Unexpected tensor: " << tv->toString();
    }
  }

  checkStep2Results(&fusion, tester);

  auto id8 = getChildIdByName(tv4->getRootDomain().at(0), 8);
  auto id23 = getChildIdByName(id8, 23);
  auto id24 = getChildIdByName(id8, 24);

  // Check Step 3 results. See the design doc for the expected results
  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s3_reference_map = {// 3, 4, 5, 14, 6, 7, 8, -> 8
                          {std::unordered_set<Val*>{
                               tv2->getRootDomain().at(0),
                               tv3->getRootDomain().at(0),
                               tv3->getRootDomain().at(1),
                               getChildId(tv3->getRootDomain().at(0), 1),
                               tv4->getRootDomain().at(0),
                               tv4->getRootDomain().at(1),
                               id8},
                           id8},
                          // 9, 15, 17, 23 -> 9
                          {std::unordered_set<Val*>{
                               tv2->axis(0), tv3->axis(0), tv4->axis(0), id23},
                           tv4->axis(0)},
                          // 16, 24 -> 24
                          {std::unordered_set<Val*>{tv3->axis(1), id24}, id24}};

  checkStep3Results(tester, s3_reference_map);

  // For tv2
  auto id28 = getChildIdByName(id8, 28);
  auto id29 = getChildIdByName(id8, 29);
  auto id36 = getChildIdByName(id29, 36);
  auto id37 = getChildIdByName(id29, 37);

  // For tv3
  auto id32 = getChildIdByName(id8, 32);
  auto id33 = getChildIdByName(id8, 33);

  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s4_reference_map = {
          // tv2: 17 -> 28
          {std::unordered_set<Val*>{tv2->axis(0)}, id28},
          // tv2: 18 -> 29
          {std::unordered_set<Val*>{getParentId(tv2->axis(1), 1)}, id29},
          // tv2: 21 -> 36
          {std::unordered_set<Val*>{tv2->axis(1)}, id36},
          // tv2: 22 -> 37
          {std::unordered_set<Val*>{tv2->axis(2)}, id37},
          // tv3: 15 -> 32
          {std::unordered_set<Val*>{tv3->axis(0)}, id32},
          // tv3: 16 -> 33
          {std::unordered_set<Val*>{tv3->axis(1)}, id33},
      };

  checkStep4Results(tester, s4_reference_map);
}

// Same fusion as NvFuserTest.FusionIndexing20
TEST_F(IdModelTest, LoopPromotion8) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({5});
  fusion.addInput(tv0);

  // [5]
  auto tv1 = set(tv0);
  auto tv2 = broadcast(tv1, {true, false});
  // [1, 5]
  auto tv3 = makeConcreteTensor({3, 5});
  fusion.addInput(tv3);
  auto tv4 = add(tv3, tv2);
  // [3, 5]

  auto tv5 = broadcast(tv4, {false, false, true});
  // [3, 5, 1]
  auto tv6 = makeConcreteTensor({3, 5, 7});
  fusion.addInput(tv6);
  auto tv7 = add(tv5, tv6);
  // [3, 5, 7]
  fusion.addOutput(tv7);

  tv4->merge(0)->split(0, 2, false);
  // [3, 5]
  // [3, 3*5//2]

  TransformPropagatorWithCheck propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv1->inlineAt(1);
  tv2->inlineAt(1);
  tv4->inlineAt(1);

  // [2, 3*5//2]
  tv5->merge(1)->split(1, 4, false);
  // [2, 4, (3*5//2)*1//4]
  tv7->merge(1)->split(1, 4, false);
  // [2, 4, (3*5//2)*7//4]
  tv5->inlineAt(2);

  auto all_tvs = ir_utils::allTvs(&fusion);

  fusion.printMath();
  fusion.print();

  IdModelTester tester(&fusion);

  // Verify all tensors with root broadcast have correct resolutions
  for (auto tv : all_tvs) {
    // Skip tensors with no broadcast or non-inlined
    if (std::none_of(
            tv->getRootDomain().begin(),
            tv->getRootDomain().end(),
            [](auto id) { return id->isBroadcast(); }) ||
        tv->getComputeAtPosition() == 0) {
      continue;
    }

    switch (tv->name()) {
      case 2:
        // T2_l[ iS21{2}, iS22{( ceilDiv(( 1 * 5 ), 2) )} ] ca_pos( 1 )
        // produce_pos( 1 ) root domain : (bS2{1}, iS3{5})
        validateIELResolution(
            tv->getRootDomain().at(0),
            tv7->getRootDomain().at(0),
            tester,
            tester.s1_root_resolution_map);
        break;
      case 5:
        // T5_l[ iS27{2}, iS40{4}, iS41{( ceilDiv(( ( ceilDiv(( 3 * 5 ), 2) ) *
        // 1 ), 4) )} ] ca_pos( 2 ) produce_pos( 1 ) root domain : (iS8{3},
        // iS9{5}, bS10{1})
        validateIELResolution(
            tv->getRootDomain().at(2),
            tv7->getRootDomain().at(2),
            tester,
            tester.s1_root_resolution_map);
        break;
      default:
        FAIL() << "Unexpected tensor: " << tv->toString();
    }
  }

  checkStep2Results(&fusion, tester);

  // tv7
  auto id14 = tv7->getRootDomain().at(0);
  auto id15 = tv7->getRootDomain().at(1);
  auto id16 = tv7->getRootDomain().at(2);
  auto id29 = getChildIdByName(id14, 29);
  auto id30 = getChildIdByName(id29, 30);
  auto id31 = getChildIdByName(id29, 31);
  auto id43 = tv7->axis(1);

  // tv2
  auto id2 = tv2->getRootDomain().at(0);
  auto id3 = tv2->getRootDomain().at(1);
  auto id20 = getChildIdByName(id2, 20);
  auto id21 = tv2->axis(0);
  auto id22 = tv2->axis(1);
  auto id45 = getChildIdByName(id29, 45);
  auto id46 = getChildIdByName(id29, 46);

  // tv5
  auto id8 = tv5->getRootDomain().at(0);
  auto id9 = tv5->getRootDomain().at(1);
  auto id10 = tv5->getRootDomain().at(2);
  auto id27 = tv5->axis(0);
  auto id26 = getChildIdByName(id8, 26);
  auto id28 = getChildIdByName(id26, 28);
  auto id39 = getChildIdByName(id28, 39);
  auto id40 = tv5->axis(1);
  auto id41 = tv5->axis(2);
  auto id42 = getChildIdByName(id16, 42);
  auto id47 = getChildIdByName(id42, 47);
  auto id48 = getChildIdByName(id42, 48);

  // tv4
  auto id6 = tv4->getRootDomain().at(0);
  auto id7 = tv4->getRootDomain().at(1);
  auto id17 = getChildIdByName(id6, 17);
  auto id18 = tv4->axis(0);

  // tv1
  auto id1 = tv1->getRootDomain().at(0);
  auto id35 = tv1->axis(0);
  auto id36 = tv1->axis(1);

  // Check Step 3 results. See the design doc for the expected results
  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s3_reference_map = {
          // 1, 2, 3, 20, 6, 7, 17, 8, 9, 26, 14, 15, 29 -> 29
          {std::unordered_set<Val*>{
               id1,
               id2,
               id3,
               id20,
               id6,
               id7,
               id17,
               id8,
               id9,
               id26,
               id14,
               id15,
               id29},
           id29},
          // 18, 21, 27, 30, 35, 45 -> 30
          {std::unordered_set<Val*>{id18, id21, id27, id30, id35, id45}, id30},
          // 10, 16, 28, 31, 39, 42 -> 42
          {std::unordered_set<Val*>{id10, id16, id28, id31, id39, id42}, id42},
          // 22, 46 -> 46
          {std::unordered_set<Val*>{id22, id46}, id46},
          // 40, 43, 47 -> 43
          {std::unordered_set<Val*>{id40, id43, id47}, id43},
          // 41, 48 -> 48
          {std::unordered_set<Val*>{id41, id48}, id48}};

  checkStep3Results(tester, s3_reference_map);

  // tv1
  auto id53 = getChildIdByName(id29, 53);
  auto id54 = getChildIdByName(id29, 54);

  // tv2
  auto id55 = getChildIdByName(id29, 55);
  auto id56 = getChildIdByName(id29, 56);

  // tv5
  auto id67 = getChildIdByName(id42, 67);
  auto id68 = getChildIdByName(id42, 68);

  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s4_reference_map = {
          // tv1: 35 -> 53
          {std::unordered_set<Val*>{id35}, id53},
          // tv1: 36 -> 54
          {std::unordered_set<Val*>{id36}, id54},
          // tv2: 21 -> 55
          {std::unordered_set<Val*>{id21}, id55},
          // tv2: 22 -> 56
          {std::unordered_set<Val*>{id22}, id56},
          // tv5: 40 -> 67
          {std::unordered_set<Val*>{id40}, id67},
          // tv5: 41 -> 68
          {std::unordered_set<Val*>{id41}, id68},
      };

  checkStep4Results(tester, s4_reference_map);
}

TEST_F(IdModelTest, LoopPromotionPromoteToSameLoopGroup) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, false});
  auto tv3 = add(tv2, tv1);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  // [I0, I1]
  tv4->split(1, 2);
  // [I0, I1/2, 2]
  tv4->split(0, 8);
  // [I0/8*I1/2, 8, 2]
  tv4->merge(0, 2);
  // [I0/8*I1/2, 8*2]
  tv4->merge(1, 2);

  TransformPropagatorWithCheck propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  for (auto tv : {tv0, tv1, tv2, tv3}) {
    tv->inlineAt(1);
  }

  IdModelTester tester(&fusion);

  ASSERT_EQ(tester.s1_root_resolution_map.size(), 1);
  validateIELResolution(
      tv2->getRootDomain().at(0),
      tv4->getRootDomain().at(0),
      tester,
      tester.s1_root_resolution_map);

  checkStep2Results(&fusion, tester);

  // tv4
  auto id7 = tv4->getRootDomain().at(0);
  auto id8 = tv4->getRootDomain().at(1);
  auto id11 = getChildIdByName(id7, 11);
  auto id9 = getChildIdByName(id8, 9);
  auto id13 = getChildIdByName(id11, 13);

  // tv3
  auto id5 = tv3->getRootDomain().at(0);
  auto id6 = tv3->getRootDomain().at(1);
  auto id15 = getChildIdByName(id5, 15);
  auto id17 = getChildIdByName(id6, 17);
  auto id19 = getChildIdByName(id15, 19);

  // tv2
  auto id3 = tv2->getRootDomain().at(0);
  auto id4 = tv2->getRootDomain().at(1);
  auto id27 = getChildIdByName(id3, 27);
  auto id28 = getChildIdByName(id3, 28);
  auto id29 = getChildIdByName(id4, 29);
  auto id30 = getChildIdByName(id4, 30);
  auto id31 = getChildIdByName(id27, 31);
  auto id32 = getChildIdByName(id30, 32);

  auto id35 = getChildIdByName(id7, 35);
  auto id36 = getChildIdByName(id7, 36);
  auto id37 = getChildIdByName(id35, 37);
  auto id38 = getChildIdByName(id30, 38);

  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s3_reference_map = {
          {std::unordered_set<Val*>{id32, id38}, id38},
          {std::unordered_set<Val*>{id4, id6, id8}, id8},
          {std::unordered_set<Val*>{id28, id36}, id36},
          {std::unordered_set<Val*>{
               id9, id11, id13, id15, id17, id19, id27, id29, id31, id35, id37},
           id13},
          {std::unordered_set<Val*>{id3, id5, id7}, id7}};

  checkStep3Results(tester, s3_reference_map);

  auto id45 = getChildIdByName(id7, 45);
  auto id46 = getChildIdByName(id7, 46);
  auto id59 = getChildIdByName(id45, 59);
  auto id60 = getChildIdByName(id46, 60);

  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s4_reference_map = {
          {std::unordered_set<Val*>{id27}, id45},
          {std::unordered_set<Val*>{id28}, id46},
          {std::unordered_set<Val*>{id31}, id59},
          {std::unordered_set<Val*>{id32}, id60}};

  checkStep4Results(tester, s4_reference_map);
}

namespace {
bool iterDomainsAreMapped(
    const IdModel& id_model,
    IterDomain* a,
    IterDomain* b) {
  const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);
  return exact_graph.disjointValSets().strictAreMapped(a, b);
}
} // namespace

TEST_F(IdModelTest, SomeButNotAllArePermuted) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 2, 5});
  TensorView* s0 = slice(in, {0, 0, 0}, {2, 2, 2});
  TensorView* s1 = slice(in, {0, 0, 2}, {2, 2, 5});
  TensorView* t0 = permute(s0, {1, 0, 2});
  TensorView* out = cat({t0, s1}, /*dim=*/-1);

  fusion->addInput(in);
  fusion->addOutput(out);

  IdModel id_model(
      fusion.get(), /*build_graphs=*/true, /*allow_self_mapping=*/true);
  EXPECT_TRUE(iterDomainsAreMapped(id_model, s0->axis(0), t0->axis(1)));
  EXPECT_TRUE(iterDomainsAreMapped(id_model, s0->axis(1), t0->axis(0)));
  EXPECT_TRUE(iterDomainsAreMapped(id_model, s0->axis(2), t0->axis(2)));
}

TEST_F(IdModelTest, PermutedDifferently) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 2, 5});
  TensorView* s0 = slice(in, {0, 0, 0}, {2, 2, 2});
  TensorView* s1 = slice(in, {0, 0, 2}, {2, 2, 5});
  TensorView* t0 = permute(s0, {1, 0, 2});
  TensorView* t1 = set(s1);
  TensorView* out = cat({t0, t1}, /*dim=*/-1);

  fusion->addInput(in);
  fusion->addOutput(out);

  IdModel id_model(
      fusion.get(), /*build_graphs=*/true, /*allow_self_mapping=*/true);

  // Due to the `slice`s, `s0` and `s1`'s non-split dimensions (0 and 1) are
  // mapped respectively. The split dimension (2) isn't.
  EXPECT_TRUE(iterDomainsAreMapped(id_model, s0->axis(0), s1->axis(0)));
  EXPECT_TRUE(iterDomainsAreMapped(id_model, s0->axis(1), s1->axis(1)));
  EXPECT_FALSE(iterDomainsAreMapped(id_model, s0->axis(2), s1->axis(2)));

  // Due to the `cat`, t0' and `t1`'s non-catted dimensions (0 and 1) are
  // respectively mapped. The catted dimension (2) isn't.
  EXPECT_TRUE(iterDomainsAreMapped(id_model, t0->axis(0), t1->axis(0)));
  EXPECT_TRUE(iterDomainsAreMapped(id_model, t0->axis(1), t1->axis(1)));
  EXPECT_FALSE(iterDomainsAreMapped(id_model, t0->axis(2), t1->axis(2)));

  // Check the mapping introduced by `t0 = permute(s0, ...)`.
  EXPECT_TRUE(iterDomainsAreMapped(id_model, s0->axis(1), t0->axis(0)));
  EXPECT_TRUE(iterDomainsAreMapped(id_model, s0->axis(0), t0->axis(1)));
  EXPECT_TRUE(iterDomainsAreMapped(id_model, s0->axis(2), t0->axis(2)));

  // Check the mapping introduced by `t1 = set(s1, ...)`.
  EXPECT_TRUE(iterDomainsAreMapped(id_model, s1->axis(0), t1->axis(0)));
  EXPECT_TRUE(iterDomainsAreMapped(id_model, s1->axis(1), t1->axis(1)));
  EXPECT_TRUE(iterDomainsAreMapped(id_model, s1->axis(2), t1->axis(2)));
}

} // namespace nvfuser
