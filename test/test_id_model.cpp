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

#include <test/utils.h>
#include <test/validator.h>

#include <fusion.h>
#include <id_model/id_model.h>
#include <id_model/to_string.h>
#include <inlining.h>
#include <ops/all_ops.h>
#include <transform_iter.h>
#include <val_graph_visitor.h>

namespace nvfuser {

class IdModelTest : public NVFuserTest {};

TEST_F(IdModelTest, DetectSelfMapping) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 2});
  fusion.addInput(tv0);
  auto tv1 = transpose(tv0, 0, 1);
  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  EXPECT_THAT(
      [&]() {
        IdModel id_model(&fusion);
        id_model.buildAllGraphs();
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("!hasSelfMapping")));
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

// Helper class to test IdModel
class IdModelTester : public IdModel {
 public:
  // Do not automatically build the graphs
  IdModelTester(Fusion* fusion) : IdModel(fusion, /* build_graphs */ false) {
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

    s3_loop_promotion_map = projectIELPromotionToLoopGraph(
        iel_graph,
        s2_iel_promotion_map,
        idGraph(IdMappingMode::LOOP),
        inlining_info);
  }

  ValGraph iel_graph;
  std::unordered_map<ValGroup, IterDomain*> s1_root_resolution_map;
  std::unordered_map<ValGroup, IterDomain*> s2_iel_promotion_map;
  std::unordered_map<ValGroup, IterDomain*> s3_loop_promotion_map;
};

// Test if id is resolved to an ID that is exact mapped with
// ref_id. If ref_id  is nullptr, test if root_broadcast_id has no
// resolution.
void validateIELResolution(
    IterDomain* id,
    IterDomain* ref_id,
    const ValGraph& iel_graph,
    const ValGraph& exact_graph,
    const std::unordered_map<ValGroup, IterDomain*>& iel_promotion_map) {
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
void checkStep2Results(
    Fusion* fusion,
    const ValGraph& iel_graph,
    const ValGraph& exact_graph,
    const std::unordered_map<ValGroup, IterDomain*>& iel_promotion_map) {
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
      validateIELResolution(
          p_id, c_id, iel_graph, exact_graph, iel_promotion_map);
    }
  }
}

// Validate the loop promotion map at Step 3. This validation ensures
// the promotion map is exactly the same as a given reference
// map. Since the valid promotion map may not be unique, the exact
// equality is not required, however, as long as everything is done
// deterministically, the resulting map should alwasy be the
// same. The exact equality helps ensure the determinism as well.
void checkStep3Results(
    const ValGraph& loop_graph,
    const std::unordered_map<ValGroup, IterDomain*>& loop_promotion_map,
    const std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>&
        ref_promotion_map) {
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

    ASSERT_NE(ref_promotion_it, ref_promotion_map.end())
        << "No matching loop group found in the reference map: "
        << nvfuser::toString(loop_group);

    auto ref_promotion_id = ref_promotion_it->second;
    ASSERT_EQ(promotion_id, ref_promotion_id)
        << "Expected promotion: " << ref_promotion_id->toString()
        << ". Actual: " << promotion_id->toString();
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
        tester.iel_graph,
        tester.idGraph(IdMappingMode::EXACT),
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

    checkStep3Results(
        tester.idGraph(IdMappingMode::LOOP),
        tester.s3_loop_promotion_map,
        s3_reference_map);
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
      tester.iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      tester.s1_root_resolution_map);

  validateIELResolution(
      t3->getRootDomain().at(0),
      t4->getRootDomain().at(0),
      tester.iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
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

  checkStep3Results(
      tester.idGraph(IdMappingMode::LOOP),
      tester.s3_loop_promotion_map,
      s3_reference_map);
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
      tester.iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      tester.s1_root_resolution_map);

  validateIELResolution(
      tv2->getRootDomain().at(3),
      nullptr,
      tester.iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      tester.s1_root_resolution_map);

  // Check Step 2 results
  validateIELResolution(
      tv2->axis(0),
      tv3->axis(0),
      tester.iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      tester.s2_iel_promotion_map);

  validateIELResolution(
      tv2->axis(1),
      nullptr,
      tester.iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      tester.s2_iel_promotion_map);

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

  checkStep3Results(
      tester.idGraph(IdMappingMode::LOOP),
      tester.s3_loop_promotion_map,
      s3_reference_map);
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
            tester.iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
            tester.s1_root_resolution_map);
        break;
      default:
        FAIL() << "Unexpected tensor: " << tv->toString();
    }
  }

  checkStep2Results(
      &fusion,
      tester.iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      tester.s2_iel_promotion_map);

  // Check Step 3 results. See the design doc for the expected results
  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s3_reference_map = {
          // 4, 6, 8 -> 8
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
          {std::unordered_set<Val*>{
               getParentId(tv2->axis(0), 1),
               getParentId(tv3->axis(0), 1),
               getParentId(tv4->axis(0), 1)},
           getParentId(tv4->axis(0), 1)},
          // 11, 14, 20 -> 11
          {std::unordered_set<Val*>{tv2->axis(0), tv3->axis(0), tv4->axis(0)},
           tv4->axis(0)},
          // 21 -> 12
          {std::unordered_set<Val*>{tv2->axis(1)}, tv4->axis(1)}};

  checkStep3Results(
      tester.idGraph(IdMappingMode::LOOP),
      tester.s3_loop_promotion_map,
      s3_reference_map);
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
            tester.iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
            tester.s1_root_resolution_map);
        break;
      default:
        FAIL() << "Unexpected tensor: " << tv->toString();
    }
  }

  // Check Step 2 results
  checkStep2Results(
      &fusion,
      tester.iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      tester.s2_iel_promotion_map);

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
          // 34, 28, 21 -> 21
          {std::unordered_set<Val*>{
               getParentId(tv2->axis(0), 1),
               getParentId(tv3->axis(0), 1),
               getParentId(tv4->axis(0), 1)},
           getParentId(tv4->axis(0), 1)},
          // 29 -> 22
          {std::unordered_set<Val*>{tv3->axis(2)}, tv4->axis(2)},
          // 31 -> 24
          {std::unordered_set<Val*>{tv3->axis(1)}, tv4->axis(1)},
          // 36, 30, 23 -> 23
          {std::unordered_set<Val*>{tv2->axis(0), tv3->axis(0), tv4->axis(0)},
           tv4->axis(0)},
      };

  checkStep3Results(
      tester.idGraph(IdMappingMode::LOOP),
      tester.s3_loop_promotion_map,
      s3_reference_map);
}

// Test root resolution with the same fusion as Indexing19
TEST_F(IdModelTest, LoopPromotion6) {
  auto fusion = createFusionWithMultipleResolutionPaths();
  FusionGuard fg(fusion.get());
  auto all_tvs = ir_utils::allTvs(fusion.get());

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
            tester.iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
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
            tester.iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
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
            tester.iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
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
            tester.iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
            tester.s1_root_resolution_map);
        break;
      default:
        FAIL() << "Unexpected tensor: " << tv->toString();
    }
  }

  checkStep2Results(
      fusion.get(),
      tester.iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      tester.s2_iel_promotion_map);

  auto id79 =
      getValByName(ir_utils::consumerValsOf(tv9->getRootDomain().at(2)), 79)
          ->as<IterDomain>();
  ASSERT_NE(id79, nullptr) << "IterDomain 79 not found";
  auto id80 =
      getValByName(ir_utils::consumerValsOf(tv9->getRootDomain().at(2)), 80)
          ->as<IterDomain>();
  ASSERT_NE(id80, nullptr) << "IterDomain 80 not found";
  auto id81 = getChildId(id79, 1);
  ASSERT_EQ(id81->name(), 81);
  auto id82 = getChildId(id79, 1, 1);
  ASSERT_EQ(id82->name(), 82);
  auto id83 = getChildId(id80, 1);
  ASSERT_EQ(id83->name(), 83);
  auto id84 = getChildId(id80, 1, 1);
  ASSERT_EQ(id84->name(), 84);
  auto id85 = getChildId(id81, 1);
  ASSERT_EQ(id85->name(), 85);
  auto id86 = getChildId(id81, 1, 1);
  ASSERT_EQ(id86->name(), 86);
  auto id87 = getChildId(id83, 1);
  ASSERT_EQ(id87->name(), 87);
  auto id88 = getChildId(id83, 1, 1);
  ASSERT_EQ(id88->name(), 88);

  // Check Step 3 results. See the design doc for the expected results
  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s3_reference_map = {
          // 1, 2, 3, 46, 6, 7, 41, 8, 9, 10, 35, 36, 11, 12, 61, 15, 16, 56,
          // 17, 18, 29, 30, 79, 80 -> 79
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
               id79,
               id80},
           id80},
          // 71, 47, 42, 37, 62, 57, 31, 81, 83 -> 83
          {std::unordered_set<Val*>{
               getChildId(tv1->getRootDomain().at(0), 1),
               getChildId(tv2->getRootDomain().at(0), 2),
               getChildId(tv4->getRootDomain().at(0), 2),
               getChildId(tv5->getRootDomain().at(0), 3),
               getChildId(tv6->getRootDomain().at(0), 2),
               getChildId(tv8->getRootDomain().at(0), 2),
               getChildId(tv9->getRootDomain().at(0), 3),
               id81,
               id83},
           id83},
          // 73, 49, 44, 39, 64, 59, 33, 85, 87 -> 87
          {std::unordered_set<Val*>{
               tv1->axis(0),
               tv2->axis(0),
               tv4->axis(0),
               tv5->axis(0),
               tv6->axis(0),
               tv8->axis(0),
               tv9->axis(0),
               id85,
               id87},
           id87},
          // 48 -> 43
          {std::unordered_set<Val*>{tv2->axis(2)}, tv4->axis(2)},
          // 50 -> 45
          {std::unordered_set<Val*>{tv2->axis(1)}, tv4->axis(1)},
          // 40, 88 -> 88
          {std::unordered_set<Val*>{tv5->axis(1), id88}, id88},
          // 63 -> 58
          {std::unordered_set<Val*>{tv6->axis(2)}, tv8->axis(2)},
          // 65 -> 60
          {std::unordered_set<Val*>{tv6->axis(1)}, tv8->axis(1)},
          // 34, 86 -> 86
          {std::unordered_set<Val*>{tv9->axis(1), id86}, id86},
          // 38, 84 -> 84
          {std::unordered_set<Val*>{tv5->axis(2), id84}, id84},
          // 32, 82 -> 82 (TODO: update the doc)
          {std::unordered_set<Val*>{tv9->axis(2), id82}, id82},
      };

  checkStep3Results(
      tester.idGraph(IdMappingMode::LOOP),
      tester.s3_loop_promotion_map,
      s3_reference_map);
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
            tester.iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
            tester.s1_root_resolution_map);
        break;
      default:
        FAIL() << "Unexpected tensor: " << tv->toString();
    }
  }

  checkStep2Results(
      &fusion,
      tester.iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      tester.s2_iel_promotion_map);

  // Check Step 3 results. See the design doc for the expected results
  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s3_reference_map = {
          // 3, 4, 5, 14, 6, 7, 8, -> 8
          {std::unordered_set<Val*>{
               tv2->getRootDomain().at(0),
               tv3->getRootDomain().at(0),
               tv3->getRootDomain().at(1),
               getChildId(tv3->getRootDomain().at(0), 1),
               tv4->getRootDomain().at(0),
               tv4->getRootDomain().at(1),
               getChildId(tv4->getRootDomain().at(0), 1)},
           getChildId(tv4->getRootDomain().at(0), 1)},
          // 3, 4, 5, 14, 6, 7, 8, -> 8
          {std::unordered_set<Val*>{
               tv2->getRootDomain().at(0),
               tv3->getRootDomain().at(0),
               tv3->getRootDomain().at(1),
               getChildId(tv3->getRootDomain().at(0), 1),
               tv4->getRootDomain().at(0),
               tv4->getRootDomain().at(1),
               getChildId(tv4->getRootDomain().at(0), 1)},
           getChildId(tv4->getRootDomain().at(0), 1)},
          // 17, 15, 9 -> 9
          {std::unordered_set<Val*>{tv2->axis(0), tv3->axis(0), tv4->axis(0)},
           tv4->axis(0)},
          // 16 -> 10
          {std::unordered_set<Val*>{tv3->axis(1)}, tv4->axis(1)}};

  checkStep3Results(
      tester.idGraph(IdMappingMode::LOOP),
      tester.s3_loop_promotion_map,
      s3_reference_map);
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
            tester.iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
            tester.s1_root_resolution_map);
        break;
      case 5:
        // T5_l[ iS27{2}, iS40{4}, iS41{( ceilDiv(( ( ceilDiv(( 3 * 5 ), 2) ) *
        // 1 ), 4) )} ] ca_pos( 2 ) produce_pos( 1 ) root domain : (iS8{3},
        // iS9{5}, bS10{1})
        validateIELResolution(
            tv->getRootDomain().at(2),
            tv7->getRootDomain().at(2),
            tester.iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
            tester.s1_root_resolution_map);
        break;
      default:
        FAIL() << "Unexpected tensor: " << tv->toString();
    }
  }

  checkStep2Results(
      &fusion,
      tester.iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      tester.s2_iel_promotion_map);

  // Check Step 3 results. See the design doc for the expected results
  std::vector<std::pair<std::unordered_set<Val*>, IterDomain*>>
      s3_reference_map = {
          // 1, 2, 3, 20, 6, 7, 17, 8, 9, 26, 14, 15, 29
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
               getChildId(tv5->getRootDomain().at(0), 1),
               tv7->getRootDomain().at(0),
               tv7->getRootDomain().at(1),
               getChildId(tv7->getRootDomain().at(0), 1)},
           getChildId(tv7->getRootDomain().at(0), 1)},
          // 35, 21, 18, 27, 30 -> 30
          {std::unordered_set<Val*>{
               tv1->axis(0),
               tv2->axis(0),
               tv4->axis(0),
               tv5->axis(0),
               tv7->axis(0)},
           tv7->axis(0)},
          // 28, 10, 39, 31, 16, 42 -> 42
          {std::unordered_set<Val*>{
               getChildId(
                   getChildId(tv5->getRootDomain().at(0), 1), 1, 1), // 28
               tv5->getRootDomain().at(2), // 10
               getChildId(tv5->getRootDomain().at(2), 1), // 39
               getChildId(
                   getChildId(tv7->getRootDomain().at(0), 1), 1, 1), // 31
               tv7->getRootDomain().at(2), // 16
               getChildId(tv7->getRootDomain().at(2), 1)}, // 42
           getChildId(tv7->getRootDomain().at(2), 1)},
          // 22 -> 19
          {std::unordered_set<Val*>{tv2->axis(1)}, tv4->axis(1)},
          // 40, 43 -> 43
          {std::unordered_set<Val*>{tv5->axis(1), tv7->axis(1)}, tv7->axis(1)},
          // 41 -> 44
          {std::unordered_set<Val*>{tv5->axis(2)}, tv7->axis(2)},
      };

  checkStep3Results(
      tester.idGraph(IdMappingMode::LOOP),
      tester.s3_loop_promotion_map,
      s3_reference_map);
}

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
  const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);
  EXPECT_TRUE(
      exact_graph.disjointValSets().strictAreMapped(s0->axis(0), t0->axis(1)));
  EXPECT_TRUE(
      exact_graph.disjointValSets().strictAreMapped(s0->axis(1), t0->axis(0)));
  EXPECT_TRUE(
      exact_graph.disjointValSets().strictAreMapped(s0->axis(2), t0->axis(2)));
}

TEST_F(IdModelTest, PermutedDifferently) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 2, 2, 5});
  TensorView* s0 = slice(in, {0, 0, 0, 0}, {2, 2, 2, 2});
  TensorView* s1 = slice(in, {0, 0, 0, 2}, {2, 2, 2, 5});
  TensorView* t0 = permute(s0, {2, 1, 0, 3});
  TensorView* t1 = permute(s1, {1, 0, 2, 3});
  TensorView* out = cat({t0, t1}, /*dim=*/-1);

  fusion->addInput(in);
  fusion->addOutput(out);

  IdModel id_model(
      fusion.get(), /*build_graphs=*/true, /*allow_self_mapping=*/true);
  const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);
  EXPECT_TRUE(
      exact_graph.disjointValSets().strictAreMapped(s0->axis(2), t0->axis(0)));
  EXPECT_TRUE(
      exact_graph.disjointValSets().strictAreMapped(s0->axis(1), t0->axis(1)));
  EXPECT_TRUE(
      exact_graph.disjointValSets().strictAreMapped(s0->axis(0), t0->axis(2)));
  EXPECT_TRUE(
      exact_graph.disjointValSets().strictAreMapped(s0->axis(3), t0->axis(3)));
  EXPECT_TRUE(
      exact_graph.disjointValSets().strictAreMapped(s1->axis(1), t1->axis(0)));
  EXPECT_TRUE(
      exact_graph.disjointValSets().strictAreMapped(s1->axis(0), t1->axis(1)));
  EXPECT_TRUE(
      exact_graph.disjointValSets().strictAreMapped(s1->axis(2), t1->axis(2)));
  EXPECT_TRUE(
      exact_graph.disjointValSets().strictAreMapped(s1->axis(3), t1->axis(3)));
}

} // namespace nvfuser
