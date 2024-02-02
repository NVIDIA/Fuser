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

#include <fusion.h>
#include <id_model/id_model.h>
#include <id_model/to_string.h>
#include <ops/all_ops.h>
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
      [&]() { IdModel id_model(&fusion); },
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

TensorView* getTensorByName(
    const std::vector<TensorView*>& tvs,
    StmtNameType name) {
  if (auto it = std::find_if(
          tvs.begin(),
          tvs.end(),
          [&](TensorView* tv) { return tv->name() == name; });
      it != tvs.end()) {
    return *it;
  } else {
    return nullptr;
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
        << "Expected: " << nvfuser::toString(graph.toGroup(ref_expr))
        << ". Actual: " << nvfuser::toString(eg);
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

  // No ID expr yet
  {
    IdModel id_model(&fusion);
    const ValGraph& vg = id_model.idGraph(IdMappingMode::EXACT);
    ValGraphStmtSort vg_stmt_sort(vg);
    checkSortingResults(vg, vg_stmt_sort.exprs(), vg_stmt_sort.vals(), {});
  }

  tv2->merge(0)->split(0, 4);

  TransformPropagator propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

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

  // Merge adn split by one. The split input and output will be mapped.
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

  IdModel id_model(fusion.get(), true, false, false);

  const ValGraph& vg = id_model.idGraph(IdMappingMode::EXACT);

  ValGraphStmtSort vg_stmt_sort(vg);

  auto tv1 = getTensorByName(all_tvs, 1);
  auto tv2 = getTensorByName(all_tvs, 2);
  auto tv4 = getTensorByName(all_tvs, 4);
  auto tv5 = getTensorByName(all_tvs, 5);
  auto tv6 = getTensorByName(all_tvs, 6);
  auto tv8 = getTensorByName(all_tvs, 8);
  auto tv9 = getTensorByName(all_tvs, 9);
  auto tv10 = getTensorByName(all_tvs, 10);

  // Expected reference order:
  //
  // exprg{39}: Merge: iS2{7} and bS3{1} -> iS46{( 7 * 1 )}
  // exprg{57}: Merge: iS11{7} and bS12{1} -> iS61{( 7 * 1 )}
  // exprg{17}: Merge: iS17{7} and bS18{1} -> iS29{( 7 * 1 )}
  // exprg{69 73 89}: Split: iS1{7} by factor 5 -> iS71{( ceilDiv(7, 5) )},
  // iS72{5}, start offset: 0, stop offset: 0 exprg{51 63 93}: Merge: iS15{7}
  // and iS16{13} -> iS56{( 7 * 13 )} exprg{9 25 33 45 91 95}: Merge: iS20{7}
  // and iS21{11} -> iS23{( 7 * 11 )} exprg{27}: Merge: iS35{( 7 * 11 )} and
  // bS10{1} -> iS36{( ( 7 * 11 ) * 1 )} exprg{19}: Merge: iS29{( 7 * 1 )} and
  // iS19{13} -> iS30{( ( 7 * 1 ) * 13 )} exprg{11 77 79 99}: Merge: iS23{( 7 *
  // 11 )} and iS22{13} -> iS24{( ( 7 * 11 ) * 13 )} exprg{41}: Split: iS46{( 7
  // * 1 )} by factor 5 -> iS47{( ceilDiv(( 7 * 1 ), 5) )}, iS48{5}, start
  // offset: 0, stop offset: 0 exprg{59}: Split: iS61{( 7 * 1 )} by factor 5 ->
  // iS62{( ceilDiv(( 7 * 1 ), 5) )}, iS63{5}, start offset: 0, stop offset: 0
  // exprg{71 75 101}: Split: iS71{( ceilDiv(7, 5) )} by factor 3 -> iS73{(
  // ceilDiv(( ceilDiv(7, 5) ), 3) )}, iS74{3}, start offset: 0, stop offset: 0
  // exprg{53 65 109}: Split: iS56{( 7 * 13 )} by factor 5 -> iS57{( ceilDiv(( 7
  // * 13 ), 5) )}, iS58{5}, start offset: 0, stop offset: 0 exprg{35 47 105}:
  // Split: iS41{( 7 * 11 )} by factor 5 -> iS42{( ceilDiv(( 7 * 11 ), 5) )},
  // iS43{5}, start offset: 0, stop offset: 0 exprg{29}: Split: iS36{( ( 7 * 11
  // ) * 1 )} by factor 5 -> iS37{( ceilDiv(( ( 7 * 11 ) * 1 ), 5) )}, iS38{5},
  // start offset: 0, stop offset: 0 exprg{21}: Split: iS30{( ( 7 * 1 ) * 13 )}
  // by factor 5 -> iS31{( ceilDiv(( ( 7 * 1 ) * 13 ), 5) )}, iS32{5}, start
  // offset: 0, stop offset: 0 exprg{13 81 83 97 103 107 111 115 117 119 121}:
  // Split: iS24{( ( 7 * 11 ) * 13 )} by factor 5 -> iS25{( ceilDiv(( ( 7 * 11 )
  // * 13 ), 5) )}, iS26{5}, start offset: 0, stop offset: 0 exprg{43}: Split:
  // iS47{( ceilDiv(( 7 * 1 ), 5) )} by factor 3 -> iS49{( ceilDiv(( ceilDiv(( 7
  // * 1 ), 5) ), 3) )}, iS50{3}, start offset: 0, stop offset: 0 exprg{61}:
  // Split: iS62{( ceilDiv(( 7 * 1 ), 5) )} by factor 3 -> iS64{( ceilDiv((
  // ceilDiv(( 7 * 1 ), 5) ), 3) )}, iS65{3}, start offset: 0, stop offset: 0
  // exprg{55 67 129}: Split: iS57{( ceilDiv(( 7 * 13 ), 5) )} by factor 3 ->
  // iS59{( ceilDiv(( ceilDiv(( 7 * 13 ), 5) ), 3) )}, iS60{3}, start offset: 0,
  // stop offset: 0 exprg{37 49 125}: Split: iS42{( ceilDiv(( 7 * 11 ), 5) )} by
  // factor 3 -> iS44{( ceilDiv(( ceilDiv(( 7 * 11 ), 5) ), 3) )}, iS45{3},
  // start offset: 0, stop offset: 0 exprg{31}: Split: iS37{( ceilDiv(( ( 7 * 11
  // ) * 1 ), 5) )} by factor 3 -> iS39{( ceilDiv(( ceilDiv(( ( 7 * 11 ) * 1 ),
  // 5) ), 3) )}, iS40{3}, start offset: 0, stop offset: 0 exprg{23}: Split:
  // iS31{( ceilDiv(( ( 7 * 1 ) * 13 ), 5) )} by factor 3 -> iS33{( ceilDiv((
  // ceilDiv(( ( 7 * 1 ) * 13 ), 5) ), 3) )}, iS34{3}, start offset: 0, stop
  // offset: 0 exprg{15 85 87 113 123 127 131 133 135 137 139}: Split: iS25{(
  // ceilDiv(( ( 7 * 11 ) * 13 ), 5) )} by factor 3 -> iS27{( ceilDiv((
  // ceilDiv(( ( 7 * 11 ) * 13 ), 5) ), 3) )}, iS28{3}, start offset: 0, stop
  // offset: 0

  std::vector<Expr*> ref_order;
  ref_order.push_back(getParentExpr(tv2->axis(0), 3));
  ref_order.push_back(getParentExpr(tv6->axis(0), 3));
  ref_order.push_back(getParentExpr(tv9->axis(0), 4));
  ref_order.push_back(getParentExpr(tv1->axis(0), 2));
  ref_order.push_back(getParentExpr(tv8->axis(0), 3));
  ref_order.push_back(getParentExpr(tv10->axis(0), 4));
  ref_order.push_back(getParentExpr(tv5->axis(0), 3));
  ref_order.push_back(getParentExpr(tv9->axis(0), 3));
  ref_order.push_back(getParentExpr(tv10->axis(0), 3));
  ref_order.push_back(getParentExpr(tv2->axis(0), 2));
  ref_order.push_back(getParentExpr(tv6->axis(0), 2));
  ref_order.push_back(getParentExpr(tv1->axis(0), 1));
  ref_order.push_back(getParentExpr(tv8->axis(0), 2));
  ref_order.push_back(getParentExpr(tv4->axis(0), 2));
  ref_order.push_back(getParentExpr(tv5->axis(0), 2));
  ref_order.push_back(getParentExpr(tv9->axis(0), 2));
  ref_order.push_back(getParentExpr(tv10->axis(0), 2));
  ref_order.push_back(getParentExpr(tv2->axis(0), 1));
  ref_order.push_back(getParentExpr(tv6->axis(0), 1));
  ref_order.push_back(getParentExpr(tv8->axis(0), 1));
  ref_order.push_back(getParentExpr(tv4->axis(0), 1));
  ref_order.push_back(getParentExpr(tv5->axis(0), 1));
  ref_order.push_back(getParentExpr(tv9->axis(0), 1));
  ref_order.push_back(getParentExpr(tv10->axis(0), 1));

  checkSortingResults(vg, vg_stmt_sort.exprs(), vg_stmt_sort.vals(), ref_order);
}

} // namespace nvfuser
