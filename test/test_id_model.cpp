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
#include <inlining.h>
#include <ops/all_ops.h>

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

// Helper class to test IdModel
class IdModelTester : public IdModel {
 public:
  // Do not automatically build the graphs
  IdModelTester(Fusion* fusion) : IdModel(fusion, /* build_graphs */ false) {}

  std::pair<ValGraph, std::unordered_map<ValGroup, IterDomain*>>
  getInlineRootResolutionMap() {
    // Make sure the depedent graphs are already built
    maybeBuildGraph(IdMappingMode::EXACT);
    maybeBuildGraph(IdMappingMode::PERMISSIVE);

    // Gather broadcast resolution and inlining information
    const StatefulInliningInfo inlining_info = buildStatefulInliningInfo(
        tv_exprs_,
        idGraph(IdMappingMode::EXACT),
        idGraph(IdMappingMode::PERMISSIVE));

    initializeLoopGraph(inlining_info);

    ValGraph iel_graph = buildIntersection(
        idGraph(IdMappingMode::EXACT), idGraph(IdMappingMode::LOOP), false);

    std::unordered_map<ValGroup, IterDomain*> root_promotion_map =
        buildInlineRootResolutionmap(iel_graph, inlining_info);

    return {std::move(iel_graph), std::move(root_promotion_map)};
  }
};

// Test if root_broadcast_id is resolved to ref_id. If ref_id is
// nullptr, test if root_broadcast_id has no resolution.
void validateResolution(
    IterDomain* root_broadcast_id,
    IterDomain* ref_id,
    const ValGraph& iel_graph,
    const std::unordered_map<ValGroup, IterDomain*>& root_resolution_map) {
  ASSERT_TRUE(root_broadcast_id->isBroadcast());
  const auto& iel_group = iel_graph.toGroup(root_broadcast_id);
  auto root_promotion_map_it = root_resolution_map.find(iel_group);
  if (ref_id != nullptr) {
    ASSERT_TRUE(root_promotion_map_it != root_resolution_map.end())
        << "Root resolution not found for: " << nvfuser::toString(iel_group);
    ASSERT_FALSE(ref_id->isBroadcast());
    auto resolution_id = root_promotion_map_it->second;
    ASSERT_TRUE(
        iel_graph.disjointValSets().strictAreMapped(resolution_id, ref_id))
        << "Unexpected root resolution. "
        << "Expected: " << ref_id->toString()
        << ". Actual: " << resolution_id->toString();
  } else {
    ASSERT_TRUE(root_promotion_map_it == root_resolution_map.end())
        << "Root resolution should not exist for: "
        << nvfuser::toString(iel_group)
        << ", but found: " << root_promotion_map_it->second->toString();
  }
}

// Create a fusion where we're missing a valid concrete id so the compute at map
// processing will fail. We need to be able to create the concrete ID not just
// look for one. It is not yet possible to lower this fusion as the
// current indexing cannot generate correct indices. Also used in
// FusionIndeixing19
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

TensorView* findTensorByName(
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

} // namespace

// Testing root resolution with a simple broadcast pattern
TEST_F(IdModelTest, LoopGraphRootResolution1) {
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
    const auto& [iel_graph, root_resolution_map] =
        tester.getInlineRootResolutionMap();

    // Nothing inlined. Should be no resolution
    ASSERT_TRUE(root_resolution_map.empty());
  }

  t2->inlineAt(2);
  ASSERT_EQ(t2->getComputeAtPosition(), 2);

  {
    IdModelTester tester(fusion.get());
    const auto& [iel_graph, root_resolution_map] =
        tester.getInlineRootResolutionMap();

    // t2 is now fully inlined. Its root broadcast domain should be
    // resoled with the corresponding domain of t3
    validateResolution(
        t2->getRootDomain().at(0),
        t3->getRootDomain().at(0),
        iel_graph,
        root_resolution_map);
  }
}

// Test with a fusion with progressive broadcasting
TEST_F(IdModelTest, LoopGraphRootResolution2) {
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
  const auto& [iel_graph, root_resolution_map] =
      tester.getInlineRootResolutionMap();

  // Validate t2 and t3 as they have root broadcast domains
  validateResolution(
      t2->getRootDomain().at(0),
      t4->getRootDomain().at(1),
      iel_graph,
      root_resolution_map);

  validateResolution(
      t3->getRootDomain().at(0),
      t4->getRootDomain().at(0),
      iel_graph,
      root_resolution_map);
}

// Multiple inlined and non-inlined broadcast domains
TEST_F(IdModelTest, LoopGraphRootResolution3) {
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
  const auto& [iel_graph, root_resolution_map] =
      tester.getInlineRootResolutionMap();

  // The b1 broadcast domain tv2 should be resolved as it's inlined,
  // but b3 should not.
  validateResolution(
      tv2->getRootDomain().at(1),
      tv3->getRootDomain().at(1),
      iel_graph,
      root_resolution_map);

  validateResolution(
      tv2->getRootDomain().at(3), nullptr, iel_graph, root_resolution_map);
}

TEST_F(IdModelTest, LoopGraphRootResolution4) {
  auto fusion = createFusionWithMultipleResolutionPaths();
  auto all_tvs = ir_utils::allTvs(fusion.get());

  IdModelTester tester(fusion.get());
  const auto& [iel_graph, root_resolution_map] =
      tester.getInlineRootResolutionMap();

  // Verify all tensors with broadcast have correct resolution of root
  // broadcast domains
  for (auto tv : ir_utils::allTvs(fusion.get())) {
    // Skip tensors with no broadcast
    if (std::none_of(
            tv->getRootDomain().begin(),
            tv->getRootDomain().end(),
            [](auto id) { return id->isBroadcast(); })) {
      continue;
    }

    switch (tv->name()) {
      case 2:
        // T2_l[ iS49{( ceilDiv(( ceilDiv(( 7 * 1 ), 5) ), 3) )}, iS50{3},
        // iS48{5} ] ca_pos( 1 ) produce_pos( 1 )
        //  root domain : (iS2{7}, bS3{1})
        // Resolution: Resolved by the immediate consumer (T4)
        validateResolution(
            tv->getRootDomain().at(1),
            findTensorByName(all_tvs, 4)->getRootDomain().at(1),
            iel_graph,
            root_resolution_map);
        break;
      case 5:
        // T5_l[ iS39{( ceilDiv(( ceilDiv(( ( 7 * 11 ) * 1 ), 5) ), 3) )},
        // iS40{3}, iS38{5} ] produce_pos( 1 )
        //  root domain : (iS8{7}, iS9{11}, bS10{1})
        // Resolution: T5 is not inlined to the immediate consumer,
        // T10. Resolution is done with the other path from T1, such
        // as T8 or T9.
        validateResolution(
            tv->getRootDomain().at(2),
            findTensorByName(all_tvs, 9)->getRootDomain().at(2),
            iel_graph,
            root_resolution_map);
        break;
      case 6:
        // T6_l[ iS64{( ceilDiv(( ceilDiv(( 7 * 1 ), 5) ), 3) )}, iS65{3},
        // iS63{5} ] ca_pos( 1 ) produce_pos( 1 )
        //  root domain : (iS11{7}, bS12{1})
        // Resolution: Resolved by the immediate consumer (T8)
        validateResolution(
            tv->getRootDomain().at(1),
            findTensorByName(all_tvs, 8)->getRootDomain().at(1),
            iel_graph,
            root_resolution_map);
        break;
      case 9:
        // T9_l[ iS33{( ceilDiv(( ceilDiv(( ( 7 * 1 ) * 13 ), 5) ), 3) )},
        // iS34{3}, iS32{5} ] produce_pos( 1 )
        //  root domain : (iS17{7}, bS18{1}, iS19{13})
        // Resolution: T9 is not inlined to the immediate consumer,
        // T10. Resolution is done with the other path from T1, such
        // as T4 or T5
        validateResolution(
            tv->getRootDomain().at(1),
            findTensorByName(all_tvs, 5)->getRootDomain().at(1),
            iel_graph,
            root_resolution_map);
        break;
      default:
        FAIL() << "Unexpected tensor: " << tv->toString();
    }
  }
}

} // namespace nvfuser
