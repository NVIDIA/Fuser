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
#include <id_model/utils.h>
#include <inlining.h>
#include <ops/all_ops.h>
#include <transform_iter.h>

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

  // Returns the IEL graph and the results of Steps 1 and 2
  std::tuple<
      ValGraph,
      std::unordered_map<ValGroup, IterDomain*>,
      std::unordered_map<ValGroup, IterDomain*>>
  getLoopPromotionInfo() {
    // Make sure the depedent graphs are already built
    maybeBuildGraph(IdMappingMode::EXACT);
    maybeBuildGraph(IdMappingMode::PERMISSIVE);

    // Gather broadcast resolution and inlining information
    const StatefulInliningInfo inlining_info = buildStatefulInliningInfo(
        tv_exprs_,
        idGraph(IdMappingMode::EXACT),
        idGraph(IdMappingMode::PERMISSIVE));

    initializeLoopGraph(inlining_info);

    VERBOSE() << "Initial loop graph:\n";
    for (const auto& group :
         idGraph(IdMappingMode::LOOP).disjointValSets().disjointSets()) {
      VERBOSE() << nvfuser::toString(group) << std::endl;
    }

    ValGraph iel_graph = buildIntersection(
        idGraph(IdMappingMode::EXACT), idGraph(IdMappingMode::LOOP), false);

    std::unordered_map<ValGroup, IterDomain*> root_promotion_map =
        buildInlineRootResolutionMap(iel_graph, inlining_info);

    {
      std::stringstream ss;
      ss << "Step 1: Root promotion map\n";
      for (const auto& [iel_group, promoted_id] : root_promotion_map) {
        ss << "\t" << nvfuser::toString(iel_group) << " -> "
           << promoted_id->name() << std::endl;
      }
      VERBOSE() << ss.str();
    }

    auto iel_promotion_map = root_promotion_map;

    propagatePromotionsInIELGraph(iel_graph, iel_promotion_map);

    {
      std::stringstream ss;
      ss << "Step 2: IEL promotion map\n";
      for (const auto& [iel_group, promoted_id] : iel_promotion_map) {
        ss << "\t" << nvfuser::toString(iel_group) << " -> "
           << promoted_id->name() << std::endl;
      }
      VERBOSE() << ss.str();
    }

    return {
        std::move(iel_graph),
        std::move(root_promotion_map),
        std::move(iel_promotion_map)};
  }
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

// Create a simple fusion with outer split. Currently invalid code
// will be generated.
//
// Used as Example 1 in the design doc about Loop
// Promotion Analysis.
std::unique_ptr<Fusion> createFusionWithInlinedOuterSplit() {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({1, 4});
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor({3, 4});
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  fusion.printMath();

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

  return fusion_ptr;
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
    const auto& [iel_graph, root_resolution_map, iel_promotion_map] =
        tester.getLoopPromotionInfo();

    // Nothing inlined. Should be no resolution
    ASSERT_TRUE(root_resolution_map.empty());
  }

  t2->inlineAt(2);
  ASSERT_EQ(t2->getComputeAtPosition(), 2);

  {
    IdModelTester tester(fusion.get());
    const auto& [iel_graph, root_resolution_map, iel_promotion_map] =
        tester.getLoopPromotionInfo();

    // Check Step 1 results
    // t2 is now fully inlined. Its root broadcast domain should be
    // resoled with the corresponding domain of t3
    validateIELResolution(
        t2->getRootDomain().at(0),
        t3->getRootDomain().at(0),
        iel_graph,
        tester.idGraph(IdMappingMode::EXACT),
        root_resolution_map);

    // Check Step 2 results
    // Nothing to propagate in this fusion, so iel_promotion_map
    // should be equivalent to root_resolution_map
    ASSERT_EQ(root_resolution_map, iel_promotion_map)
        << "Unexpected IEL promotion map";
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
  const auto& [iel_graph, root_resolution_map, iel_promotion_map] =
      tester.getLoopPromotionInfo();

  // Check Step 1 results
  // Validate t2 and t3 as they have root broadcast domains
  validateIELResolution(
      t2->getRootDomain().at(0),
      t4->getRootDomain().at(1),
      iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      root_resolution_map);

  validateIELResolution(
      t3->getRootDomain().at(0),
      t4->getRootDomain().at(0),
      iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      root_resolution_map);

  // Check Step 2 results
  // Nothing to propagate in this fusion, so iel_promotion_map
  // should be equivalent to root_resolution_map
  ASSERT_EQ(root_resolution_map, iel_promotion_map)
      << "Unexpected IEL promotion map";
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
  const auto& [iel_graph, root_resolution_map, iel_promotion_map] =
      tester.getLoopPromotionInfo();

  // Check Step 1 results
  // The b1 broadcast domain tv2 should be resolved as it's inlined,
  // but b3 should not.
  validateIELResolution(
      tv2->getRootDomain().at(1),
      tv3->getRootDomain().at(1),
      iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      root_resolution_map);

  validateIELResolution(
      tv2->getRootDomain().at(3),
      nullptr,
      iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      root_resolution_map);

  // Check Step 2 results
  validateIELResolution(
      tv2->axis(0),
      tv3->axis(0),
      iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      iel_promotion_map);

  validateIELResolution(
      tv2->axis(1),
      nullptr,
      iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      iel_promotion_map);
}

// Test root resolution with a fusion with outer split
TEST_F(IdModelTest, LoopPromotion4) {
  auto fusion = createFusionWithInlinedOuterSplit();
  auto all_tvs = ir_utils::allTvs(fusion.get());

  IdModelTester tester(fusion.get());
  const auto& [iel_graph, root_resolution_map, iel_promotion_map] =
      tester.getLoopPromotionInfo();

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
        // T2_l[ iS20{4}, iS21{( ceilDiv(( 1 * 4 ), 4) )} ] ca_pos( 1 )
        //  root domain : (bS4{1}, iS5{4})
        validateIELResolution(
            tv->getRootDomain().at(0),
            findTensorByName(all_tvs, 4)->getRootDomain().at(0),
            iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
            root_resolution_map);
        break;
      default:
        FAIL() << "Unexpected tensor: " << tv->toString();
    }
  }

  //
  checkStep2Results(
      fusion.get(),
      iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      iel_promotion_map);
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
  const auto& [iel_graph, root_resolution_map, iel_promotion_map] =
      tester.getLoopPromotionInfo();

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
            iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
            root_resolution_map);
        break;
      default:
        FAIL() << "Unexpected tensor: " << tv->toString();
    }
  }

  // Check Step 2 results
  checkStep2Results(
      &fusion,
      iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      iel_promotion_map);
}

// Test root resolution with the same fusion as Indexing19
TEST_F(IdModelTest, LoopPromotion6) {
  auto fusion = createFusionWithMultipleResolutionPaths();
  FusionGuard fg(fusion.get());
  auto all_tvs = ir_utils::allTvs(fusion.get());

  IdModelTester tester(fusion.get());
  const auto& [iel_graph, root_resolution_map, iel_promotion_map] =
      tester.getLoopPromotionInfo();

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
            findTensorByName(all_tvs, 4)->getRootDomain().at(1),
            iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
            root_resolution_map);
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
            findTensorByName(all_tvs, 9)->getRootDomain().at(2),
            iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
            root_resolution_map);
        break;
      case 6:
        // T6_l[ iS64{( ceilDiv(( ceilDiv(( 7 * 1 ), 5) ), 3) )}, iS65{3},
        // iS63{5} ] ca_pos( 1 ) produce_pos( 1 )
        //  root domain : (iS11{7}, bS12{1})
        // Resolution: Resolved by the immediate consumer (T8)
        validateIELResolution(
            tv->getRootDomain().at(1),
            findTensorByName(all_tvs, 8)->getRootDomain().at(1),
            iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
            root_resolution_map);
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
            findTensorByName(all_tvs, 5)->getRootDomain().at(1),
            iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
            root_resolution_map);
        break;
      default:
        FAIL() << "Unexpected tensor: " << tv->toString();
    }
  }

  checkStep2Results(
      fusion.get(),
      iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      iel_promotion_map);
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
  const auto& [iel_graph, root_resolution_map, iel_promotion_map] =
      tester.getLoopPromotionInfo();

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
            findTensorByName(all_tvs, 4)->getRootDomain().at(0),
            iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
            root_resolution_map);
        break;
      default:
        FAIL() << "Unexpected tensor: " << tv->toString();
    }
  }

  checkStep2Results(
      &fusion,
      iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      iel_promotion_map);
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
  const auto& [iel_graph, root_resolution_map, iel_promotion_map] =
      tester.getLoopPromotionInfo();

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
            findTensorByName(all_tvs, 7)->getRootDomain().at(0),
            iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
            root_resolution_map);
        break;
      case 5:
        // T5_l[ iS27{2}, iS40{4}, iS41{( ceilDiv(( ( ceilDiv(( 3 * 5 ), 2) ) *
        // 1 ), 4) )} ] ca_pos( 2 ) produce_pos( 1 ) root domain : (iS8{3},
        // iS9{5}, bS10{1})
        validateIELResolution(
            tv->getRootDomain().at(2),
            findTensorByName(all_tvs, 7)->getRootDomain().at(2),
            iel_graph,
            tester.idGraph(IdMappingMode::EXACT),
            root_resolution_map);
        break;
      default:
        FAIL() << "Unexpected tensor: " << tv->toString();
    }
  }

  checkStep2Results(
      &fusion,
      iel_graph,
      tester.idGraph(IdMappingMode::EXACT),
      iel_promotion_map);
}

} // namespace nvfuser
