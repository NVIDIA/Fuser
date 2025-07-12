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

#include <device_lower/utils.h>
#include <fusion.h>
#include <ops/all_ops.h>
#include <runtime/executor_utils.h>
#include <scheduler/tools/abstract_tensor.h>
#include <scheduler/utils.h>
#include <scheduler/vectorize_helper.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <utils.h>

#include <cstdlib>
#include <filesystem>
#include <forward_list>
#include <fstream>
#include <list>
#include <random>
#include <ranges>
#include <system_error>
#include <vector>

namespace nvfuser {

int myFavoriteFunction(int a, int b) {
  DEBUG_PRINT_SCOPE(a, b);
  if (a > 0) {
    RECORD_AND_RETURN(a + b);
  } else {
    RECORD_AND_RETURN(a - b);
  }
}

TEST_F(NVFuserTest, FunctionTrace1) {
#ifndef NDEBUG
  std::stringstream ss;
  DebugStreamGuard g(ss);
  DebugDumpOptionsGuard gg;
  gg.getCurOptions().set(DebugDumpOption::FunctionTrace, {".*Favorite.*"});
  EXPECT_EQ(myFavoriteFunction(1, 2), 3);
  EXPECT_THAT(
      ss.str(), ::testing::HasSubstr("Entering myFavoriteFunction(1, 2)"));
  EXPECT_THAT(
      ss.str(),
      ::testing::HasSubstr("Leaving myFavoriteFunction returning 3 at "));
  EXPECT_THAT(ss.str(), ::testing::HasSubstr("test_utils.cpp:32"));
#else
  GTEST_SKIP() << "Test only runs in debug mode";
#endif
}

TEST_F(NVFuserTest, FunctionTrace2) {
#ifndef NDEBUG
  std::stringstream ss;
  DebugStreamGuard g(ss);
  DebugDumpOptionsGuard gg;
  gg.getCurOptions().set(DebugDumpOption::FunctionTrace, {".*Favorite.*"});
  EXPECT_EQ(myFavoriteFunction(-1, 2), -3);
  EXPECT_THAT(
      ss.str(), ::testing::HasSubstr("Entering myFavoriteFunction(-1, 2)"));
  EXPECT_THAT(
      ss.str(),
      ::testing::HasSubstr("Leaving myFavoriteFunction returning -3 at "));
  EXPECT_THAT(ss.str(), ::testing::HasSubstr("test_utils.cpp:34"));
#else
  GTEST_SKIP() << "Test only runs in debug mode";
#endif
}

TEST_F(NVFuserTest, FusionSplitDims) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto p = prime_number;
  auto tv = makeConcreteTensor(
      {p(0) * p(1) * p(2), p(3), p(4), p(5) * p(6), p(7), p(8), p(9) * p(10)});
  std::vector<int64_t> dims{0, 1, 2, 3, 4, 5, 6};
  scheduler_utils::splitDims(
      tv, {{0, p(2)}, {0, p(1)}, {3, p(6)}, {6, p(10)}}, dims);
  EXPECT_EQ(tv->nDims(), 11);
  for (auto i : arange(11)) {
    EXPECT_EQ(tv->axis(i)->extent()->evaluate(), p(i));
  }
  std::vector<int64_t> expect{0, 3, 4, 5, 7, 8, 9};
  EXPECT_EQ(dims, expect);
}

TEST_F(NVFuserTest, FusionMergeDims) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto p = prime_number;
  auto tv = makeConcreteTensor(
      {p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9), p(10)});
  std::vector<int64_t> dims{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int64_t> to_merge{3, 2, 9, 7, 8};
  auto merged = scheduler_utils::mergeDims(tv, to_merge, dims);
  EXPECT_EQ(merged, 2);
  std::vector<int64_t> expect_shape{
      p(0), p(1), p(2) * p(3) * p(7) * p(8) * p(9), p(4), p(5), p(6), p(10)};
  EXPECT_EQ(tv->nDims(), expect_shape.size());
  for (auto i : arange(expect_shape.size())) {
    EXPECT_EQ(tv->axis(i)->extent()->evaluate(), expect_shape[i]);
  }
  std::vector<int64_t> expect_dims{0, 1, 2, 2, 3, 4, 5, 2, 2, 2, 6};
  EXPECT_EQ(dims, expect_dims);
  auto logical_domain = tv->getLogicalDomain();
  auto num_merged_dim = to_merge.size();
  auto inputs = IterVisitor::getInputsTo({tv->axis(2)});
  for (auto index : arange(num_merged_dim)) {
    EXPECT_TRUE(logical_domain[to_merge[num_merged_dim - 1 - index]]->sameAs(
        inputs[index]));
  }
}

TEST_F(NVFuserTest, FusionReorderAsRFactor) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int a = 1, b = 2, c = 3, d = 4;

  TensorView* tv0 = makeConcreteTensor({a, b, c, d});
  fusion.addInput(tv0);
  fusion.addOutput(tv0);

  // [a, b, c, d]
  tv0->merge(0, 2);
  // [a*c, b, d]
  tv0->split(1, 2);
  // [a*c, bo, bi, d]
  tv0->split(3, 3);
  // [a*c, bo, bi, do, di]
  tv0->reorder({{1, 4}, {2, 1}, {3, 3}, {4, 2}});
  // [a*c, bi, di, do, bo]
  tv0->merge(3);
  tv0->merge(1);
  // [a*c, bi*di, do*bo]
  tv0->reorder({{0, 2}});
  // [bi*di, do*bo, a*c]
  // Order we want is:
  // [a*c, do*bo, bi*di]
  auto old2new = scheduler_utils::domainReorderAsLogicalMap(tv0);
  EXPECT_EQ(old2new[0], 2);
  EXPECT_EQ(old2new[1], 1);
  EXPECT_EQ(old2new[2], 0);
}

TEST_F(NVFuserTest, FusionDisjointViewSet) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion->addInput(tv0);

  auto tv1 = reshape(tv0, {2, 3, 4}, {2, 12});

  auto tv2 = makeConcreteTensor({2, 12});
  fusion->addInput(tv2);

  auto tv3 = add(tv2, tv1);
  fusion->addOutput(tv3);

  auto disjoint_exact = scheduler_utils::disjointLogicalSets(fusion.get());

  NVF_ERROR(disjoint_exact.strictAreMapped(tv0->axis(1), tv0->axis(2)));
}

TEST_F(NVFuserTest, FusionBroadcastViewMultiples) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int a = 2, b = 3, c = 5, d = 7, e = 11, f = 13;

  auto tv0 = makeConcreteTensor({a, b, c, d, e, f});
  fusion.addInput(tv0);

  // tie e and f together (swapping values next to eachother enforces they'll be
  // merged then split by reshape)
  auto tv1 = reshape(tv0, {a, b, c, d, e, f}, {a, b, c, d, f, e});
  fusion.addOutput(tv1);

  // swap d and e
  auto tv2 = transpose(tv1, 3, 4);
  // tie c and e together
  auto tv3 = reshape(tv2, {a, b, c, e, d, f}, {a, b, e, c, d, f});

  fusion.addOutput(tv3);

  auto tv4 = set(tv0);
  // Use tv4 as the reference
  fusion.addOutput(tv4);

  // a, b, d aren't tied to anything so they are valid broadcasts from the
  // perspective of broadcast multiples analysis.
  auto tv5 = makeConcreteTensor({1, 1, c, 1, e, f});
  fusion.addInput(tv5);

  // c, e, and f are tied together so this shouldn't be counted as a broadcast
  // dim in the reference since it's a partial bcast
  auto tv6 = makeConcreteTensor({a, b, c, 1, 1, 1});
  fusion.addInput(tv6);

  // c, e, and f are tied together this should be counted as a broadcast dim in
  // the reference since it's a partial bcast
  auto tv7 = makeConcreteTensor({a, b, 1, 1, 1, 1});
  fusion.addInput(tv7);

  // plug the broadcasts into the fusion
  auto tv8 = add(tv5, tv4);
  auto tv9 = add(tv6, tv8);
  auto tv10 = add(tv7, tv9);
  fusion.addOutput(tv10);

  auto bcast_info =
      scheduler_utils::getBroadcastMultiples(tv4, DataType::Int32);

  // linked c, e, and f together so they should have the same id.
  EXPECT_EQ(bcast_info.view_disjoint_set_ids[5], 0);
  EXPECT_EQ(bcast_info.view_disjoint_set_ids[4], 0);
  EXPECT_EQ(bcast_info.view_disjoint_set_ids[3], 1);
  EXPECT_EQ(bcast_info.view_disjoint_set_ids[2], 0);
  EXPECT_EQ(bcast_info.view_disjoint_set_ids[1], 2);
  EXPECT_EQ(bcast_info.view_disjoint_set_ids[0], 3);

  EXPECT_TRUE(
      scheduler_utils::breakIsDisjoint(bcast_info.view_disjoint_set_ids, 0));
  EXPECT_TRUE(
      scheduler_utils::breakIsDisjoint(bcast_info.view_disjoint_set_ids, 1));
  EXPECT_TRUE(
      scheduler_utils::breakIsDisjoint(bcast_info.view_disjoint_set_ids, 2));
  EXPECT_TRUE(
      !scheduler_utils::breakIsDisjoint(bcast_info.view_disjoint_set_ids, 3));
  EXPECT_TRUE(
      !scheduler_utils::breakIsDisjoint(bcast_info.view_disjoint_set_ids, 4));
  EXPECT_TRUE(
      !scheduler_utils::breakIsDisjoint(bcast_info.view_disjoint_set_ids, 5));

  // tv0  [a, b, c, d, e, f]
  // tv1  [a, b, c, d, e, f]
  // tv3  [a, b, c, d, e, f]
  // tv4  [a, b, c, d, e, f]
  // tv5  [1, 1, c, 1, e, f] -> Left bcasts should show up in some multiples
  // tv6  [a, b, c, 1, 1, 1] -> reshape interferes with bcasts, non of these
  // should
  //                            show up
  // tv7  [a, b, 1, 1, 1, 1] -> These broadcasts could be recognized
  // tv10 [a, b, c, d, e, f]

  // Units are in bits

  EXPECT_EQ(bcast_info.broadcast_multiples[0].lhs_multiple, 0);
  EXPECT_EQ(bcast_info.broadcast_multiples[0].rhs_multiple, 8 * 4 * 8);

  EXPECT_EQ(bcast_info.broadcast_multiples[1].lhs_multiple, 7 * 4 * 8);
  EXPECT_EQ(bcast_info.broadcast_multiples[1].rhs_multiple, 8 * 4 * 8);

  EXPECT_EQ(bcast_info.broadcast_multiples[2].lhs_multiple, 7 * 4 * 8);
  EXPECT_EQ(bcast_info.broadcast_multiples[2].rhs_multiple, 7 * 4 * 8);

  EXPECT_EQ(bcast_info.broadcast_multiples[3].lhs_multiple, 8 * 4 * 8);
  EXPECT_EQ(bcast_info.broadcast_multiples[3].rhs_multiple, 7 * 4 * 8);

  EXPECT_EQ(bcast_info.broadcast_multiples[4].lhs_multiple, 8 * 4 * 8);
  EXPECT_EQ(bcast_info.broadcast_multiples[4].rhs_multiple, 7 * 4 * 8);

  EXPECT_EQ(bcast_info.broadcast_multiples[5].lhs_multiple, 8 * 4 * 8);
  EXPECT_EQ(bcast_info.broadcast_multiples[5].rhs_multiple, 7 * 4 * 8);
}

TEST_F(NVFuserTest, FusionTVDomainGuard) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<std::optional<bool>> all_true = {true, true};
  std::vector<std::optional<bool>> all_false = {false, false};
  std::vector<std::optional<bool>> false_true = {false, true};
  auto tv = TensorViewBuilder().ndims(2).contiguity(false_true).build();
  EXPECT_EQ(tv->domain()->contiguity(), false_true);
  {
    auto guard = ir_utils::overrideContiguityGuard(tv, true);
    EXPECT_EQ(tv->domain()->contiguity(), all_true);
  }
  EXPECT_EQ(tv->domain()->contiguity(), false_true);
  {
    auto guard = ir_utils::overrideContiguityGuard(tv, false);
    EXPECT_EQ(tv->domain()->contiguity(), all_false);
  }
  EXPECT_EQ(tv->domain()->contiguity(), false_true);
  {
    auto guard1 = ir_utils::overrideContiguityGuard(tv, true);
    auto guard2 = std::move(guard1);
    EXPECT_EQ(tv->domain()->contiguity(), all_true);
  }
  EXPECT_EQ(tv->domain()->contiguity(), false_true);
}

class VectorizeHelperTest : public NVFuserTest {};

// Test simple backward mapping through split
TEST_F(VectorizeHelperTest, BackwardMapper1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2 * 3});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {2 * 3}, {2, 3});
  fusion.addOutput(tv1);

  {
    // No mappings
    auto mapper =
        vectorize_helper::ContiguousInnerDimensionsMapper::map(tv1, {});

    EXPECT_TRUE(
        !mapper.hasMappedDims(tv0) || mapper.mappedLogicalIds(tv0).empty());
    EXPECT_TRUE(
        !mapper.hasMappedDims(tv1) || mapper.mappedLogicalIds(tv1).empty());
  }

  {
    // Inner mapping partial propogates
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv1, {tv1->axis(1)});

    EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(0)));
    EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(1)));

    EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluate(), 3);
  }

  {
    // Full mapping
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv1, {tv1->axis(0), tv1->axis(1)});

    EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(0)));
    EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(0)));
    EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[1]->sameAs(tv1->axis(1)));

    EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluate(), 2);
    EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(1))->evaluate(), 3);

    EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluate(), 2 * 3);
  }
}

// Test backward mapping through multiple splits
TEST_F(VectorizeHelperTest, BackwardMapper2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2 * 3 * 4});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {2 * 3 * 4}, {2 * 3, 4});
  auto tv2 = reshape(tv1, {2 * 3, 4}, {2, 3, 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv2, {tv2->axis(1), tv2->axis(2)});

  EXPECT_THAT(
      [&]() { mapper.getProjectedExtent(tv2->axis(0)); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Not projected")));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluate(), 3);
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(2))->evaluate(), 4);
  // Inner dim fully maps, outer dim of split partially maps
  EXPECT_EQ(mapper.mappedLogicalIds(tv2).size(), 2);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[0]->sameAs(tv2->axis(1)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[1]->sameAs(tv2->axis(2)));

  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluate(), 3);
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(1))->evaluate(), 4);
  // Inner dim fully maps, outer dim of split partially maps
  EXPECT_EQ(mapper.mappedLogicalIds(tv1).size(), 2);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[1]->sameAs(tv1->axis(1)));

  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluate(), 4 * 3);
  EXPECT_EQ(mapper.mappedLogicalIds(tv0).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(0)));
}

// Test backward mapping through multiple splits
TEST_F(VectorizeHelperTest, BackwardMapper3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2 * 3 * 4});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {2 * 3 * 4}, {2, 3 * 4});
  auto tv2 = reshape(tv1, {2, 3 * 4}, {2, 3, 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv2, {tv2->axis(2)});

  // Partial map forwarding
  EXPECT_EQ(mapper.mappedLogicalIds(tv0).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(0)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluate(), 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(1))->evaluate(), 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv2).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[0]->sameAs(tv2->axis(2)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(2))->evaluate(), 4);
}

// Test simple backward mapping through merge
TEST_F(VectorizeHelperTest, BackwardMapper4) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {2, 3}, {2 * 3});
  fusion.addOutput(tv1);

  {
    // No mapping
    auto mapper =
        vectorize_helper::ContiguousInnerDimensionsMapper::map(tv1, {});

    EXPECT_TRUE(
        !mapper.hasMappedDims(tv0) || mapper.mappedLogicalIds(tv0).empty());
    EXPECT_TRUE(
        !mapper.hasMappedDims(tv1) || mapper.mappedLogicalIds(tv1).empty());
  }

  {
    // Full merge mapping
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv1, {tv1->axis(0)});

    EXPECT_EQ(mapper.mappedLogicalIds(tv0).size(), 2);
    EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(0)));
    EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[1]->sameAs(tv0->axis(1)));
    EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluate(), 2);
    EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluate(), 3);

    EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluate(), 2 * 3);
  }
}

// Test symbolic partial mapping through merge
TEST_F(VectorizeHelperTest, BackwardMapper5) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {2 * 3, 4}, {2 * 3 * 4});
  auto tv2 = reshape(tv1, {2 * 3 * 4}, {2, 3 * 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv2, {tv2->axis(1)});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor inp = at::randn({2 * 3, 4}, options);

  KernelArgumentHolder args({inp});
  auto expr_eval = executor_utils::bindInputs(args, &fusion);

  EXPECT_EQ(mapper.mappedLogicalIds(tv0).size(), 2);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(0)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[1]->sameAs(tv0->axis(1)));
  EXPECT_EQ(
      expr_eval.evaluate(mapper.getProjectedExtent(tv0->axis(0))).as<int64_t>(),
      3);
  EXPECT_EQ(
      expr_eval.evaluate(mapper.getProjectedExtent(tv0->axis(1))).as<int64_t>(),
      4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_EQ(
      expr_eval.evaluate(mapper.getProjectedExtent(tv1->axis(0))).as<int64_t>(),
      3 * 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv2).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[0]->sameAs(tv2->axis(1)));
  EXPECT_EQ(
      expr_eval.evaluate(mapper.getProjectedExtent(tv2->axis(1))).as<int64_t>(),
      3 * 4);
}

// Test concrete partial outer dim mapping through merge
TEST_F(VectorizeHelperTest, BackwardMapper6) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2 * 3, 4});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {2 * 3, 4}, {2 * 3 * 4});
  auto tv2 = reshape(tv1, {2 * 3 * 4}, {2, 3 * 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv2, {tv2->axis(1)});

  EXPECT_EQ(mapper.mappedLogicalIds(tv0).size(), 2);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(0)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[1]->sameAs(tv0->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluate(), 3);
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluate(), 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluate(), 3 * 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv2).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[0]->sameAs(tv2->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluate(), 3 * 4);
}

// Test concrete exact inner dim mapping through merge
TEST_F(VectorizeHelperTest, BackwardMapper7) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3 * 4});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {2, 3 * 4}, {2 * 3 * 4});
  auto tv2 = reshape(tv1, {2 * 3 * 4}, {2, 3 * 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv2, {tv2->axis(1)});

  EXPECT_EQ(mapper.mappedLogicalIds(tv0).size(), 2);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(0)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[1]->sameAs(tv0->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluate(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluate(), 3 * 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluate(), 3 * 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv2).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[0]->sameAs(tv2->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluate(), 3 * 4);
}

// Test concrete partial inner dim mapping through merge
TEST_F(VectorizeHelperTest, BackwardMapper8) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3 * 4});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {2, 3 * 4}, {2 * 3 * 4});
  auto tv2 = reshape(tv1, {2 * 3 * 4}, {2 * 3, 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv2, {tv2->axis(1)});

  EXPECT_EQ(mapper.mappedLogicalIds(tv0).size(), 2);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(0)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[1]->sameAs(tv0->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluate(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluate(), 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluate(), 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv2).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[0]->sameAs(tv2->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluate(), 4);
}

// Test concrete partial inner dim mapping through merge
TEST_F(VectorizeHelperTest, BackwardMapper9) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({3, 5, 7});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {3, 5, 7}, {7, 5 * 3});
  auto tv2 = reshape(tv1, {7, 5 * 3}, {3, 5, 7});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv2, {tv2->axis(1), tv2->axis(2)});

  EXPECT_EQ(mapper.mappedLogicalIds(tv0).size(), 3);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(0)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[1]->sameAs(tv0->axis(1)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[2]->sameAs(tv0->axis(2)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluate(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluate(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(2))->evaluate(), 1);

  EXPECT_EQ(mapper.mappedLogicalIds(tv1).size(), 2);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[1]->sameAs(tv1->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluate(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(1))->evaluate(), 5);

  EXPECT_EQ(mapper.mappedLogicalIds(tv2).size(), 2);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[0]->sameAs(tv2->axis(1)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[1]->sameAs(tv2->axis(2)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluate(), 5);
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(2))->evaluate(), 7);
}

// Similar to BackwardMapper1 but in the reverse direction
TEST_F(VectorizeHelperTest, ForwardMapper1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {2, 3}, {2 * 3});
  fusion.addOutput(tv1);
  {
    // No mappings
    auto mapper =
        vectorize_helper::ContiguousInnerDimensionsMapper::map(tv0, {});

    EXPECT_TRUE(
        !mapper.hasMappedDims(tv1) || mapper.mappedLogicalIds(tv1).empty());
    EXPECT_TRUE(
        !mapper.hasMappedDims(tv0) || mapper.mappedLogicalIds(tv0).empty());
  }

  {
    // Inner mapping partial propogates
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv0, {tv0->axis(1)});

    EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(0)));
    EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(1)));

    EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluate(), 3);
  }

  {
    // Full mapping
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv0, {tv0->axis(0), tv0->axis(1)});

    EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(0)));
    EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(0)));
    EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[1]->sameAs(tv0->axis(1)));

    EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluate(), 2);
    EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluate(), 3);

    EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluate(), 2 * 3);
  }
}

// Similar to BackwardMapper2 but in the reverse direction
TEST_F(VectorizeHelperTest, ForwardMapper2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {2, 3, 4}, {2 * 3, 4});
  auto tv2 = reshape(tv1, {2 * 3, 4}, {2 * 3 * 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv0, {tv0->axis(1), tv0->axis(2)});

  EXPECT_THAT(
      [&]() { mapper.getProjectedExtent(tv0->axis(0)); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Not projected")));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluate(), 3);
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(2))->evaluate(), 4);
  // Inner dim fully maps, outer dim of split partially maps
  EXPECT_EQ(mapper.mappedLogicalIds(tv0).size(), 2);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(1)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[1]->sameAs(tv0->axis(2)));

  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluate(), 3);
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(1))->evaluate(), 4);
  // Inner dim fully maps, outer dim of split partially maps
  EXPECT_EQ(mapper.mappedLogicalIds(tv1).size(), 2);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[1]->sameAs(tv1->axis(1)));

  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(0))->evaluate(), 4 * 3);
  EXPECT_EQ(mapper.mappedLogicalIds(tv2).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[0]->sameAs(tv2->axis(0)));
}

// Similar to BackwardMapper3 but in the reverse direction
TEST_F(VectorizeHelperTest, ForwardMapper3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {2, 3, 4}, {2, 3 * 4});
  auto tv2 = reshape(tv1, {2, 3 * 4}, {2 * 3 * 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv0, {tv0->axis(2)});

  // Partial map forwarding
  EXPECT_EQ(mapper.mappedLogicalIds(tv2).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[0]->sameAs(tv2->axis(0)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(0))->evaluate(), 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(1))->evaluate(), 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv0).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(2)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(2))->evaluate(), 4);
}

// Similar to BackwardMapper4 but in the reverse direction
TEST_F(VectorizeHelperTest, ForwardMapper4) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2 * 3});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {2 * 3}, {2, 3});
  fusion.addOutput(tv1);

  {
    // No mapping
    auto mapper =
        vectorize_helper::ContiguousInnerDimensionsMapper::map(tv0, {});

    EXPECT_TRUE(
        !mapper.hasMappedDims(tv1) || mapper.mappedLogicalIds(tv1).empty());
    EXPECT_TRUE(
        !mapper.hasMappedDims(tv0) || mapper.mappedLogicalIds(tv0).empty());
  }

  {
    // Full mapping
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv0, {tv0->axis(0)});

    EXPECT_EQ(mapper.mappedLogicalIds(tv1).size(), 2);
    EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(0)));
    EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[1]->sameAs(tv1->axis(1)));
    EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluate(), 2);
    EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(1))->evaluate(), 3);

    EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluate(), 2 * 3);
  }
}

// Similar to BackwardMapper5 but in the reverse direction
TEST_F(VectorizeHelperTest, ForwardMapper5) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {2, 3 * 4}, {2 * 3 * 4});
  auto tv2 = reshape(tv1, {2 * 3 * 4}, {2 * 3, 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv0, {tv0->axis(1)});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor inp = at::randn({2, 3 * 4}, options);

  KernelArgumentHolder args({inp});
  auto expr_eval = executor_utils::bindInputs(args, &fusion);

  EXPECT_EQ(mapper.mappedLogicalIds(tv2).size(), 2);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[0]->sameAs(tv2->axis(0)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[1]->sameAs(tv2->axis(1)));
  EXPECT_EQ(
      expr_eval.evaluate(mapper.getProjectedExtent(tv2->axis(0))).as<int64_t>(),
      3);
  EXPECT_EQ(
      expr_eval.evaluate(mapper.getProjectedExtent(tv2->axis(1))).as<int64_t>(),
      4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_EQ(
      expr_eval.evaluate(mapper.getProjectedExtent(tv1->axis(0))).as<int64_t>(),
      3 * 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv0).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(1)));
  EXPECT_EQ(
      expr_eval.evaluate(mapper.getProjectedExtent(tv0->axis(1))).as<int64_t>(),
      3 * 4);
}

// Similar to BackwardMapper6 but in the reverse direction
TEST_F(VectorizeHelperTest, ForwardMapper6) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3 * 4});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {2, 3 * 4}, {2 * 3 * 4});
  auto tv2 = reshape(tv1, {2 * 3 * 4}, {{2 * 3, 4}});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv0, {tv0->axis(1)});

  EXPECT_EQ(mapper.mappedLogicalIds(tv2).size(), 2);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[0]->sameAs(tv2->axis(0)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[1]->sameAs(tv2->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(0))->evaluate(), 3);
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluate(), 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluate(), 3 * 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv0).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluate(), 3 * 4);
}

// Similar to BackwardMapper7 but in the reverse direction
TEST_F(VectorizeHelperTest, ForwardMapper7) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3 * 4});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {2, 3 * 4}, {2 * 3 * 4});
  auto tv2 = reshape(tv1, {2 * 3 * 4}, {2, 3 * 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv0, {tv0->axis(1)});

  EXPECT_EQ(mapper.mappedLogicalIds(tv2).size(), 2);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[0]->sameAs(tv2->axis(0)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[1]->sameAs(tv2->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(0))->evaluate(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluate(), 3 * 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluate(), 3 * 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv0).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluate(), 3 * 4);
}

// Similar to BackwardMapper8 but in the reverse direction
TEST_F(VectorizeHelperTest, ForwardMapper8) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2 * 3, 4});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {2 * 3, 4}, {2 * 3 * 4});
  auto tv2 = reshape(tv1, {2 * 3 * 4}, {2, 3 * 4});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv0, {tv0->axis(1)});

  EXPECT_EQ(mapper.mappedLogicalIds(tv2).size(), 2);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[0]->sameAs(tv2->axis(0)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[1]->sameAs(tv2->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(0))->evaluate(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluate(), 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluate(), 4);

  EXPECT_EQ(mapper.mappedLogicalIds(tv0).size(), 1);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluate(), 4);
}

// Make sure partial mappings are mapped to gcd(combined, inner) for inner
// dimension
TEST_F(VectorizeHelperTest, ForwardMapper9) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({3, 5, 7});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {3, 5, 7}, {7, 5 * 3});
  auto tv2 = reshape(tv1, {7, 5 * 3}, {3, 5, 7});
  fusion.addOutput(tv2);

  auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
      tv0, {tv0->axis(1), tv0->axis(2)});

  EXPECT_EQ(mapper.mappedLogicalIds(tv2).size(), 3);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[0]->sameAs(tv2->axis(0)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[1]->sameAs(tv2->axis(1)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv2)[2]->sameAs(tv2->axis(2)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(0))->evaluate(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluate(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(2))->evaluate(), 1);

  EXPECT_EQ(mapper.mappedLogicalIds(tv1).size(), 2);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv1)[1]->sameAs(tv1->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluate(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(1))->evaluate(), 5);

  EXPECT_EQ(mapper.mappedLogicalIds(tv0).size(), 2);
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(1)));
  EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[1]->sameAs(tv0->axis(2)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluate(), 5);
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(2))->evaluate(), 7);
}

// Test propogation doesn't proceed across missing dimensions
TEST_F(VectorizeHelperTest, MapperAdvanced) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // For broadcast we can't back propogate mapped axes to the left of bcast
  // axis.
  // For reduction we can't forward propogate mapped axes to the left of the
  // reduce axis.

  auto tv0 = makeContigConcreteTensor({3, 4 * 6});
  fusion.addInput(tv0);

  auto tv1 = reshape(tv0, {3, 4 * 6}, {3, 4, 6});
  auto tv2 = broadcast(tv1, {false, false, true, false});

  auto tv3 = makeContigConcreteTensor({3, 4, 5, 6});
  fusion.addInput(tv3);
  auto tv4 = add(tv3, tv2);

  auto tv5 = reshape(tv4, {3, 4, 5, 6}, {3 * 4 * 5, 6});

  // Broadcast path from tv0->tv5
  fusion.addOutput(tv5);

  // Sum path from tv3->tv6
  auto tv6 = sum(tv3, {2});
  auto tv7 = reshape(tv6, {3, 4, 6}, {3, 4 * 6});
  fusion.addOutput(tv7);
  {
    // tv5[3*4*5, 6]
    // tv0[3, 4*6]
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv5, {tv5->axis(0), tv5->axis(1)});
    EXPECT_EQ(mapper.mappedLogicalIds(tv0).size(), 1);
    EXPECT_TRUE(mapper.mappedLogicalIds(tv0)[0]->sameAs(tv0->axis(1)));
    EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluate(), 6);
  }

  {
    // tv3[3, 4, 5, 6]
    // tv7[3, 4*6]
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv3, {tv3->axis(0), tv3->axis(1), tv3->axis(2), tv3->axis(3)});
    EXPECT_EQ(mapper.mappedLogicalIds(tv7).size(), 1);
    EXPECT_TRUE(mapper.mappedLogicalIds(tv7)[0]->sameAs(tv7->axis(1)));
    EXPECT_EQ(mapper.getProjectedExtent(tv7->axis(1))->evaluate(), 6);
  }
}

// Test propogation doesn't proceed across missing dimensions
TEST_F(VectorizeHelperTest, SpanningTree) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<TensorView*> inputs;
  std::vector<TensorView*> intermediates;
  std::vector<TensorView*> outputs;

  auto bcast_inp = makeContigConcreteTensor({2});
  inputs.push_back(bcast_inp);
  auto bcast = broadcast(bcast_inp, {false, true});

  for (auto i : arange(10)) {
    auto resolution_inp = makeContigConcreteTensor({2, 2});
    inputs.push_back(resolution_inp);
    auto intermediate = add(bcast, resolution_inp);
    if (i > 0) {
      auto output = add(intermediates.back(), intermediate);
      outputs.push_back(output);
    }
    intermediates.push_back(intermediate);
  }

  for (auto rev_inp : {false, true}) {
    for (auto rev_out : {false, true}) {
      // Clear fusion inputs / outputs
      {
        auto fusion_outs = fusion.outputs();
        for (auto out : fusion_outs) {
          fusion.removeOutput(out);
        }
        auto fusion_inps = fusion.inputs();
        for (auto inp : fusion_inps) {
          fusion.removeInput(inp);
        }
      }

      if (rev_inp) {
        std::reverse(inputs.begin(), inputs.end());
      }

      if (rev_out) {
        std::reverse(outputs.begin(), outputs.end());
      }

      {
        // Populate outputs and inputs
        for (auto out : outputs) {
          fusion.addOutput(out);
        }

        for (auto inp : inputs) {
          fusion.addInput(inp);
        }
      }

      for (auto out : outputs) {
        auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
            out, {out->axis(0), out->axis(1)});

        for (auto tv : fusion.allTvs()) {
          if (tv->name() == 0 || tv->name() == 1) {
            continue;
          }
          for (auto axis : tv->getLogicalDomain()) {
            EXPECT_EQ(mapper.getProjectedExtent(axis)->evaluate(), 2);
          }
        }
      }
    }
  }
}

#if 0
TEST_F(NVFuserTest, FusionSASSDumpError) {
  // create a fake nvdisasm that prints "I am fake" to stderr
  namespace fs = std::filesystem;
  struct FakeNVDisasm {
    const std::string tmpdir = "/tmp/__nvfuser_fake_nvdisasm";

    FakeNVDisasm() {
      std::string nvdisasm = tmpdir + "/nvdisasm";
      fs::create_directory(tmpdir);
      {
        std::ofstream exec(nvdisasm);
        exec << "#!/bin/bash" << std::endl << ">&2 echo I am fake" << std::endl;
      }
      fs::permissions(nvdisasm, fs::perms::owner_exec, fs::perm_options::add);
    }

    ~FakeNVDisasm() {
      std::error_code do_not_throw;
      fs::remove_all(tmpdir, do_not_throw);
    }
  } fake_nvdisasm_raii;

  // Set PATH env to prioritize using this fake nvdisasm
  std::string path = "";
  if (auto original_path = std::getenv("PATH")) {
    path = original_path;
  }
  path = fake_nvdisasm_raii.tmpdir + ":" + path;
  EXPECT_EQ(setenv("PATH", path.c_str(), true), 0);

  // Use fake nvdisasm to do disassembly
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({8});
  auto tv1 = set(tv0);
  fusion.addInput(tv0);
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({8}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  EXPECT_THAT(
      [&]() { ke.compiledKernel()->disassembledKernelSASS(); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("I am fake")));

  auto cg_outputs = ke.run({t0});
  testValidate(
      ke.compiledKernel()->kernel(), cg_outputs, {t0}, __LINE__, __FILE__);
}
#endif

TEST_F(NVFuserTest, ProveLinearAndGetStride) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto size = IrBuilder::create<Val>(DataType::Index);
  IterDomainBuilder builder(fusion.zeroVal(), size);

  ValGraph g;
  auto id0 = builder.build();
  auto id1 = builder.build();
  g.initializeVal(id0);
  g.initializeVal(id1);
  ValGroupAndItsGraph g0{g.toGroup(id0), &g};
  ValGroupAndItsGraph g1{g.toGroup(id1), &g};

  AbstractTensor v1_({g0, g1});
  AbstractTensor v2_ = v1_;
  AbstractTensor v3_ = v1_;
  AbstractTensor v4_ = v1_;

  // v1:
  //        I0         I1
  //       /  \       /  \.
  //          128        128
  //          / \        / \.
  //         /   \      /   \.
  //        /     \    /     \.
  //       /       \  /       \.
  //      /         \/        64.
  //     /          /\       /  \.
  //    /          /  \     /    \.
  //   16         2    8   8      8
  //                    \ /
  //                    xor
  //                    / \.
  //                   8   8
  v1_.split(-1, 128);
  v1_.split(-1, 64);
  v1_.split(-1, 8);
  v1_.split(0, 128);
  v1_.split(1, 8);
  // [I0o, 16, 8, I1o, 2, 8, 8]
  v1_.reorder({{3, 1}, {2, 4}});
  // [I0o, I1o, 16, 2, 8, 8, 8]
  v1_.swizzle(SwizzleType::XOR, 4, 5);
  auto v1__ = v1_.as<ValGroupAndItsGraph>();
  std::vector<ValGroup> v1(v1__.begin(), v1__.end());

  // v2:
  //        I0         I1
  //       /  \       /  \.
  //          128        128
  //          / \        / \.
  //         2  64      8  16
  //            / \        / \.
  //           8   8      1   64
  v2_.split(-1, 128);
  v2_.split(-1, 16);
  v2_.split(-1, 64);
  v2_.split(0, 128);
  v2_.split(1, 64);
  v2_.split(2, 8);
  // [I0o, 2, 8, 8, I1o, 8, 1, 64]
  v2_.reorder({{4, 1}});
  // [I0o, I1o, 2, 8, 8, 8, 1, 64]
  auto v2__ = v2_.as<ValGroupAndItsGraph>();
  std::vector<ValGroup> v2(v2__.begin(), v2__.end());

  // v3:
  //        I0         I1
  //       /  \       /  \.
  //          32         256
  //          / \        / \.
  //         /   \      /   \.
  //        /     \    /     \.
  //       /       \  /       \.
  //      /         \/        64.
  //     /          /\       /  \.
  //    /          /  \     /    \.
  //   4          4    8   8      8
  //                    \ /
  //                    xor
  //                   /   \.
  //                  8     8
  v3_.split(-1, 256);
  v3_.split(-1, 64);
  v3_.split(-1, 8);
  v3_.split(0, 32);
  v3_.split(1, 8);
  // [I0o, 4, 8, I1o, 4, 8, 8]
  v3_.reorder({{3, 1}, {2, 4}});
  // [I0o, I1o, 4, 4, 8, 8, 8]
  v3_.swizzle(SwizzleType::XOR, 4, 5);
  auto v3__ = v3_.as<ValGroupAndItsGraph>();
  std::vector<ValGroup> v3(v3__.begin(), v3__.end());

  // v4:
  //        I0         I1
  //       /  \       /  \.
  //          32         256
  //          / \        / \.
  //         2  16      2  128
  //            / \        / \.
  //           2   8      2   64
  v4_.split(-1, 256);
  v4_.split(-1, 128);
  v4_.split(-1, 64);
  v4_.split(0, 32);
  v4_.split(1, 16);
  v4_.split(2, 8);
  // [I0o, 2, 2, 8, I1o, 2, 2, 64]
  v4_.reorder({{4, 1}});
  // [I0o, I1o, 2, 2, 8, 2, 2, 64]
  auto v4__ = v4_.as<ValGroupAndItsGraph>();
  std::vector<ValGroup> v4(v4__.begin(), v4__.end());

  // v1 in v1
  Val* v1_0_in_v1 = lower_utils::proveLinearAndGetStride(g, v1[0], v1);
  EXPECT_NE(v1_0_in_v1, nullptr);

  Val* v1_1_in_v1 = lower_utils::proveLinearAndGetStride(g, v1[1], v1);
  EXPECT_EQ(simplifyExpr(v1_1_in_v1)->value(), 16384);

  Val* v1_2_in_v1 = lower_utils::proveLinearAndGetStride(g, v1[2], v1);
  EXPECT_EQ(simplifyExpr(v1_2_in_v1)->value(), 1024);

  Val* v1_3_in_v1 = lower_utils::proveLinearAndGetStride(g, v1[3], v1);
  EXPECT_EQ(simplifyExpr(v1_3_in_v1)->value(), 512);

  Val* v1_4_in_v1 = lower_utils::proveLinearAndGetStride(g, v1[4], v1);
  EXPECT_EQ(simplifyExpr(v1_4_in_v1)->value(), 64);

  Val* v1_5_in_v1 = lower_utils::proveLinearAndGetStride(g, v1[5], v1);
  EXPECT_EQ(simplifyExpr(v1_5_in_v1)->value(), 8);

  Val* v1_6_in_v1 = lower_utils::proveLinearAndGetStride(g, v1[6], v1);
  EXPECT_EQ(simplifyExpr(v1_6_in_v1)->value(), 1);

  // v1 in v2
  Val* v1_0_in_v2 = lower_utils::proveLinearAndGetStride(g, v1[0], v2);
  EXPECT_NE(v1_0_in_v2, nullptr);

  Val* v1_1_in_v2 = lower_utils::proveLinearAndGetStride(g, v1[1], v2);
  EXPECT_EQ(simplifyExpr(v1_1_in_v2)->value(), 65536);

  Val* v1_2_in_v2 = lower_utils::proveLinearAndGetStride(g, v1[2], v2);
  EXPECT_EQ(simplifyExpr(v1_2_in_v2)->value(), 4096);

  Val* v1_3_in_v2 = lower_utils::proveLinearAndGetStride(g, v1[3], v2);
  EXPECT_EQ(simplifyExpr(v1_3_in_v2)->value(), 256);

  Val* v1_4_in_v2 = lower_utils::proveLinearAndGetStride(g, v1[4], v2);
  EXPECT_EQ(v1_4_in_v2, nullptr);

  Val* v1_5_in_v2 = lower_utils::proveLinearAndGetStride(g, v1[5], v2);
  EXPECT_EQ(v1_5_in_v2, nullptr);

  Val* v1_6_in_v2 = lower_utils::proveLinearAndGetStride(g, v1[6], v2);
  EXPECT_EQ(simplifyExpr(v1_6_in_v2)->value(), 1);

  // v1 in v3
  Val* v1_0_in_v3 = lower_utils::proveLinearAndGetStride(g, v1[0], v3);
  EXPECT_NE(v1_0_in_v3, nullptr);

  Val* v1_1_in_v3 = lower_utils::proveLinearAndGetStride(g, v1[1], v3);
  EXPECT_EQ(v1_1_in_v3, nullptr);

  Val* v1_2_in_v3 = lower_utils::proveLinearAndGetStride(g, v1[2], v3);
  EXPECT_EQ(v1_2_in_v3, nullptr);

  Val* v1_3_in_v3 = lower_utils::proveLinearAndGetStride(g, v1[3], v3);
  EXPECT_EQ(simplifyExpr(v1_3_in_v3)->value(), 512);

#if 0
  // Not support yet, need to map mathematical equivalence in the almost-exact graph.
  Val* v1_4_in_v3 = lower_utils::proveLinearAndGetStride(g, v1[4], v3);
  EXPECT_EQ(simplifyExpr(v1_4_in_v3)->value(), 64);

  Val* v1_5_in_v3 = lower_utils::proveLinearAndGetStride(g, v1[5], v3);
  EXPECT_EQ(simplifyExpr(v1_5_in_v3)->value(), 8);
#endif

  Val* v1_6_in_v3 = lower_utils::proveLinearAndGetStride(g, v1[6], v3);
  EXPECT_EQ(simplifyExpr(v1_6_in_v3)->value(), 1);

  // v1 in v4
  Val* v1_0_in_v4 = lower_utils::proveLinearAndGetStride(g, v1[0], v4);
  EXPECT_NE(v1_0_in_v4, nullptr);

  Val* v1_1_in_v4 = lower_utils::proveLinearAndGetStride(g, v1[1], v4);
  EXPECT_EQ(v1_1_in_v4, nullptr);

  Val* v1_2_in_v4 = lower_utils::proveLinearAndGetStride(g, v1[2], v4);
  EXPECT_EQ(v1_2_in_v4, nullptr);

  Val* v1_3_in_v4 = lower_utils::proveLinearAndGetStride(g, v1[3], v4);
  EXPECT_EQ(simplifyExpr(v1_3_in_v4)->value(), 64);

  Val* v1_4_in_v4 = lower_utils::proveLinearAndGetStride(g, v1[4], v4);
  EXPECT_EQ(v1_4_in_v4, nullptr);

  Val* v1_5_in_v4 = lower_utils::proveLinearAndGetStride(g, v1[5], v4);
  EXPECT_EQ(v1_5_in_v4, nullptr);

  Val* v1_6_in_v4 = lower_utils::proveLinearAndGetStride(g, v1[6], v4);
  EXPECT_EQ(simplifyExpr(v1_6_in_v4)->value(), 1);

  // v2 in v1
  Val* v2_0_in_v1 = lower_utils::proveLinearAndGetStride(g, v2[0], v1);
  EXPECT_NE(v2_0_in_v1, nullptr);

  Val* v2_1_in_v1 = lower_utils::proveLinearAndGetStride(g, v2[1], v1);
  EXPECT_EQ(simplifyExpr(v2_1_in_v1)->value(), 16384);

  Val* v2_2_in_v1 = lower_utils::proveLinearAndGetStride(g, v2[2], v1);
  EXPECT_EQ(simplifyExpr(v2_2_in_v1)->value(), 8192);

  Val* v2_3_in_v1 = lower_utils::proveLinearAndGetStride(g, v2[3], v1);
  EXPECT_EQ(simplifyExpr(v2_3_in_v1)->value(), 1024);

  Val* v2_4_in_v1 = lower_utils::proveLinearAndGetStride(g, v2[4], v1);
  EXPECT_EQ(v2_4_in_v1, nullptr);

  Val* v2_5_in_v1 = lower_utils::proveLinearAndGetStride(g, v2[5], v1);
  EXPECT_EQ(v2_5_in_v1, nullptr);

  Val* v2_6_in_v1 = lower_utils::proveLinearAndGetStride(g, v2[6], v1);
  EXPECT_EQ(simplifyExpr(v2_6_in_v1)->value(), 0);

  Val* v2_7_in_v1 = lower_utils::proveLinearAndGetStride(g, v2[7], v1);
  EXPECT_EQ(v2_7_in_v1, nullptr);

  // v2 in v2
  Val* v2_0_in_v2 = lower_utils::proveLinearAndGetStride(g, v2[0], v2);
  EXPECT_NE(v2_0_in_v2, nullptr);

  Val* v2_1_in_v2 = lower_utils::proveLinearAndGetStride(g, v2[1], v2);
  EXPECT_EQ(simplifyExpr(v2_1_in_v2)->value(), 65536);

  Val* v2_2_in_v2 = lower_utils::proveLinearAndGetStride(g, v2[2], v2);
  EXPECT_EQ(simplifyExpr(v2_2_in_v2)->value(), 32768);

  Val* v2_3_in_v2 = lower_utils::proveLinearAndGetStride(g, v2[3], v2);
  EXPECT_EQ(simplifyExpr(v2_3_in_v2)->value(), 4096);

  Val* v2_4_in_v2 = lower_utils::proveLinearAndGetStride(g, v2[4], v2);
  EXPECT_EQ(simplifyExpr(v2_4_in_v2)->value(), 512);

  Val* v2_5_in_v2 = lower_utils::proveLinearAndGetStride(g, v2[5], v2);
  EXPECT_EQ(simplifyExpr(v2_5_in_v2)->value(), 64);

  Val* v2_6_in_v2 = lower_utils::proveLinearAndGetStride(g, v2[6], v2);
  EXPECT_EQ(simplifyExpr(v2_6_in_v2)->value(), 0);

  Val* v2_7_in_v2 = lower_utils::proveLinearAndGetStride(g, v2[7], v2);
  EXPECT_EQ(simplifyExpr(v2_7_in_v2)->value(), 1);

  // v2 in v3
  Val* v2_0_in_v3 = lower_utils::proveLinearAndGetStride(g, v2[0], v3);
  EXPECT_NE(v2_0_in_v3, nullptr);

  Val* v2_1_in_v3 = lower_utils::proveLinearAndGetStride(g, v2[1], v3);
  EXPECT_EQ(v2_1_in_v3, nullptr);

  Val* v2_2_in_v3 = lower_utils::proveLinearAndGetStride(g, v2[2], v3);
  EXPECT_NE(v2_2_in_v3, nullptr);

  Val* v2_3_in_v3 = lower_utils::proveLinearAndGetStride(g, v2[3], v3);
  EXPECT_EQ(v2_3_in_v3, nullptr);

  Val* v2_4_in_v3 = lower_utils::proveLinearAndGetStride(g, v2[4], v3);
  EXPECT_EQ(v2_4_in_v3, nullptr);

  Val* v2_5_in_v3 = lower_utils::proveLinearAndGetStride(g, v2[5], v3);
  EXPECT_EQ(v2_5_in_v3, nullptr);

  Val* v2_6_in_v3 = lower_utils::proveLinearAndGetStride(g, v2[6], v3);
  EXPECT_EQ(simplifyExpr(v2_6_in_v3)->value(), 0);

  Val* v2_7_in_v3 = lower_utils::proveLinearAndGetStride(g, v2[7], v3);
  EXPECT_EQ(v2_7_in_v3, nullptr);

  // v2 in v4
  Val* v2_0_in_v4 = lower_utils::proveLinearAndGetStride(g, v2[0], v4);
  EXPECT_NE(v2_0_in_v4, nullptr);

  Val* v2_1_in_v4 = lower_utils::proveLinearAndGetStride(g, v2[1], v4);
  EXPECT_EQ(v2_1_in_v4, nullptr);

  Val* v2_2_in_v4 = lower_utils::proveLinearAndGetStride(g, v2[2], v4);
  EXPECT_NE(v2_2_in_v4, nullptr);

  Val* v2_3_in_v4 = lower_utils::proveLinearAndGetStride(g, v2[3], v4);
  EXPECT_EQ(v2_3_in_v4, nullptr);

  Val* v2_4_in_v4 = lower_utils::proveLinearAndGetStride(g, v2[4], v4);
  EXPECT_EQ(simplifyExpr(v2_4_in_v4)->value(), 256);

  Val* v2_5_in_v4 = lower_utils::proveLinearAndGetStride(g, v2[5], v4);
  EXPECT_EQ(simplifyExpr(v2_5_in_v4)->value(), 16);

  Val* v2_6_in_v4 = lower_utils::proveLinearAndGetStride(g, v2[6], v4);
  EXPECT_EQ(simplifyExpr(v2_6_in_v4)->value(), 0);

  Val* v2_7_in_v4 = lower_utils::proveLinearAndGetStride(g, v2[7], v4);
  EXPECT_EQ(simplifyExpr(v2_7_in_v4)->value(), 1);

  // v3 in v1
  Val* v3_0_in_v1 = lower_utils::proveLinearAndGetStride(g, v3[0], v1);
  EXPECT_EQ(v3_0_in_v1, nullptr);

  Val* v3_1_in_v1 = lower_utils::proveLinearAndGetStride(g, v3[1], v1);
  EXPECT_EQ(simplifyExpr(v3_1_in_v1)->value(), 32768);

  Val* v3_2_in_v1 = lower_utils::proveLinearAndGetStride(g, v3[2], v1);
  EXPECT_EQ(simplifyExpr(v3_2_in_v1)->value(), 1024);

  Val* v3_3_in_v1 = lower_utils::proveLinearAndGetStride(g, v3[3], v1);
  EXPECT_EQ(v3_3_in_v1, nullptr);

#if 0
  // Not support yet, need to map mathematical equivalence in the almost-exact graph.
  Val* v3_4_in_v1 = lower_utils::proveLinearAndGetStride(g, v3[4], v1);
  EXPECT_EQ(simplifyExpr(v3_4_in_v1)->value(), 64);

  Val* v3_5_in_v1 = lower_utils::proveLinearAndGetStride(g, v3[5], v1);
  EXPECT_EQ(simplifyExpr(v3_5_in_v1)->value(), 8);
#endif

  Val* v3_6_in_v1 = lower_utils::proveLinearAndGetStride(g, v3[6], v1);
  EXPECT_EQ(simplifyExpr(v3_6_in_v1)->value(), 1);

  // v3 in v2
  Val* v3_0_in_v2 = lower_utils::proveLinearAndGetStride(g, v3[0], v2);
  EXPECT_EQ(v3_0_in_v2, nullptr);

  Val* v3_1_in_v2 = lower_utils::proveLinearAndGetStride(g, v3[1], v2);
  EXPECT_EQ(simplifyExpr(v3_1_in_v2)->value(), 131072);

  Val* v3_2_in_v2 = lower_utils::proveLinearAndGetStride(g, v3[2], v2);
  EXPECT_EQ(simplifyExpr(v3_2_in_v2)->value(), 4096);

  Val* v3_3_in_v2 = lower_utils::proveLinearAndGetStride(g, v3[3], v2);
  EXPECT_EQ(v3_3_in_v2, nullptr);

  Val* v3_4_in_v2 = lower_utils::proveLinearAndGetStride(g, v3[4], v2);
  EXPECT_EQ(v3_4_in_v2, nullptr);

  Val* v3_5_in_v2 = lower_utils::proveLinearAndGetStride(g, v3[5], v2);
  EXPECT_EQ(v3_5_in_v2, nullptr);

  Val* v3_6_in_v2 = lower_utils::proveLinearAndGetStride(g, v3[6], v2);
  EXPECT_EQ(simplifyExpr(v3_6_in_v2)->value(), 1);

  // v3 in v3
  Val* v3_0_in_v3 = lower_utils::proveLinearAndGetStride(g, v3[0], v3);
  EXPECT_NE(v3_0_in_v3, nullptr);

  Val* v3_1_in_v3 = lower_utils::proveLinearAndGetStride(g, v3[1], v3);
  EXPECT_EQ(simplifyExpr(v3_1_in_v3)->value(), 8192);

  Val* v3_2_in_v3 = lower_utils::proveLinearAndGetStride(g, v3[2], v3);
  EXPECT_EQ(simplifyExpr(v3_2_in_v3)->value(), 2048);

  Val* v3_3_in_v3 = lower_utils::proveLinearAndGetStride(g, v3[3], v3);
  EXPECT_EQ(simplifyExpr(v3_3_in_v3)->value(), 512);

  Val* v3_4_in_v3 = lower_utils::proveLinearAndGetStride(g, v3[4], v3);
  EXPECT_EQ(simplifyExpr(v3_4_in_v3)->value(), 64);

  Val* v3_5_in_v3 = lower_utils::proveLinearAndGetStride(g, v3[5], v3);
  EXPECT_EQ(simplifyExpr(v3_5_in_v3)->value(), 8);

  Val* v3_6_in_v3 = lower_utils::proveLinearAndGetStride(g, v3[6], v3);
  EXPECT_EQ(simplifyExpr(v3_6_in_v3)->value(), 1);

  // v3 in v4
  Val* v3_0_in_v4 = lower_utils::proveLinearAndGetStride(g, v3[0], v4);
  EXPECT_NE(v3_0_in_v4, nullptr);

  Val* v3_1_in_v4 = lower_utils::proveLinearAndGetStride(g, v3[1], v4);
  EXPECT_EQ(simplifyExpr(v3_1_in_v4)->value(), 8192);

  Val* v3_2_in_v4 = lower_utils::proveLinearAndGetStride(g, v3[2], v4);
  EXPECT_EQ(simplifyExpr(v3_2_in_v4)->value(), 2048);

  Val* v3_3_in_v4 = lower_utils::proveLinearAndGetStride(g, v3[3], v4);
  EXPECT_EQ(simplifyExpr(v3_3_in_v4)->value(), 64);

  Val* v3_4_in_v4 = lower_utils::proveLinearAndGetStride(g, v3[4], v4);
  EXPECT_EQ(v3_4_in_v4, nullptr);

  Val* v3_5_in_v4 = lower_utils::proveLinearAndGetStride(g, v3[5], v4);
  EXPECT_EQ(v3_5_in_v4, nullptr);

  Val* v3_6_in_v4 = lower_utils::proveLinearAndGetStride(g, v3[6], v4);
  EXPECT_EQ(simplifyExpr(v3_6_in_v4)->value(), 1);

  // v4 in v1
  Val* v4_0_in_v1 = lower_utils::proveLinearAndGetStride(g, v4[0], v1);
  EXPECT_EQ(v4_0_in_v1, nullptr);

  Val* v4_1_in_v1 = lower_utils::proveLinearAndGetStride(g, v4[1], v1);
  EXPECT_EQ(simplifyExpr(v4_1_in_v1)->value(), 32768);

  Val* v4_2_in_v1 = lower_utils::proveLinearAndGetStride(g, v4[2], v1);
  EXPECT_EQ(simplifyExpr(v4_2_in_v1)->value(), 2048);

  Val* v4_3_in_v1 = lower_utils::proveLinearAndGetStride(g, v4[3], v1);
  EXPECT_EQ(simplifyExpr(v4_3_in_v1)->value(), 1024);

  Val* v4_4_in_v1 = lower_utils::proveLinearAndGetStride(g, v4[4], v1);
  EXPECT_EQ(v4_4_in_v1, nullptr);

  Val* v4_5_in_v1 = lower_utils::proveLinearAndGetStride(g, v4[5], v1);
  EXPECT_EQ(simplifyExpr(v4_5_in_v1)->value(), 16384);

  Val* v4_6_in_v1 = lower_utils::proveLinearAndGetStride(g, v4[6], v1);
  EXPECT_EQ(simplifyExpr(v4_6_in_v1)->value(), 512);

  Val* v4_7_in_v1 = lower_utils::proveLinearAndGetStride(g, v4[7], v1);
  EXPECT_EQ(v4_7_in_v1, nullptr);

  // v4 in v2
  Val* v4_0_in_v2 = lower_utils::proveLinearAndGetStride(g, v4[0], v2);
  EXPECT_EQ(v4_0_in_v2, nullptr);

  Val* v4_1_in_v2 = lower_utils::proveLinearAndGetStride(g, v4[1], v2);
  EXPECT_EQ(simplifyExpr(v4_1_in_v2)->value(), 131072);

  Val* v4_2_in_v2 = lower_utils::proveLinearAndGetStride(g, v4[2], v2);
  EXPECT_EQ(simplifyExpr(v4_2_in_v2)->value(), 8192);

  Val* v4_3_in_v2 = lower_utils::proveLinearAndGetStride(g, v4[3], v2);
  EXPECT_EQ(simplifyExpr(v4_3_in_v2)->value(), 4096);

  Val* v4_4_in_v2 = lower_utils::proveLinearAndGetStride(g, v4[4], v2);
  EXPECT_EQ(simplifyExpr(v4_4_in_v2)->value(), 512);

  Val* v4_5_in_v2 = lower_utils::proveLinearAndGetStride(g, v4[5], v2);
  EXPECT_EQ(simplifyExpr(v4_5_in_v2)->value(), 65536);

  Val* v4_6_in_v2 = lower_utils::proveLinearAndGetStride(g, v4[6], v2);
  EXPECT_EQ(simplifyExpr(v4_6_in_v2)->value(), 256);

  Val* v4_7_in_v2 = lower_utils::proveLinearAndGetStride(g, v4[7], v2);
  EXPECT_EQ(v4_7_in_v2, nullptr);

  // v4 in v3
  Val* v4_0_in_v3 = lower_utils::proveLinearAndGetStride(g, v4[0], v3);
  EXPECT_NE(v4_0_in_v3, nullptr);

  Val* v4_1_in_v3 = lower_utils::proveLinearAndGetStride(g, v4[1], v3);
  EXPECT_EQ(simplifyExpr(v4_1_in_v3)->value(), 8192);

  Val* v4_2_in_v3 = lower_utils::proveLinearAndGetStride(g, v4[2], v3);
  EXPECT_EQ(simplifyExpr(v4_2_in_v3)->value(), 4096);

  Val* v4_3_in_v3 = lower_utils::proveLinearAndGetStride(g, v4[3], v3);
  EXPECT_EQ(simplifyExpr(v4_3_in_v3)->value(), 2048);

  Val* v4_4_in_v3 = lower_utils::proveLinearAndGetStride(g, v4[4], v3);
  EXPECT_EQ(v4_4_in_v3, nullptr);

  Val* v4_5_in_v3 = lower_utils::proveLinearAndGetStride(g, v4[5], v3);
  EXPECT_EQ(simplifyExpr(v4_5_in_v3)->value(), 1024);

  Val* v4_6_in_v3 = lower_utils::proveLinearAndGetStride(g, v4[6], v3);
  EXPECT_EQ(simplifyExpr(v4_6_in_v3)->value(), 512);

  Val* v4_7_in_v3 = lower_utils::proveLinearAndGetStride(g, v4[7], v3);
  EXPECT_EQ(v4_7_in_v3, nullptr);

  // v4 in v4
  Val* v4_0_in_v4 = lower_utils::proveLinearAndGetStride(g, v4[0], v4);
  EXPECT_NE(v4_0_in_v4, nullptr);

  Val* v4_1_in_v4 = lower_utils::proveLinearAndGetStride(g, v4[1], v4);
  EXPECT_EQ(simplifyExpr(v4_1_in_v4)->value(), 8192);

  Val* v4_2_in_v4 = lower_utils::proveLinearAndGetStride(g, v4[2], v4);
  EXPECT_EQ(simplifyExpr(v4_2_in_v4)->value(), 4096);

  Val* v4_3_in_v4 = lower_utils::proveLinearAndGetStride(g, v4[3], v4);
  EXPECT_EQ(simplifyExpr(v4_3_in_v4)->value(), 2048);

  Val* v4_4_in_v4 = lower_utils::proveLinearAndGetStride(g, v4[4], v4);
  EXPECT_EQ(simplifyExpr(v4_4_in_v4)->value(), 256);

  Val* v4_5_in_v4 = lower_utils::proveLinearAndGetStride(g, v4[5], v4);
  EXPECT_EQ(simplifyExpr(v4_5_in_v4)->value(), 128);

  Val* v4_6_in_v4 = lower_utils::proveLinearAndGetStride(g, v4[6], v4);
  EXPECT_EQ(simplifyExpr(v4_6_in_v4)->value(), 64);

  Val* v4_7_in_v4 = lower_utils::proveLinearAndGetStride(g, v4[7], v4);
  EXPECT_EQ(simplifyExpr(v4_7_in_v4)->value(), 1);
}

// Test that lower_utils::proveLinearAndGetStride still works even if some
// dependency are missing, as long as the missing dependency is irrelevant to
// result.
TEST_F(NVFuserTest, ProveLinearAndGetStrideWithMissingDependency) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  for (auto _ : arange(100)) {
    (void)_;
    // [16, 8, 2, 4]
    auto id16 =
        IterDomainBuilder(
            fusion.zeroVal(), IrBuilder::create<Val>(16, DataType::Index))
            .build();
    auto id8 = IterDomainBuilder(
                   fusion.zeroVal(), IrBuilder::create<Val>(8, DataType::Index))
                   .build();
    auto id2 = IterDomainBuilder(
                   fusion.zeroVal(), IrBuilder::create<Val>(2, DataType::Index))
                   .build();
    auto id4 = IterDomainBuilder(
                   fusion.zeroVal(), IrBuilder::create<Val>(4, DataType::Index))
                   .build();

    ValGraph g;
    g.initializeVal(id16);
    g.initializeVal(id8);
    g.initializeVal(id2);
    g.initializeVal(id4);
    ValGroup g16{g.toGroup(id16)};
    ValGroup g8{g.toGroup(id8)};
    ValGroup g2{g.toGroup(id2)};
    ValGroup g4{g.toGroup(id4)};
    ValGroupAndItsGraph gg16{g16, &g};
    ValGroupAndItsGraph gg8{g8, &g};
    ValGroupAndItsGraph gg2{g2, &g};
    ValGroupAndItsGraph gg4{g4, &g};

    AbstractTensor v({gg16, gg8, gg2, gg4});
    // Merge all dims in random order
    while (v.size() > 1) {
      v.merge(std::rand() % (v.size() - 1));
    }
    v.split(0, 32);

    ValGroup linear_g = v[1].as<ValGroupAndItsGraph>().group;
    // Although linear_g depend on g16, whether it is linear w.r.t. [8, 2, 4] is
    // not relevant to g16. So we should not require g16 to exist in order to
    // prove linearity.
    Val* stride =
        lower_utils::proveLinearAndGetStride(g, linear_g, {g8, g2, g4});
    ASSERT_NE(stride, nullptr);
    EXPECT_EQ(simplifyExpr(stride)->value(), 1);
  }
}

TEST_F(NVFuserTest, ProveLinearAndGetStrideEarlyStopping) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [4, 2]
  auto id4 = IterDomainBuilder(
                 fusion.zeroVal(), IrBuilder::create<Val>(4, DataType::Index))
                 .build();
  auto id2 = IterDomainBuilder(
                 fusion.zeroVal(), IrBuilder::create<Val>(2, DataType::Index))
                 .build();

  ValGraph g;
  g.initializeVal(id4);
  g.initializeVal(id2);
  ValGroup g4{g.toGroup(id4)};
  ValGroup g2{g.toGroup(id2)};
  ValGroupAndItsGraph gg4{g4, &g};
  ValGroupAndItsGraph gg2{g2, &g};
  AbstractTensor v({gg4, gg2});
  v.merge(0);
  v.split(0, 2);
  ValGroup g4_ = v[0].as<ValGroupAndItsGraph>().group;
  ValGroup g2_ = v[1].as<ValGroupAndItsGraph>().group;
  Val* stride = lower_utils::proveLinearAndGetStride(g, g2_, {g4, g2_});
  ASSERT_NE(stride, nullptr);
  EXPECT_EQ(simplifyExpr(stride)->value(), 1);
}

using TestCpp23BackPort = NVFuserTest;

TEST_F(TestCpp23BackPort, ZipDifferentWaysToSayZeroToTen) {
  // vector of integers
  std::vector<int64_t> integer{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  // list of English words
  std::list<std::string> english{
      "zero",
      "one",
      "two",
      "three",
      "four",
      "five",
      "six",
      "seven",
      "eight",
      "nine"};

  // Custom iterator and range implementing the set-theoretic definition of
  // natural numbers:
  // https://en.wikipedia.org/wiki/Set-theoretic_definition_of_natural_numbers
  struct SetTheoreticNaturalNumber {
    std::vector<SetTheoreticNaturalNumber> content;
    using value_type = SetTheoreticNaturalNumber;
    using difference_type = std::ptrdiff_t;
    SetTheoreticNaturalNumber() = default; // zero
    SetTheoreticNaturalNumber(
        std::initializer_list<SetTheoreticNaturalNumber> x)
        : content(x) {}
    SetTheoreticNaturalNumber operator*() const {
      return *this;
    }
    SetTheoreticNaturalNumber& operator++() {
      content.emplace_back(*this);
      return *this;
    }
    SetTheoreticNaturalNumber operator++(int) {
      SetTheoreticNaturalNumber temp = *this;
      ++(*this);
      return temp;
    }
    bool operator==(const SetTheoreticNaturalNumber& other) const {
      return content == other.content;
    }
  };
  static_assert(std::input_iterator<SetTheoreticNaturalNumber>);
  struct ZeroToInf : std::ranges::view_interface<ZeroToInf> {
    SetTheoreticNaturalNumber begin() {
      return SetTheoreticNaturalNumber();
    }
    auto end() {
      return std::unreachable_sentinel;
    }
  } set_theoretic_zero_to_inf;
  static_assert(std::ranges::input_range<ZeroToInf>);
  static_assert(std::ranges::view<ZeroToInf>);

  int64_t counter = 0;
  auto english_it = english.begin();
  for (auto&& [i, e, s, iota] :
       zip(integer,
           english,
           set_theoretic_zero_to_inf,
           std::views::iota((int64_t)0))) {
    static_assert(std::is_same_v<decltype(i), int64_t&>);
    static_assert(std::is_same_v<decltype(e), std::string&>);
    static_assert(std::is_same_v<decltype(s), SetTheoreticNaturalNumber>);
    static_assert(std::is_same_v<decltype(iota), int64_t>);
    EXPECT_EQ(i, counter);
    EXPECT_EQ(&i, &integer[counter]);
    EXPECT_EQ(&e, &*english_it);
    EXPECT_EQ(iota, counter);
    switch (counter) {
      case 0: {
        EXPECT_EQ(e, "zero");
        SetTheoreticNaturalNumber expect = {};
        EXPECT_EQ(s, expect);
        break;
      }
      case 1: {
        EXPECT_EQ(e, "one");
        SetTheoreticNaturalNumber expect = {{}};
        EXPECT_EQ(s, expect);
        break;
      }
      case 2: {
        EXPECT_EQ(e, "two");
        SetTheoreticNaturalNumber expect = {{}, {{}}};
        EXPECT_EQ(s, expect);
        break;
      }
      case 3: {
        EXPECT_EQ(e, "three");
        SetTheoreticNaturalNumber expect = {{}, {{}}, {{}, {{}}}};
        EXPECT_EQ(s, expect);
        break;
      }
      case 4: {
        EXPECT_EQ(e, "four");
        SetTheoreticNaturalNumber expect = {
            {}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}};
        EXPECT_EQ(s, expect);
        break;
      }
      case 5: {
        EXPECT_EQ(e, "five");
        SetTheoreticNaturalNumber expect = {
            {},
            {{}},
            {{}, {{}}},
            {{}, {{}}, {{}, {{}}}},
            {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}};
        EXPECT_EQ(s, expect);
        break;
      }
      case 6: {
        EXPECT_EQ(e, "six");
        SetTheoreticNaturalNumber expect = {
            {},
            {{}},
            {{}, {{}}},
            {{}, {{}}, {{}, {{}}}},
            {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}},
            {{},
             {{}},
             {{}, {{}}},
             {{}, {{}}, {{}, {{}}}},
             {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}}};
        EXPECT_EQ(s, expect);
        break;
      }
      case 7: {
        EXPECT_EQ(e, "seven");
        SetTheoreticNaturalNumber expect = {
            {},
            {{}},
            {{}, {{}}},
            {{}, {{}}, {{}, {{}}}},
            {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}},
            {{},
             {{}},
             {{}, {{}}},
             {{}, {{}}, {{}, {{}}}},
             {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}},
            {{},
             {{}},
             {{}, {{}}},
             {{}, {{}}, {{}, {{}}}},
             {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}},
             {{},
              {{}},
              {{}, {{}}},
              {{}, {{}}, {{}, {{}}}},
              {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}}}};
        EXPECT_EQ(s, expect);
        break;
      }
      case 8: {
        EXPECT_EQ(e, "eight");
        SetTheoreticNaturalNumber expect = {
            {},
            {{}},
            {{}, {{}}},
            {{}, {{}}, {{}, {{}}}},
            {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}},
            {{},
             {{}},
             {{}, {{}}},
             {{}, {{}}, {{}, {{}}}},
             {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}},
            {{},
             {{}},
             {{}, {{}}},
             {{}, {{}}, {{}, {{}}}},
             {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}},
             {{},
              {{}},
              {{}, {{}}},
              {{}, {{}}, {{}, {{}}}},
              {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}}},
            {{},
             {{}},
             {{}, {{}}},
             {{}, {{}}, {{}, {{}}}},
             {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}},
             {{},
              {{}},
              {{}, {{}}},
              {{}, {{}}, {{}, {{}}}},
              {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}},
             {{},
              {{}},
              {{}, {{}}},
              {{}, {{}}, {{}, {{}}}},
              {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}},
              {{},
               {{}},
               {{}, {{}}},
               {{}, {{}}, {{}, {{}}}},
               {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}}}}};
        EXPECT_EQ(s, expect);
        break;
      }
      case 9: {
        EXPECT_EQ(e, "nine");
        SetTheoreticNaturalNumber expect = {
            {},
            {{}},
            {{}, {{}}},
            {{}, {{}}, {{}, {{}}}},
            {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}},
            {{},
             {{}},
             {{}, {{}}},
             {{}, {{}}, {{}, {{}}}},
             {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}},
            {{},
             {{}},
             {{}, {{}}},
             {{}, {{}}, {{}, {{}}}},
             {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}},
             {{},
              {{}},
              {{}, {{}}},
              {{}, {{}}, {{}, {{}}}},
              {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}}},
            {{},
             {{}},
             {{}, {{}}},
             {{}, {{}}, {{}, {{}}}},
             {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}},
             {{},
              {{}},
              {{}, {{}}},
              {{}, {{}}, {{}, {{}}}},
              {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}},
             {{},
              {{}},
              {{}, {{}}},
              {{}, {{}}, {{}, {{}}}},
              {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}},
              {{},
               {{}},
               {{}, {{}}},
               {{}, {{}}, {{}, {{}}}},
               {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}}}},
            {{},
             {{}},
             {{}, {{}}},
             {{}, {{}}, {{}, {{}}}},
             {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}},
             {{},
              {{}},
              {{}, {{}}},
              {{}, {{}}, {{}, {{}}}},
              {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}},
             {{},
              {{}},
              {{}, {{}}},
              {{}, {{}}, {{}, {{}}}},
              {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}},
              {{},
               {{}},
               {{}, {{}}},
               {{}, {{}}, {{}, {{}}}},
               {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}}},
             {{},
              {{}},
              {{}, {{}}},
              {{}, {{}}, {{}, {{}}}},
              {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}},
              {{},
               {{}},
               {{}, {{}}},
               {{}, {{}}, {{}, {{}}}},
               {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}},
              {{},
               {{}},
               {{}, {{}}},
               {{}, {{}}, {{}, {{}}}},
               {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}},
               {{},
                {{}},
                {{}, {{}}},
                {{}, {{}}, {{}, {{}}}},
                {{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}}}}}};
        EXPECT_EQ(s, expect);
        break;
      }
    }
    counter++;
    english_it++;
  }
  EXPECT_EQ(counter, 10);
}

TEST_F(TestCpp23BackPort, ZipWithReverse) {
  std::vector<int> v{1, 2, 3, 4, 5};
  std::vector<int> v2{5, 4, 3, 2, 1};

  int64_t count = 0;
  for (auto&& [x, y] : zip(v, v2)) {
    EXPECT_EQ(x, 6 - y);
    EXPECT_EQ(x, count + 1);
    count++;
  }
  EXPECT_EQ(count, 5);

  count = 0;
  for (auto&& [x, y] : zip(v, v2) | std::views::reverse) {
    EXPECT_EQ(x, 6 - y);
    EXPECT_EQ(x, 5 - count);
    count++;
  }
  EXPECT_EQ(count, 5);

  count = 0;
  std::forward_list<int> fl{1, 2, 3, 4, 5};
  for (auto&& [x, y] : zip(v, fl)) {
    EXPECT_EQ(x, y);
    EXPECT_EQ(x, count + 1);
    count++;
  }
  EXPECT_EQ(count, 5);

  // Can not do zip(v, fl) | std::views::reverse because fl is not bidirectional
}

TEST_F(TestCpp23BackPort, Enumerate) {
  std::vector<int> v{1, 2, 3, 4, 5};

  int64_t count = 0;
  for (auto&& [i, x] : enumerate(v)) {
    EXPECT_EQ(i, count);
    EXPECT_EQ(x, count + 1);
    count++;
  }
  EXPECT_EQ(count, 5);

  count = 0;
  for (auto&& [i, x] : enumerate(v) | std::views::reverse) {
    EXPECT_EQ(i + 1, x);
    EXPECT_EQ(i, 4 - count);
    count++;
  }
  EXPECT_EQ(count, 5);

  std::forward_list<int> fl{1, 2, 3, 4, 5};
  for (auto&& [i, x] : enumerate(fl)) {
    EXPECT_EQ(i + 1, x);
  }
  EXPECT_EQ(count, 5);

  // Can not do enumerate(fl) | std::views::reverse because fl is not
  // bidirectional
}

namespace {

// Generator that yields integers from 0 to n-1
Generator<int> zeroToN(int n) {
  for (int i = 0; i < n; ++i) {
    co_yield i;
  }
}

// Generator that yields integers from n to 2*n - 1
Generator<int> nTo2N(int n) {
  for (int i = n; i < 2 * n; ++i) {
    co_yield i;
  }
}

// Generator that yields integers from m to m + 2*n - 1
Generator<int> mTo2NplusM(int n, int m) {
  for (auto x : zeroToN(n)) {
    co_yield x + m;
  }
  for (auto x : nTo2N(n)) {
    co_yield x + m;
  }
}

// Generator that yields references
Generator<int&> items(std::vector<int>& v) {
  for (auto& x : v) {
    co_yield x;
  }
}

} // namespace

TEST_F(NVFuserTest, Generator1) {
  static_assert(std::ranges::view<decltype(zeroToN(10))>);
  std::vector<int> generated;
  for (auto x : zeroToN(10) |
           std::views::filter([](int x) { return x % 2 == 0; }) |
           std::views::transform([](int x) { return x * x; })) {
    generated.push_back(x);
  }
  std::vector<int> expect{0, 4, 16, 36, 64};
  EXPECT_EQ(generated, expect);
}

TEST_F(NVFuserTest, Generator2) {
  static_assert(std::ranges::view<decltype(mTo2NplusM(10, 10))>);
  std::vector<int> generated;
  for (auto x : mTo2NplusM(10, 10)) {
    generated.push_back(x);
  }
  std::vector<int> expect{10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                          20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
  EXPECT_EQ(generated, expect);
}

TEST_F(NVFuserTest, Generator3) {
  std::vector<int> v{0, 0, 0, 0, 0};
  for (auto&& [i, x] : enumerate(items(v))) {
    x = i * 10;
  }
  std::vector<int> expect{0, 10, 20, 30, 40};
  EXPECT_EQ(v, expect);
}

TEST_F(NVFuserTest, Generator4) {
  auto one2five = []() -> Generator<int> {
    for (int i = 1; i <= 5; ++i) {
      co_yield i;
    }
  };
  std::vector<int> v;
  for (auto x : one2five()) {
    v.push_back(x);
  }
  std::vector<int> expect{1, 2, 3, 4, 5};
  EXPECT_EQ(v, expect);
}

TEST_F(NVFuserTest, Generator5) {
  auto excepted_exception = []() -> Generator<int> {
    co_yield 1;
    throw std::runtime_error("Hello, world!");
    co_yield 2;
  };
  auto run_generator = [&]() {
    for (auto x : excepted_exception()) {
      EXPECT_EQ(x, 1);
    }
  };
  EXPECT_THAT(
      run_generator,
      ::testing::ThrowsMessage<std::runtime_error>("Hello, world!"));
}

} // namespace nvfuser
