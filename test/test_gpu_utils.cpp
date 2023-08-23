// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <device_lower/utils.h>
#include <executor_utils.h>
#include <fusion.h>
#include <ops/all_ops.h>
#include <scheduler/utils.h>
#include <scheduler/vectorize_helper.h>
#include <test/utils.h>
#include <test/validator.h>

#include <cstdlib>
#include <filesystem>
#include <system_error>

namespace nvfuser {

TEST_F(NVFuserTest, FusionSplitDims_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto p = prime_number;
  auto tv = makeConcreteTensor(
      {p(0) * p(1) * p(2), p(3), p(4), p(5) * p(6), p(7), p(8), p(9) * p(10)});
  std::vector<size_t> dims{0, 1, 2, 3, 4, 5, 6};
  scheduler_utils::splitDims(
      tv, {{0, p(2)}, {0, p(1)}, {3, p(6)}, {6, p(10)}}, dims);
  EXPECT_EQ(tv->nDims(), 11);
  for (auto i : c10::irange(11)) {
    EXPECT_EQ(tv->axis(i)->extent()->evaluateInt(), p(i));
  }
  std::vector<size_t> expect{0, 3, 4, 5, 7, 8, 9};
  EXPECT_EQ(dims, expect);
}

TEST_F(NVFuserTest, FusionMergeDims_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto p = prime_number;
  auto tv = makeConcreteTensor(
      {p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9), p(10)});
  std::vector<size_t> dims{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto merged = scheduler_utils::mergeDims(tv, {3, 2, 9, 8, 7}, dims);
  EXPECT_EQ(merged, (size_t)2);
  std::vector<int64_t> expect_shape{
      p(0), p(1), p(2) * p(3) * p(7) * p(8) * p(9), p(4), p(5), p(6), p(10)};
  EXPECT_EQ(tv->nDims(), expect_shape.size());
  for (auto i : c10::irange(expect_shape.size())) {
    EXPECT_EQ(tv->axis(i)->extent()->evaluateInt(), expect_shape[i]);
  }
  std::vector<size_t> expect_dims{0, 1, 2, 2, 3, 4, 5, 2, 2, 2, 6};
  EXPECT_EQ(dims, expect_dims);

  auto merged_dim = tv->axis(2)->toString(0);
  EXPECT_EQ(
      merged_dim.substr(merged_dim.find("{")),
      "{( 23 * ( 19 * ( 29 * ( 5 * 7 ) ) ) )}");
}

TEST_F(NVFuserTest, FusionReorderAsRFactor_CUDA) {
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
  auto old2new = scheduler_utils::domainReorderAsRfactorMap(tv0);
  EXPECT_EQ(old2new[0], 2);
  EXPECT_EQ(old2new[1], 1);
  EXPECT_EQ(old2new[2], 0);
}

TEST_F(NVFuserTest, FusionDisjointViewSet_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion->addInput(tv0);

  auto tv1 = reshape(tv0, {2, 3, 4}, {2, 12});

  auto tv2 = makeConcreteTensor({2, 12});
  fusion->addInput(tv2);

  auto tv3 = add(tv2, tv1);
  fusion->addOutput(tv3);

  auto disjoint_exact = scheduler_utils::disjointRFactorSets(fusion.get());

  TORCH_INTERNAL_ASSERT(
      disjoint_exact.strictAreMapped(tv0->axis(1), tv0->axis(2)));
}

TEST_F(NVFuserTest, FusionBroadcastViewMultiples_CUDA) {
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

  EXPECT_EQ(bcast_info.broadcast_multiples[0].lhs_multiple, 0);
  EXPECT_EQ(bcast_info.broadcast_multiples[0].rhs_multiple, 8 * 4);

  EXPECT_EQ(bcast_info.broadcast_multiples[1].lhs_multiple, 7 * 4);
  EXPECT_EQ(bcast_info.broadcast_multiples[1].rhs_multiple, 8 * 4);

  EXPECT_EQ(bcast_info.broadcast_multiples[2].lhs_multiple, 7 * 4);
  EXPECT_EQ(bcast_info.broadcast_multiples[2].rhs_multiple, 7 * 4);

  EXPECT_EQ(bcast_info.broadcast_multiples[3].lhs_multiple, 8 * 4);
  EXPECT_EQ(bcast_info.broadcast_multiples[3].rhs_multiple, 7 * 4);

  EXPECT_EQ(bcast_info.broadcast_multiples[4].lhs_multiple, 8 * 4);
  EXPECT_EQ(bcast_info.broadcast_multiples[4].rhs_multiple, 7 * 4);

  EXPECT_EQ(bcast_info.broadcast_multiples[5].lhs_multiple, 8 * 4);
  EXPECT_EQ(bcast_info.broadcast_multiples[5].rhs_multiple, 7 * 4);
}

TEST_F(NVFuserTest, FusionTVDomainGuard_CUDA) {
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
TEST_F(VectorizeHelperTest, BackwardMapper1_CUDA) {
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
        !mapper.hasMappedDims(tv0) || mapper.mappedRFactorIds(tv0).empty());
    EXPECT_TRUE(
        !mapper.hasMappedDims(tv1) || mapper.mappedRFactorIds(tv1).empty());
  }

  {
    // Inner mapping partial propogates
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv1, {tv1->axis(1)});

    EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
    EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(1)));

    EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluateInt(), 3);
  }

  {
    // Full mapping
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv1, {tv1->axis(0), tv1->axis(1)});

    EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
    EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
    EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[1]->sameAs(tv1->axis(1)));

    EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluateInt(), 2);
    EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(1))->evaluateInt(), 3);

    EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluateInt(), 2 * 3);
  }
}

// Test backward mapping through multiple splits
TEST_F(VectorizeHelperTest, BackwardMapper2_CUDA) {
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
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("Not projected")));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluateInt(), 3);
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(2))->evaluateInt(), 4);
  // Inner dim fully maps, outer dim of split partially maps
  EXPECT_EQ(mapper.mappedRFactorIds(tv2).size(), 2);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(1)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[1]->sameAs(tv2->axis(2)));

  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluateInt(), 3);
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(1))->evaluateInt(), 4);
  // Inner dim fully maps, outer dim of split partially maps
  EXPECT_EQ(mapper.mappedRFactorIds(tv1).size(), 2);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[1]->sameAs(tv1->axis(1)));

  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluateInt(), 4 * 3);
  EXPECT_EQ(mapper.mappedRFactorIds(tv0).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
}

// Test backward mapping through multiple splits
TEST_F(VectorizeHelperTest, BackwardMapper3_CUDA) {
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
  EXPECT_EQ(mapper.mappedRFactorIds(tv0).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluateInt(), 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(1))->evaluateInt(), 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv2).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(2)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(2))->evaluateInt(), 4);
}

// Test simple backward mapping through merge
TEST_F(VectorizeHelperTest, BackwardMapper4_CUDA) {
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
        !mapper.hasMappedDims(tv0) || mapper.mappedRFactorIds(tv0).empty());
    EXPECT_TRUE(
        !mapper.hasMappedDims(tv1) || mapper.mappedRFactorIds(tv1).empty());
  }

  {
    // Full merge mapping
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv1, {tv1->axis(0)});

    EXPECT_EQ(mapper.mappedRFactorIds(tv0).size(), 2);
    EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
    EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(1)));
    EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluateInt(), 2);
    EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluateInt(), 3);

    EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluateInt(), 2 * 3);
  }
}

// Test symbolic partial mapping through merge
TEST_F(VectorizeHelperTest, BackwardMapper5_CUDA) {
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

  KernelArgumentHolder args =
      KernelArgumentHolder::createKernelArgumentHolder({inp});
  auto expr_eval = executor_utils::bindInputs(args, &fusion);

  EXPECT_EQ(mapper.mappedRFactorIds(tv0).size(), 2);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(1)));
  EXPECT_EQ(
      expr_eval.evaluate(mapper.getProjectedExtent(tv0->axis(0))).as<int64_t>(),
      3);
  EXPECT_EQ(
      expr_eval.evaluate(mapper.getProjectedExtent(tv0->axis(1))).as<int64_t>(),
      4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_EQ(
      expr_eval.evaluate(mapper.getProjectedExtent(tv1->axis(0))).as<int64_t>(),
      3 * 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv2).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(1)));
  EXPECT_EQ(
      expr_eval.evaluate(mapper.getProjectedExtent(tv2->axis(1))).as<int64_t>(),
      3 * 4);
}

// Test concrete partial outer dim mapping through merge
TEST_F(VectorizeHelperTest, BackwardMapper6_CUDA) {
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

  EXPECT_EQ(mapper.mappedRFactorIds(tv0).size(), 2);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluateInt(), 3);
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluateInt(), 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluateInt(), 3 * 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv2).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluateInt(), 3 * 4);
}

// Test concrete exact inner dim mapping through merge
TEST_F(VectorizeHelperTest, BackwardMapper7_CUDA) {
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

  EXPECT_EQ(mapper.mappedRFactorIds(tv0).size(), 2);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluateInt(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluateInt(), 3 * 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluateInt(), 3 * 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv2).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluateInt(), 3 * 4);
}

// Test concrete partial inner dim mapping through merge
TEST_F(VectorizeHelperTest, BackwardMapper8_CUDA) {
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

  EXPECT_EQ(mapper.mappedRFactorIds(tv0).size(), 2);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluateInt(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluateInt(), 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluateInt(), 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv2).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluateInt(), 4);
}

// Test concrete partial inner dim mapping through merge
TEST_F(VectorizeHelperTest, BackwardMapper9_CUDA) {
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

  EXPECT_EQ(mapper.mappedRFactorIds(tv0).size(), 3);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(1)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[2]->sameAs(tv0->axis(2)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluateInt(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluateInt(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(2))->evaluateInt(), 1);

  EXPECT_EQ(mapper.mappedRFactorIds(tv1).size(), 2);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[1]->sameAs(tv1->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluateInt(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(1))->evaluateInt(), 5);

  EXPECT_EQ(mapper.mappedRFactorIds(tv2).size(), 2);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(1)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[1]->sameAs(tv2->axis(2)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluateInt(), 5);
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(2))->evaluateInt(), 7);
}

// Similar to BackwardMapper1_CUDA but in the reverse direction
TEST_F(VectorizeHelperTest, ForwardMapper1_CUDA) {
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
        !mapper.hasMappedDims(tv1) || mapper.mappedRFactorIds(tv1).empty());
    EXPECT_TRUE(
        !mapper.hasMappedDims(tv0) || mapper.mappedRFactorIds(tv0).empty());
  }

  {
    // Inner mapping partial propogates
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv0, {tv0->axis(1)});

    EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
    EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(1)));

    EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluateInt(), 3);
  }

  {
    // Full mapping
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv0, {tv0->axis(0), tv0->axis(1)});

    EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
    EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(0)));
    EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(1)));

    EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluateInt(), 2);
    EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluateInt(), 3);

    EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluateInt(), 2 * 3);
  }
}

// Similar to BackwardMapper2_CUDA but in the reverse direction
TEST_F(VectorizeHelperTest, ForwardMapper2_CUDA) {
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
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("Not projected")));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluateInt(), 3);
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(2))->evaluateInt(), 4);
  // Inner dim fully maps, outer dim of split partially maps
  EXPECT_EQ(mapper.mappedRFactorIds(tv0).size(), 2);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(1)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(2)));

  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluateInt(), 3);
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(1))->evaluateInt(), 4);
  // Inner dim fully maps, outer dim of split partially maps
  EXPECT_EQ(mapper.mappedRFactorIds(tv1).size(), 2);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[1]->sameAs(tv1->axis(1)));

  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(0))->evaluateInt(), 4 * 3);
  EXPECT_EQ(mapper.mappedRFactorIds(tv2).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(0)));
}

// Similar to BackwardMapper3_CUDA but in the reverse direction
TEST_F(VectorizeHelperTest, ForwardMapper3_CUDA) {
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
  EXPECT_EQ(mapper.mappedRFactorIds(tv2).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(0)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(0))->evaluateInt(), 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(1))->evaluateInt(), 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv0).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(2)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(2))->evaluateInt(), 4);
}

// Similar to BackwardMapper4_CUDA but in the reverse direction
TEST_F(VectorizeHelperTest, ForwardMapper4_CUDA) {
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
        !mapper.hasMappedDims(tv1) || mapper.mappedRFactorIds(tv1).empty());
    EXPECT_TRUE(
        !mapper.hasMappedDims(tv0) || mapper.mappedRFactorIds(tv0).empty());
  }

  {
    // Full mapping
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv0, {tv0->axis(0)});

    EXPECT_EQ(mapper.mappedRFactorIds(tv1).size(), 2);
    EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
    EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[1]->sameAs(tv1->axis(1)));
    EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluateInt(), 2);
    EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(1))->evaluateInt(), 3);

    EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(0))->evaluateInt(), 2 * 3);
  }
}

// Similar to BackwardMapper5_CUDA but in the reverse direction
TEST_F(VectorizeHelperTest, ForwardMapper5_CUDA) {
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

  KernelArgumentHolder args =
      KernelArgumentHolder::createKernelArgumentHolder({inp});
  auto expr_eval = executor_utils::bindInputs(args, &fusion);

  EXPECT_EQ(mapper.mappedRFactorIds(tv2).size(), 2);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(0)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[1]->sameAs(tv2->axis(1)));
  EXPECT_EQ(
      expr_eval.evaluate(mapper.getProjectedExtent(tv2->axis(0))).as<int64_t>(),
      3);
  EXPECT_EQ(
      expr_eval.evaluate(mapper.getProjectedExtent(tv2->axis(1))).as<int64_t>(),
      4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_EQ(
      expr_eval.evaluate(mapper.getProjectedExtent(tv1->axis(0))).as<int64_t>(),
      3 * 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv0).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(1)));
  EXPECT_EQ(
      expr_eval.evaluate(mapper.getProjectedExtent(tv0->axis(1))).as<int64_t>(),
      3 * 4);
}

// Similar to BackwardMapper6_CUDA but in the reverse direction
TEST_F(VectorizeHelperTest, ForwardMapper6_CUDA) {
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

  EXPECT_EQ(mapper.mappedRFactorIds(tv2).size(), 2);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(0)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[1]->sameAs(tv2->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(0))->evaluateInt(), 3);
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluateInt(), 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluateInt(), 3 * 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv0).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluateInt(), 3 * 4);
}

// Similar to BackwardMapper7_CUDA but in the reverse direction
TEST_F(VectorizeHelperTest, ForwardMapper7_CUDA) {
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

  EXPECT_EQ(mapper.mappedRFactorIds(tv2).size(), 2);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(0)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[1]->sameAs(tv2->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(0))->evaluateInt(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluateInt(), 3 * 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluateInt(), 3 * 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv0).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluateInt(), 3 * 4);
}

// Similar to BackwardMapper8_CUDA but in the reverse direction
TEST_F(VectorizeHelperTest, ForwardMapper8_CUDA) {
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

  EXPECT_EQ(mapper.mappedRFactorIds(tv2).size(), 2);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(0)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[1]->sameAs(tv2->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(0))->evaluateInt(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluateInt(), 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv1).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluateInt(), 4);

  EXPECT_EQ(mapper.mappedRFactorIds(tv0).size(), 1);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluateInt(), 4);
}

// Make sure partial mappings are mapped to gcd(combined, inner) for inner
// dimension
TEST_F(VectorizeHelperTest, ForwardMapper9_CUDA) {
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

  EXPECT_EQ(mapper.mappedRFactorIds(tv2).size(), 3);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[0]->sameAs(tv2->axis(0)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[1]->sameAs(tv2->axis(1)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv2)[2]->sameAs(tv2->axis(2)));
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(0))->evaluateInt(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(1))->evaluateInt(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv2->axis(2))->evaluateInt(), 1);

  EXPECT_EQ(mapper.mappedRFactorIds(tv1).size(), 2);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[0]->sameAs(tv1->axis(0)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv1)[1]->sameAs(tv1->axis(1)));
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(0))->evaluateInt(), 1);
  EXPECT_EQ(mapper.getProjectedExtent(tv1->axis(1))->evaluateInt(), 5);

  EXPECT_EQ(mapper.mappedRFactorIds(tv0).size(), 2);
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(1)));
  EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[1]->sameAs(tv0->axis(2)));
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluateInt(), 5);
  EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(2))->evaluateInt(), 7);
}

// Test propogation doesn't proceed across missing dimensions
TEST_F(VectorizeHelperTest, MapperAdvanced_CUDA) {
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
    EXPECT_EQ(mapper.mappedRFactorIds(tv0).size(), 1);
    EXPECT_TRUE(mapper.mappedRFactorIds(tv0)[0]->sameAs(tv0->axis(1)));
    EXPECT_EQ(mapper.getProjectedExtent(tv0->axis(1))->evaluateInt(), 6);
  }

  {
    // tv3[3, 4, 5, 6]
    // tv7[3, 4*6]
    auto mapper = vectorize_helper::ContiguousInnerDimensionsMapper::map(
        tv3, {tv3->axis(0), tv3->axis(1), tv3->axis(2), tv3->axis(3)});
    EXPECT_EQ(mapper.mappedRFactorIds(tv7).size(), 1);
    EXPECT_TRUE(mapper.mappedRFactorIds(tv7)[0]->sameAs(tv7->axis(1)));
    EXPECT_EQ(mapper.getProjectedExtent(tv7->axis(1))->evaluateInt(), 6);
  }
}

// Test propogation doesn't proceed across missing dimensions
TEST_F(VectorizeHelperTest, SpanningTree_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<TensorView*> inputs;
  std::vector<TensorView*> intermediates;
  std::vector<TensorView*> outputs;

  auto bcast_inp = makeContigConcreteTensor({2});
  inputs.push_back(bcast_inp);
  auto bcast = broadcast(bcast_inp, {false, true});

  for (auto i : c10::irange(10)) {
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
          fusion.removeOutput(inp);
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

        for (auto tv : ir_utils::allTvs(&fusion)) {
          if (tv->name() == 0 || tv->name() == 1) {
            continue;
          }
          for (auto axis : tv->getRootDomain()) {
            EXPECT_EQ(mapper.getProjectedExtent(axis)->evaluateInt(), 2);
          }
        }
      }
    }
  }
}

TEST_F(NVFuserTest, FusionSASSDumpError_CUDA) {
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

  FusionExecutor fe;
  fe.setSaveCompiledBinaryFlag(true);
  fe.compileFusion(&fusion, {t0});

  EXPECT_THAT(
      [&]() { fe.disassembledKernelSASS(); },
      ::testing::ThrowsMessage<c10::Error>(::testing::HasSubstr("I am fake")));

  auto cg_outputs = fe.runFusion({t0});
  testValidate(fe.kernel(), cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

} // namespace nvfuser
