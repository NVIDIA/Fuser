// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <executor_kernel_arg.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <multidevice/lower_communication.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/utils.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

using namespace at::indexing;
using MultiDeviceUtilsTest = NVFuserTest;

// TODO: This test checks that isSharded generates an error when a split/merged
// axis is parallelized with DIDx. Update when this restriction is lifted.
TEST_F(MultiDeviceUtilsTest, TestIsSharded) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* a = makeSymbolicTensor(3);
  a->axis(2)->parallelize(ParallelType::DIDx);
  a->split(0, 4);
  EXPECT_TRUE(isSharded(a));

  TensorView* b = makeSymbolicTensor(3);
  b->split(1, 4);
  b->axis(1)->parallelize(ParallelType::DIDx);
  EXPECT_ANY_THROW(isSharded(b));

  TensorView* c = makeSymbolicTensor(3);
  c->axis(0)->parallelize(ParallelType::DIDx);
  c->axis(1)->parallelize(ParallelType::DIDx);
  EXPECT_ANY_THROW(isSharded(c));
}

TEST_F(MultiDeviceUtilsTest, TestPropagateSharding) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* a = makeSymbolicTensor(3);
  TensorView* b = makeSymbolicTensor(3);
  TensorView* c = add(a, b);

  DeviceMesh mesh({0, 1, 2});
  a->setDeviceMesh(mesh);
  b->setDeviceMesh(mesh);
  a->axis(0)->parallelize(ParallelType::DIDx);
  b->axis(2)->parallelize(ParallelType::DIDx);
  fusion.addInput(a);
  fusion.addInput(b);
  fusion.addOutput(c);

  // Expected behavior: a's shardings propagate to c.
  propagateShardings(&fusion);
  checkSameShardings(a, c);
}

TEST_F(MultiDeviceUtilsTest, ReshardingDetection) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  DeviceMesh mesh0, mesh1, mesh2;
  mesh0 = {0, 1};
  mesh1 = {0, 2};
  mesh2 = {0, 1, 2};

  TensorView* tv0 = makeContigTensor(3);
  fusion->addInput(tv0);
  tv0->setDeviceMesh(mesh0);

  TensorView* tv1 = set(tv0);
  tv1->setDeviceMesh(mesh0);

  TensorView* tv2 = set(tv1); // resharding
  tv2->setDeviceMesh(mesh1);

  TensorView* tv3 = set(tv2);
  tv3->setDeviceMesh(mesh1);

  TensorView* tv4 = set(tv3); // resharding
  tv4->setDeviceMesh(mesh2);

  TensorView* tv5 = set(tv4); // resharding
  tv5->setDeviceMesh(mesh2);
  tv5->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv6 = set(tv5);
  tv6->setDeviceMesh(mesh2);
  tv6->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv7 = set(tv6); // resharding
  tv7->setDeviceMesh(mesh2);

  TensorView* tv8 = sum(tv0, {0});
  tv8->setDeviceMesh(mesh0);

  TensorView* tv9 = sum(tv0, {0}); // resharding, but seems invalid
  tv9->setDeviceMesh(mesh0);
  tv9->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv10 = sum(tv0, {0}); // resharding,
  tv10->setDeviceMesh(mesh0);
  tv10->axis(1)->parallelize(ParallelType::DIDx);

  TensorView* tv11 = sum(tv0, {0}); // resharding,
  tv11->setDeviceMesh(mesh1);

  TensorView* tv12 = sum(tv5, {0}); // not resharding
  tv12->setDeviceMesh(mesh2);
  tv12->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv13 = sum(tv5, {0}); // resharding
  tv13->setDeviceMesh(mesh2);
  tv13->axis(1)->parallelize(ParallelType::DIDx);

  TensorView* tv14 = sum(tv5, {0}); // resharding
  tv14->setDeviceMesh(mesh2);
  tv14->axis(0)->parallelize(ParallelType::DIDx);
  tv14->axis(1)->parallelize(ParallelType::DIDx);

  TensorView* tv15 = add(tv0, tv1);
  tv15->setDeviceMesh(mesh0);

  TensorView* tv16 = add(tv0, tv1); // resharding
  tv16->setDeviceMesh(mesh1);

  TensorView* tv17 = add(tv0, tv1); // resharding
  tv17->setDeviceMesh(mesh0);
  tv17->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv18 = add(tv5, tv6);
  tv18->setDeviceMesh(mesh2);
  tv18->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv19 = add(tv5, tv7); // resharding
  tv19->setDeviceMesh(mesh2);
  tv19->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv20 = add(tv5, tv7); // resharding
  tv20->setDeviceMesh(mesh2);

  TensorView* tv21 = add(tv0, tv7); // resharding
  tv21->setDeviceMesh(mesh2);

  TensorView* tv22 = sum(tv5, {1});
  tv22->setDeviceMesh(mesh2);
  tv22->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv23 = sum(tv5, {1}); // resharding
  tv23->setDeviceMesh(mesh2);

  TensorView* tv24 = sum(tv7, {0});
  tv24->setDeviceMesh(mesh2);

  TensorView* tv25 = sum(tv7, {0}); // not resharding but invalid
  tv25->setDeviceMesh(mesh2);
  tv22->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv26 = add(tv5, tv6); // resharding
  tv26->setDeviceMesh(mesh2);

  fusion->addOutput(tv1);
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);
  fusion->addOutput(tv4);
  fusion->addOutput(tv5);
  fusion->addOutput(tv6);
  fusion->addOutput(tv7);
  fusion->addOutput(tv8);
  fusion->addOutput(tv9);
  fusion->addOutput(tv10);
  fusion->addOutput(tv11);
  fusion->addOutput(tv12);
  fusion->addOutput(tv13);
  fusion->addOutput(tv14);
  fusion->addOutput(tv15);
  fusion->addOutput(tv16);
  fusion->addOutput(tv17);
  fusion->addOutput(tv18);
  fusion->addOutput(tv19);
  fusion->addOutput(tv20);
  fusion->addOutput(tv21);
  fusion->addOutput(tv22);
  fusion->addOutput(tv23);
  fusion->addOutput(tv24);
  fusion->addOutput(tv25);
  fusion->addOutput(tv26);

  GTEST_EXPECT_TRUE(!isResharding(tv1->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv2->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv3->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv4->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv5->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv6->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv7->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv8->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv9->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv10->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv11->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv12->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv13->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv14->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv15->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv16->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv17->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv18->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv19->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv20->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv21->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv22->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv23->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv24->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv25->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv26->definition()));
}

TEST_F(MultiDeviceUtilsTest, InsertResharding_Before) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* a = makeSymbolicTensor(3);
  TensorView* b = makeSymbolicTensor(3);
  TensorView* c = add(a, b);
  fusion.addInput(a);
  fusion.addInput(b);
  fusion.addOutput(c);

  DeviceMesh mesh0({0, 1});
  DeviceMesh mesh1({2});
  a->setDeviceMesh(mesh0);
  b->setDeviceMesh(mesh0);
  c->setDeviceMesh(mesh1);

  a->axis(0)->parallelize(ParallelType::DIDx);
  c->axis(1)->parallelize(ParallelType::DIDx);

  insertReshardings(&fusion);
  std::vector<Val*> outputs = fusion.outputs();

  c = outputs[0]->as<TensorView>();
  for (auto input : c->definition()->inputs()) {
    checkSameShardings(c, input->as<TensorView>());
  }
}

TEST_F(MultiDeviceUtilsTest, InsertResharding_After) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* a = makeSymbolicTensor(3);
  TensorView* b = relu(a);
  fusion.addInput(a);
  fusion.addOutput(b);

  DeviceMesh mesh0({0, 1});
  DeviceMesh mesh1({2});
  a->setDeviceMesh(mesh0);
  b->setDeviceMesh(mesh1);

  a->axis(0)->parallelize(ParallelType::DIDx);
  b->axis(1)->parallelize(ParallelType::DIDx);

  insertReshardings(&fusion);
  std::vector<Val*> outputs = fusion.outputs();

  b = outputs[0]->as<TensorView>();
  Expr* expr = b->definition();
  EXPECT_TRUE(expr->isA<LoadStoreOp>());
  EXPECT_TRUE(expr->as<LoadStoreOp>()->opType() == LoadStoreOpType::Set);
  checkSameShardings(a, expr->inputs()[0]->as<TensorView>());
}

TEST_F(MultiDeviceUtilsTest, InsertShardedAxisReordering) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* a = makeSymbolicTensor(3);
  TensorView* b = relu(a);
  TensorView* c = add(a, b);
  fusion.addInput(a);
  fusion.addOutput(c);

  DeviceMesh mesh0({0, 1});
  DeviceMesh mesh1({2});
  a->setDeviceMesh(mesh0);
  b->setDeviceMesh(mesh1);
  c->setDeviceMesh(mesh1);

  a->axis(1)->parallelize(ParallelType::DIDx);
  c->axis(1)->parallelize(ParallelType::DIDx);

  insertReshardings(&fusion);
  std::cout << "Fusion now contains " << fusion.exprs().size() << std::endl;
  // int num_inner_reshardings = std::count_if(
  //     fusion.exprs().begin(), fusion.exprs().end(), [](Expr* expr) {
  //       return (isResharding(expr) && isInnerResharding(expr));
  //     });
  int num_inner_reshardings = 0;
  for (auto expr : fusion.exprs()) {
    if (isResharding(expr) && isInnerResharding(expr)) {
      num_inner_reshardings++;
    }
  }
  EXPECT_TRUE(num_inner_reshardings > 0);

  insertShardedAxisReordering(&fusion);
  for (auto expr : fusion.exprs()) {
    if (isResharding(expr)) {
      EXPECT_TRUE(!isInnerResharding(expr));
    }
  }
}

using automaticSetInsertionTestParams =
    std::tuple<DeviceMesh, DeviceMesh, DeviceMesh, bool, bool, bool>;

class automaticReshardingTest
    : public NVFuserTest,
      public ::testing::WithParamInterface<automaticSetInsertionTestParams> {
 protected:
  void SetUp() override {
    fusion = std::make_unique<Fusion>();
    fg = std::make_unique<FusionGuard>(fusion.get());
  }
  void validate() {
    for (auto expr : fusion->exprs()) {
      GTEST_EXPECT_TRUE(!isResharding(expr) || isLowerableToCommunication(expr))
          << "on expr=" << expr;
    }

    SegmentCandidateFinderOptions options{
        .run_translate_welford = false,
        .run_combine_reductions = false,
        .run_herrmann_merge = true,
        .run_final_merge = true,
        .only_segment_resharding_exprs = true};

    auto segmented_fusion =
        SegmentCandidateFinder::segment(std::move(fusion), nullptr, options);

    for (SegmentedGroup* group : segmented_fusion->groups()) {
      GTEST_EXPECT_TRUE(
          std::none_of(
              group->exprs().begin(),
              group->exprs().end(),
              [](auto expr) { return isResharding(expr); }) ||
          (group->exprs().size() == 1 && isResharding(group->exprs().at(0))));
    }
    // checks that the segments are disjoints and that the graph of segment is
    // acyclic
    segmented_fusion->validate();
  }

  std::unique_ptr<Fusion> fusion;
  std::unique_ptr<FusionGuard> fg;
};

TEST_P(automaticReshardingTest, setInsertion) {
  if (!distributedEnabled()) { // Test only works with distributed
    GTEST_SKIP() << "Requires distributed API";
  }
  auto
      [mesh0,
       mesh1,
       mesh2,
       is_tv0_tv3_tv5_sharded,
       is_tv1_tv4_sharded,
       is_tv2_sharded] = GetParam();
  int sharded_axis = 1;

  TensorView* tv0 = makeContigTensor(3);
  TensorView* tv1 = binaryOp(BinaryOpType::Mul, tv0, tv0);
  TensorView* tv2 = binaryOp(BinaryOpType::Add, tv0, tv1);
  TensorView* tv3 = sum(tv2, {1});
  TensorView* tv4 = broadcast(tv3, {false, true, false});
  TensorView* tv5 = binaryOp(BinaryOpType::Mul, tv2, tv4);

  tv0->setDeviceMesh(mesh0);
  tv1->setDeviceMesh(mesh1);
  tv2->setDeviceMesh(mesh2);
  tv3->setDeviceMesh(mesh0);
  tv4->setDeviceMesh(mesh1);
  tv5->setDeviceMesh(mesh0);
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  fusion->addOutput(tv5);

  if (is_tv0_tv3_tv5_sharded) {
    tv0->axis(sharded_axis)->parallelize(ParallelType::DIDx);
    tv3->axis(sharded_axis)->parallelize(ParallelType::DIDx);
    tv5->axis(sharded_axis)->parallelize(ParallelType::DIDx);
  }
  if (is_tv1_tv4_sharded) {
    tv1->axis(sharded_axis)->parallelize(ParallelType::DIDx);
    tv4->axis(sharded_axis)->parallelize(ParallelType::DIDx);
  }
  if (is_tv2_sharded) {
    tv2->axis(sharded_axis)->parallelize(ParallelType::DIDx);
  }

  insertReshardings(fusion.get());
  insertShardedAxisReordering(fusion.get());
  validate();
}

namespace {

DeviceMesh Mesh0({0});
DeviceMesh Mesh1({1, 2});
DeviceMesh Mesh2({0, 1, 2, 3});

} // namespace

INSTANTIATE_TEST_SUITE_P(
    ,
    automaticReshardingTest,
    ::testing::Combine(
        ::testing::Values(Mesh0, Mesh2),
        ::testing::Values(Mesh1, Mesh2),
        ::testing::Values(Mesh2),
        ::testing::Bool(),
        ::testing::Bool(),
        ::testing::Bool()));

} // namespace nvfuser
