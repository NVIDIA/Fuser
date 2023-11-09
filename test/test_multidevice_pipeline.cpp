// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <gtest/gtest.h>

#include <codegen.h>
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <executor.h>
#include <executor_params.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/graphviz.h>
#include <ir/iostream.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_cache.h>
#include <kernel_ir.h>
#include <mma_type.h>
#include <multidevice/lower_resharding_expr.h>
#include <ops/all_ops.h>
#include <root_domain_map.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <test/multidevice.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <transform_replay.h>
#include <transform_rfactor.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

using namespace torch::jit::fuser::cuda;
using namespace at::indexing;

/* To run the following tests on several devices, pytorch must be installed
   with the flag USE_DISTRIBUTED=1 and nccl support.
   Then simply run the tests on several processes, for example using mpirun
   on a node having at least 6 GPUs,
   e.g.: mpirun -np 6 build/nvfuser_tests
   --gtest_filter=PipelineTest.Pipeline
*/

TEST_F(PipelineTest, Pipeline) {
  const std::vector<int64_t> input_shape1 = {3096, 1123};
  const std::vector<int64_t> input_shape2 = {2048, 73, 81};
  // ===========================================================
  //        FUSION
  // ===========================================================
  FusionGuard fg(fusion.get());

  TensorView* tv0_ = makeConcreteTensor(input_shape1);
  fusion->addInput(tv0_);
  TensorView* tv1_ = sum(tv0_, {0});

  TensorView* tv2_ = set(tv1_);
  TensorView* tv3_ = sum(tv2_, {0});
  fusion->addOutput(tv3_);

  TensorView* tv0 = makeConcreteTensor(input_shape2);
  fusion->addInput(tv0);
  TensorView* tv1 = sum(tv0, {0});

  TensorView* tv2 = set(tv1);
  TensorView* tv2a = set(tv2);
  TensorView* tv3 = sum(tv2a, {0});

  TensorView* tv4 = set(tv1);
  TensorView* tv4a = set(tv4);
  TensorView* tv5 = sum(tv4a, {0});

  TensorView* tv6 = set(tv2a);
  TensorView* tv7 = sum(tv6, {0});
  fusion->addOutput(tv7);

  TensorView* tv8 = set(tv3);
  TensorView* tv9 = set(tv5);
  TensorView* tv10 = set(tv7);
  TensorView* tv11 = add(tv8, tv9);
  TensorView* tv12 = add(tv11, tv10);
  TensorView* tv13 = sum(tv12, {0});
  fusion->addOutput(tv13);

  // ===========================================================
  //        PIPELINE SCHEDULING
  // ===========================================================
  /* Each TensorView must be assigned to one and only one stage
     WAR: if an intermediate TensorView is automatically added
          in the Fusion during Fusion definition,
          it also needs to be assigned manually to a stage */
  PipelineStageDescriptor stage0_, stage1_, stage0, stage1, stage2, stage3,
      stage4;
  stage0_.addVal({tv0_, tv1_});
  stage1_.addVal({tv2_, tv3_});
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv2a, tv3});
  stage2.addVal({tv4, tv4a, tv5});
  stage3.addVal({tv6, tv7});
  stage4.addVal({tv8, tv9, tv10, tv11, tv12, tv13});

  // binding each stage to a device mesh
  stage0_.mesh = {5};
  stage1_.mesh = {2, 4};
  stage0.mesh = {0};
  stage1.mesh = {0, 1, 4};
  stage2.mesh = {1, 3};
  stage3.mesh = {2};
  stage4.mesh = {4, 5};

  PipelineDescriptor descriptor{.stage_descriptors{
      std::move(stage0_),
      std::move(stage1_),
      std::move(stage0),
      std::move(stage1),
      std::move(stage2),
      std::move(stage3),
      std::move(stage4)}}; // the order doesn't matter

  pipeline = std::make_unique<Pipeline>(fusion.get(), std::move(descriptor));

  // Create input tensors.
  // Note: each process is binded to a different GPU
  // Note: the concrete values are only used at the relevant ranks
  inputs = {
      at::randn(input_shape1, tensor_options),
      at::randn(input_shape2, tensor_options)};

  validate();
}

DeviceMesh mesh0({0});
DeviceMesh mesh1({1});
DeviceMesh mesh2({0, 1, 2, 3});
DeviceMesh mesh3({0, 2, 3});
DeviceMesh mesh4({1, 0, 2});
auto all_meshes = ::testing::Values(mesh0, mesh1, mesh2, mesh3, mesh4);

TEST_P(PipelineTestTwoStages, Communication) {}

INSTANTIATE_TEST_SUITE_P(
    Gather,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_meshes,
        all_meshes,
        ::testing::Values(true),
        ::testing::Values(false)));

INSTANTIATE_TEST_SUITE_P(
    Scatter,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_meshes,
        all_meshes,
        ::testing::Values(false),
        ::testing::Values(true)));

INSTANTIATE_TEST_SUITE_P(
    Bcast,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_meshes,
        all_meshes,
        ::testing::Values(false),
        ::testing::Values(false)));

INSTANTIATE_TEST_SUITE_P(
    Bcast_sharded,
    PipelineTestTwoStages,
    ::testing::Combine(
        ::testing::Values(mesh3),
        ::testing::Values(mesh4),
        ::testing::Values(true),
        ::testing::Values(true)));


TEST_F(PipelineTest, Pipeline_Reduce) {
  const std::vector<int64_t> input_shape = {4, 3, 1, 2};

  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeConcreteTensor(input_shape);
  fusion->addInput(tv0);
  TensorView* tv1 = sum(tv0, {1});
  TensorView* tv2 = sum(tv1, {0});
  TensorView* tv3 = sum(tv2, {0});
  fusion->addOutput(tv3);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0(false), stage1(false);
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});

  stage0.mesh = {0, 1, 2, 3};
  stage1.mesh = {0, 1};

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  pipeline = std::make_unique<Pipeline>(fusion.get(), std::move(descriptor));

  inputs = {at::ones(input_shape, tensor_options) * (communicator->deviceId() + 1)};

  validate();
}

TEST_F(PipelineTest, Pipeline_ReduceToExternalRoot) {
  const std::vector<int64_t> input_shape = {2, 3, 1, 2};

  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeConcreteTensor(input_shape);
  fusion->addInput(tv0);
  TensorView* tv1 = sum(tv0, {1});
  TensorView* tv2 = sum(tv1, {0});
  TensorView* tv3 = sum(tv2, {0});
  fusion->addOutput(tv3);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0(false), stage1(false);
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});

  stage0.mesh = {0, 1};
  stage1.mesh = {2};

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  pipeline = std::make_unique<Pipeline>(fusion.get(), std::move(descriptor));

  inputs = {at::ones(input_shape, tensor_options) * (communicator->deviceId() + 1)};

  validate();
}

TEST_F(PipelineTest, Pipeline_Allreduce) {
  const std::vector<int64_t> input_shape = {4, 3, 1, 2};
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeConcreteTensor(input_shape);
  fusion->addInput(tv0);
  TensorView* tv1 = sum(tv0, {1});
  TensorView* tv2 = sum(tv1, {0});
  TensorView* tv3 = sum(tv2, {0});
  fusion->addOutput(tv3);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0(false), stage1(false);
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});

  stage0.mesh = {0, 1, 2, 3};
  stage1.mesh = {0, 1, 2, 3};

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  pipeline = std::make_unique<Pipeline>(fusion.get(), std::move(descriptor));

  inputs = {at::ones(input_shape, tensor_options) * (communicator->deviceId() + 1)};

  validate();
}

TEST_F(PipelineTest, Pipeline_ReduceScatter) {
  const std::vector<int64_t> input_shape = {4, 4, 1, 2};

  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeConcreteTensor(input_shape);
  fusion->addInput(tv0);
  TensorView* tv1 = sum(tv0, {3});
  TensorView* tv2 = sum(tv1, {0});
  TensorView* tv3 = sum(tv2, {1});
  fusion->addOutput(tv3);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);
  tv2->axis(1)->parallelize(ParallelType::DIDx); //axis(0) is the "reduce" axis from previous tensor
  tv3->axis(0)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0(false), stage1(false);
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});

  stage0.mesh = {0, 1, 2, 3};
  stage1.mesh = {0, 1, 2, 3};

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  pipeline = std::make_unique<Pipeline>(fusion.get(), std::move(descriptor));

  inputs = {at::ones(input_shape, tensor_options) * (communicator->deviceId() + 1)};

  validate();
}


TEST_F(PipelineTest, Overlap) {
  // In this example we demonstrate how we can apply the optimization
  // described in 
  // Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models (acm.org)
  // https://dl.acm.org/doi/pdf/10.1145/3567955.3567959
  // We simplify the setting as much as possible by considering a multi-device Pipeline with
  // a simple "Gather" followed by a dependent compute. The paper suggest to slice those
  // two operation and to interleave them to achieve better overlap. Consider the following Pipeline:

  // /* Stage 0 */
  // TensorView* tv0 = makeContigTensor(3);
  // fusion->addInput(tv0);
  // TensorView* tv1 = sum(tv0, {2});
  // /* Stage 1 */
  // TensorView* tv2 = set(tv1); // is lowered to a "Gather" communication
  // TensorView* tv3 = sum(tv2, {1});
  // fusion->addOutput(tv3);

  // tv0->axis(0)->parallelize(ParallelType::DIDx);
  // tv1->axis(0)->parallelize(ParallelType::DIDx);

  // PipelineStageDescriptor stage0(false), stage1(false);
  // stage0.addVal({tv0, tv1});
  // stage1.addVal({tv2, tv3});

  // stage0.mesh = {0, 1, 2, 3, 4, 5, 6, 7};
  // stage1.mesh = {0};

  const int64_t number_of_devices = communicator->size();
  constexpr int64_t number_of_slices = 4;
  constexpr int64_t extent_of_axis2 = 1024;
  constexpr int64_t extent_of_slice = extent_of_axis2 / number_of_slices;
  const std::vector<int64_t> input_extents = {number_of_devices, 7, extent_of_axis2, 3};
  assert(!(extent_of_axis2 % number_of_slices)); // for simplicity

  FusionGuard fg(fusion.get());

  PipelineStageDescriptor stage0(false), stage1(false);
  // containers used later for adding ranges of tvs directly to the stages
  std::unordered_set<Val*> from_stage0, from_stage1;
  std::vector<Val*> to_stage0, to_stage1;

  TensorView* tv0 = makeConcreteTensor(input_extents);
  fusion->addInput(tv0);
  TensorView* tv1 = sum(tv0, {3});
  from_stage0.insert(tv0);
  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  TensorView *tv1x, *tv2x, *tv3x;
  std::vector<TensorView*> tv3_slices;
  std::vector<Slice> slices {3};
  for (int i = 0; i < number_of_slices; i++) {
    slices.at(2).start = IrBuilder::create<Val>(i * extent_of_slice);
    slices.at(2).stop = IrBuilder::create<Val>((i+1) * extent_of_slice);
    tv1x = slice(tv1, slices);
    tv1x->axis(0)->parallelize(ParallelType::DIDx);
    to_stage0.push_back(tv1x);

    tv2x = set(tv1x);
    from_stage1.insert(tv2x);
    tv3x = sum(tv2x, {1});
    tv3_slices.push_back(tv3x);
  }
  TensorView* tv3 = cat(tv3_slices, 1);
  fusion->addOutput(tv3);
  to_stage1.push_back(tv3);

  //instead of using "slice/cat" it would be nicer to split the dimension and use "select/stack", but "stack" is not implemented in nvFuser at the moment

  stage0.addRange(fusion.get(), from_stage0, to_stage0);
  stage1.addRange(fusion.get(), from_stage1, to_stage1);
  std::vector<DeviceIdxType> devices(number_of_devices);
  std::iota(devices.begin(), devices.end(), 0);
  stage0.mesh = devices;
  stage1.mesh = {0};

  PipelineDescriptor descriptor {
      .stage_descriptors{std::move(stage0), std::move(stage1)}};

  pipeline = std::make_unique<Pipeline>(fusion.get(), std::move(descriptor));

  inputs = {at::ones(input_extents, tensor_options) * (communicator->deviceId() + 1)};

  validate();
}

TEST_F(NVFuserTest, ReshardingDetection) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  DeviceMesh mesh0,mesh1; 
  mesh0 = {0,1};
  mesh1 = {0,2};
  mesh2 = {0,1,2};

  TensorView* tv0 = makeContigTensor(3);
  tv0->setDeviceMesh(&mesh0);

  TensorView* tv1 = set(tv0);
  tv1->setDeviceMesh(&mesh0);

  TensorView* tv2 = set(tv1); // resharding
  tv2->setDeviceMesh(&mesh1);

  TensorView* tv3 = set(tv2);
  tv3->setDeviceMesh(&mesh1);

  TensorView* tv4 = set(tv3); // resharding
  tv4->setDeviceMesh(&mesh2);

  TensorView* tv5 = set(tv4); // resharding
  tv5->setDeviceMesh(&mesh2);
  tv5->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv6 = set(tv5);
  tv6->setDeviceMesh(&mesh2);
  tv6->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv7 = set(tv6); // resharding
  tv7->setDeviceMesh(&mesh2);

  TensorView* tv8 = sum(tv0, {0});
  tv8->setDeviceMesh(&mesh0);

  TensorView* tv9 = sum(tv0, {0}); //resharding, but seems invalid
  tv9->setDeviceMesh(&mesh0);
  tv9->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv10 = sum(tv0, {0}); //resharding,
  tv10->setDeviceMesh(&mesh0);
  tv10->axis(1)->parallelize(ParallelType::DIDx);

  TensorView* tv11 = sum(tv0, {0}); //resharding,
  tv11->setDeviceMesh(&mesh1);

  TensorView* tv12 = sum(tv5, {0}); // resharding
  tv12->setDeviceMesh(&mesh2);
  tv12->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv13 = sum(tv5, {0}); // resharding
  tv13->setDeviceMesh(&mesh2);
  tv13->axis(1)->parallelize(ParallelType::DIDx);

  TensorView* tv14 = sum(tv5, {0}); // resharding
  tv14->setDeviceMesh(&mesh2);
  tv14->axis(0)->parallelize(ParallelType::DIDx);
  tv14->axis(1)->parallelize(ParallelType::DIDx);

  TensorView* tv15 = add(tv0, tv1);
  tv15->setDeviceMesh(&mesh0);

  TensorView* tv16 = add(tv0, tv1); //resharding
  tv16->setDeviceMesh(&mesh1);

  TensorView* tv17 = add(tv0, tv1); //resharding
  tv17->setDeviceMesh(&mesh0);
  tv17->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv18 = add(tv5, tv6);
  tv18->setDeviceMesh(&mesh2);
  tv18->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv19 = add(tv5, tv7); // resharding
  tv19->setDeviceMesh(&mesh2);
  tv19->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv20 = add(tv5, tv7); // resharding
  tv20->setDeviceMesh(&mesh2);

  TensorView* tv21 = add(tv0, tv7); // resharding
  tv21->setDeviceMesh(&mesh2);

  TensorView* tv22 = sum(tv5, {1});
  tv22->setDeviceMesh(&mesh2);
  tv22->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv23 = sum(tv5, {1}); //resharding
  tv23->setDeviceMesh(&mesh2);

  TensorView* tv24 = sum(tv7, {0});
  tv24->setDeviceMesh(&mesh2);

  TensorView* tv25 = sum(tv7, {0}); //not resharding but invalid
  tv25->setDeviceMesh(&mesh2);
  tv22->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv26 = add(tv5, tv6); // resharding
  tv26->setDeviceMesh(&mesh2);

  GTEST_EXPECT_TRUE(!tv1->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv2->definition()->isResharding());
  GTEST_EXPECT_TRUE(!tv3->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv4->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv5->definition()->isResharding());
  GTEST_EXPECT_TRUE(!tv6->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv7->definition()->isResharding());
  GTEST_EXPECT_TRUE(!tv8->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv9->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv10->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv11->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv12->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv13->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv14->definition()->isResharding());
  GTEST_EXPECT_TRUE(!tv15->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv16->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv17->definition()->isResharding());
  GTEST_EXPECT_TRUE(!tv18->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv19->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv20->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv21->definition()->isResharding());
  GTEST_EXPECT_TRUE(!tv22->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv23->definition()->isResharding());
  GTEST_EXPECT_TRUE(!tv24->definition()->isResharding());
  GTEST_EXPECT_TRUE(!tv25->definition()->isResharding());
  GTEST_EXPECT_TRUE(tv26->definition()->isResharding());
}

class automaticSetInsertionTest : public NVFuserTest {
protected:
  void SetUp() override {
    fusion = std::make_unique<Fusion>();
    fg = std::make_unique<FusionGuard>(fusion.get());
  }
  void validate() {
    for (auto expr: fusion->exprs()) {
      bool is_valid = !expr->isResharding()
              || (expr->isA<LoadStoreOp>()
                  && (expr->as<LoadStoreOp>()->opType() == LoadStoreOpType::Set))
              || expr->isA<ReductionOp>();
      GTEST_EXPECT_TRUE(is_valid);
    }
  }

  std::unique_ptr<Fusion> fusion;
  std::unique_ptr<FusionGuard> fg;
  // TensorView* (&unary_op)(TensorView*);
};

TEST_F(automaticSetInsertionTest, unary_ops) {
  DeviceMesh mesh0,mesh1;
  mesh0 = {0};
  mesh1 = {1};

  TensorView* tv0 = makeContigTensor(3);
  tv0->setDeviceMesh(&mesh0);
  TensorView* tv1 = exp(tv0);
  tv1->setDeviceMesh(&mesh1);
  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  std::cout << "before transformation";
  fusion->print();

  std::cout << "after transformation";
  insertSetBeforeReshardingExpr(fusion.get());
  fusion->print();

}

} // namespace nvfuser

#endif
