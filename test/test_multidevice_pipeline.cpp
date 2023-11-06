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
  // ===========================================================
  //        FUSION
  // ===========================================================
  FusionGuard fg(fusion.get());

  TensorView* tv0_ = makeContigTensor(2);
  fusion->addInput(tv0_);
  TensorView* tv1_ = sum(tv0_, {0});

  TensorView* tv2_ = set(tv1_);
  TensorView* tv3_ = sum(tv2_, {0});
  fusion->addOutput(tv3_);

  TensorView* tv0 = makeContigTensor(3);
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
      at::randn({3096, 1123}, tensor_options),
      at::randn({2048, 73, 81}, tensor_options)};

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

} // namespace nvfuser

#endif
