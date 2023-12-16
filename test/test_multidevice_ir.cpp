// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <gtest/gtest.h>
#include <disjoint_set.h>
#include <test/multidevice.h>
#include <test/validator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ops/all_ops.h>
#include <multidevice/pipeline_ir.h>
#include <multidevice/runtime.h>
#include <multidevice/utils.h>
#include <index_compute.h>

namespace nvfuser {
TEST_F(MultiDeviceTest, ShardOuterAxisConcrete) {
  int sharded_dim = 0;
  Fusion fusion;
  FusionGuard fg(&fusion);
  DeviceMesh mesh({0, 1});
  int num_devices = 2;

  // TensorView* tv0 = makeContigTensor(2);
  // TODO: Concrete 2D tensor, symbolic 3D tensor work, but not symbolic 2D tensor. 
  // Generates NANs at tv3.
  TensorView* tv0 = makeConcreteTensor({2, 3});
  TensorView* tv1 = add(tv0, tv0);
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = add(tv2, tv2);
  TensorView* tv4 = set(tv3);
  TensorView* tv5 = sum(tv4, {0});
  fusion.addInput(tv0);
  // fusion.addOutput(tv2);
  // fusion.addOutput(tv3);
  fusion.addOutput(tv5);

  // TODO: split
  // tv3->split(sharded_dim, num_devices, false);
  tv2->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  // tv3->split(sharded_dim, num_devices, false);
  tv3->axis(sharded_dim)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0(false), stage1(false), stage2(false);
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});
  stage2.addVal({tv4, tv5});
  stage0.mesh = mesh;
  stage1.mesh = mesh;
  stage2.mesh = mesh;

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1), std::move(stage2)}};
  Pipeline pipeline(&fusion, std::move(descriptor));

  auto x = at::randn({num_devices, 3}, tensor_options);
  std::vector<c10::IValue> inputs = {x};
  auto ref_outputs = at::sum(x*4, {0});
  if (communicator->deviceId() == 0) {
    fusion.printKernel();
    fusion.printMath();
    std::cout << "Inputs " << x << std::endl;
    std::cout << "Expected " << ref_outputs << std::endl;
  }
  
  MultiDeviceRuntime runtime(&pipeline, *communicator);
  auto outputs = runtime.runWithInput(inputs);
  std::cout << "Outputs: " << std::endl;
  for (auto i : outputs)
    std::cout << i << std::endl;

  testValidate(&fusion, outputs, inputs, {ref_outputs}, __LINE__, __FILE__);
}
TEST_F(MultiDeviceTest, ShardOuterAxis) {
  int sharded_dim = 0;
  Fusion fusion;
  FusionGuard fg(&fusion);
  DeviceMesh mesh({0, 1});
  int num_devices = 2;

  TensorView* tv0 = makeContigTensor(2);
  // TODO: Concrete 2D tensor, symbolic 3D tensor work, but not symbolic 2D tensor. 
  // Generates NANs at tv3.
  // TensorView* tv0 = makeConcreteTensor({2, 3});
  TensorView* tv1 = add(tv0, tv0);
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = add(tv2, tv2);
  TensorView* tv4 = set(tv3);
  TensorView* tv5 = sum(tv4, {0});
  fusion.addInput(tv0);
  // fusion.addOutput(tv2);
  // fusion.addOutput(tv3);
  fusion.addOutput(tv5);

  // TODO: split
  // tv3->split(sharded_dim, num_devices, false);
  tv2->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  // tv3->split(sharded_dim, num_devices, false);
  tv3->axis(sharded_dim)->parallelize(ParallelType::DIDx);

  PipelineStageDescriptor stage0(false), stage1(false), stage2(false);
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});
  stage2.addVal({tv4, tv5});
  stage0.mesh = mesh;
  stage1.mesh = mesh;
  stage2.mesh = mesh;

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1), std::move(stage2)}};
  Pipeline pipeline(&fusion, std::move(descriptor));

  auto x = at::randn({num_devices, 2}, tensor_options);
  std::vector<c10::IValue> inputs = {x};
  auto ref_outputs = at::sum(x*4, {0});
  if (communicator->deviceId() == 0) {
    fusion.printKernel();
    fusion.printMath();
    std::cout << "Inputs " << x << std::endl;
    std::cout << "Expected " << ref_outputs << std::endl;
  }
  
  MultiDeviceRuntime runtime(&pipeline, *communicator);
  auto outputs = runtime.runWithInput(inputs);
  std::cout << "Outputs: " << std::endl;
  for (auto i : outputs)
    std::cout << i << std::endl;

  testValidate(&fusion, outputs, inputs, {ref_outputs}, __LINE__, __FILE__);
}
}
#endif