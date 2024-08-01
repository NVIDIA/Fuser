// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <fusion.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>

namespace nvfuser {

namespace hir {

using MultiDeviceHostIrTestParams = std::tuple<bool, bool>;

class MultiDeviceHostIrTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<MultiDeviceHostIrTestParams> {};

// This file implements test that combine multidevice communications and host
// irs. See test_host_irs.cpp for an introduction on host irs and a summary of
// the different steps necessary to write a host program.

// The host program of the first test could be illustrated as follows:

// tv0_fusion: input, sharded accross devices on its first dimension

// tv1_fusion = Fusion0 (tv0_fusion), on each device

// tv2 = Allgather(tv1_fusion)

// tv2: output

// Note that the Fusion may or may not be multi-device scheduled for achieving
// the same result. We test both cases.

TEST_P(MultiDeviceHostIrTest, SingleFusionSingleComm) {
  auto [use_fusion_executor_cache, with_sharding_annotations] = GetParam();

  const int64_t communicator_size = communicator_->size();
  std::vector<int64_t> unsharded_input_sizes = {communicator_size, 8, 32};
  std::vector<int64_t> sharded_input_sizes = unsharded_input_sizes;
  sharded_input_sizes[0] = 1;

  // [Step 1] Define the Fusion we want to execute
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0_fusion = makeConcreteTensor(
      with_sharding_annotations ? unsharded_input_sizes : sharded_input_sizes);
  auto tv1_fusion = add(tv0_fusion, tv0_fusion);
  fusion->addInput(tv0_fusion);
  fusion->addOutput(tv1_fusion);

  DeviceMesh mesh = DeviceMesh::createForNumDevices(communicator_size);
  for (auto tv : {tv0_fusion, tv1_fusion}) {
    tv->setDeviceMesh(mesh);
    if (with_sharding_annotations) {
      tv->axis(0)->parallelize(ParallelType::DIDx);
    }
  }

  // [Step 2)] Instantiate an HostIrContainer
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard::setCurFusion(hic.get());

  // [Step 3a)] Create a HostUnit Ir holding the fusions
  auto hu = IrBuilder::create<HostUnit>(std::move(fusion));

  // [Step 4)] Create TensorViews at the Host level
  IrCloner ir_cloner(hic.get());
  auto tv0 = ir_cloner.clone(
      hu->fusion_to_execute()->inputs().at(0)->as<TensorView>());
  auto tv1 = ir_cloner.clone(
      hu->fusion_to_execute()->outputs().at(0)->as<TensorView>());
  auto tv2 = makeConcreteTensor(unsharded_input_sizes);
  tv2->setDeviceMesh(mesh);

  // [Step 5)a.] Create PostOnStream Irs representing executing the Fusion
  std::vector<Val*> compute_inputs = {tv0};
  std::vector<Val*> compute_outputs = {tv1};
  auto post_compute =
      IrBuilder::create<PostOnStream>(hu, compute_inputs, compute_outputs);
  // [Step 5)b.] Create Communication Ir representing executing the Fusion
  auto communication_input = tv1->as<TensorView>();
  auto communication_output = tv2->as<TensorView>();

  auto communication = IrBuilder::create<Communication>(
      CommunicationType::Allgather,
      communication_output,
      communication_input,
      mesh.vector());
  auto wait = IrBuilder::create<Wait>(communication);

  // [Step 6)] Define the Host program
  hic->pushBackTopLevelExprs(post_compute);
  hic->pushBackTopLevelExprs(communication);
  hic->pushBackTopLevelExprs(wait);

  // [Step 7)] Define the Host program's global I/O
  hic->addInput(post_compute->inputs().back());
  hic->addOutput(communication->outputs().back());

  // [Step 8)] Execute the Host program
  HostIrExecutorParams params;
  params.use_fusion_executor_cache = use_fusion_executor_cache;
  if (with_sharding_annotations && use_fusion_executor_cache) {
    // sharding + autoscheduler is not supported at this point
    params.skip_auto_scheduling = true;
  }
  HostIrExecutor hie(std::move(hic), communicator_, params);

  auto options = at::TensorOptions().device(communicator_->device());
  at::Tensor unsharded_input = at::randn(unsharded_input_sizes, options);
  c10::IValue input = unsharded_input.slice(
      0, communicator_->deviceId(), communicator_->deviceId() + 1);
  at::Tensor output = at::empty(unsharded_input_sizes, options);
  auto ref_output = unsharded_input * 2;

  auto outputs = hie.runWithInput(
      {{post_compute->inputs().back(), input},
       {communication->outputs().back(), output}});

  // validate the obtained results
  EXPECT_TRUE(torch::allclose(ref_output, outputs.back()));
}

TEST_P(MultiDeviceHostIrTest, SingleCommTwoFusionAndWait) {
  auto [use_fusion_executor_cache, with_sharding_annotations] = GetParam();

  const int64_t communicator_size = communicator_->size();
  std::vector<int64_t> unsharded_input_sizes = {communicator_size, 8, 32};
  std::vector<int64_t> sharded_input_sizes = unsharded_input_sizes;
  sharded_input_sizes[0] = 1;

  // [Step 1] Define the Fusion we want to execute
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0_fusion = makeConcreteTensor(
      with_sharding_annotations ? unsharded_input_sizes : sharded_input_sizes);
  auto tv1_fusion = add(tv0_fusion, tv0_fusion);
  fusion->addInput(tv0_fusion);
  fusion->addOutput(tv1_fusion);

  DeviceMesh mesh = DeviceMesh::createForNumDevices(communicator_size);
  for (auto tv : {tv0_fusion, tv1_fusion}) {
    tv->setDeviceMesh(mesh);
    if (with_sharding_annotations) {
      tv->axis(0)->parallelize(ParallelType::DIDx);
    }
  }

  // [Step 2)] Instantiate an HostIrContainer
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard::setCurFusion(hic.get());

  // [Step 3a)] Create a HostUnit Ir holding the fusions
  auto hu = IrBuilder::create<HostUnit>(std::move(fusion));

  // [Step 4)] Create TensorViews at the Host level
  IrCloner ir_cloner(hic.get());
  auto tv0 = ir_cloner.clone(
      hu->fusion_to_execute()->inputs().at(0)->as<TensorView>());
  auto tv1 = ir_cloner.clone(
      hu->fusion_to_execute()->outputs().at(0)->as<TensorView>());
  auto tv2 = makeConcreteTensor(unsharded_input_sizes);
  tv2->setDeviceMesh(mesh);

  // [Step 5)a.] Create PostOnStream Irs representing executing the Fusion
  std::vector<Val*> compute_inputs = {tv0};
  std::vector<Val*> compute_outputs = {tv1};
  auto post_compute =
      IrBuilder::create<PostOnStream>(hu, compute_inputs, compute_outputs);
  // [Step 5)b.] Create Communication Ir representing executing the Fusion
  TensorView* communication_input = tv1->as<TensorView>();
  TensorView* communication_output = tv2->as<TensorView>();
  auto communication = IrBuilder::create<Communication>(
      CommunicationType::Allgather,
      communication_output,
      communication_input,
      mesh.vector());
  auto wait = IrBuilder::create<Wait>(communication);

  // [Step 6)] Define the Host program
  hic->pushBackTopLevelExprs(post_compute);
  hic->pushBackTopLevelExprs(communication);
  hic->pushBackTopLevelExprs(wait);
  hic->pushBackTopLevelExprs(post_compute);
  hic->pushBackTopLevelExprs(communication);
  hic->pushBackTopLevelExprs(post_compute);
  hic->pushBackTopLevelExprs(wait);
  hic->pushBackTopLevelExprs(post_compute);

  // [Step 7)] Define the Host program's global I/O
  hic->addInput(post_compute->inputs().back());
  hic->addOutput(communication->outputs().back());

  // [Step 8)] Execute the Host program
  HostIrExecutorParams params;
  params.use_fusion_executor_cache = use_fusion_executor_cache;
  if (with_sharding_annotations && use_fusion_executor_cache) {
    // sharding + autoscheduler is not supported at this point
    params.skip_auto_scheduling = true;
  }
  HostIrExecutor hie(std::move(hic), communicator_, params);

  auto options = at::TensorOptions().device(communicator_->device());
  at::Tensor unsharded_input = at::randn(unsharded_input_sizes, options);
  c10::IValue input = unsharded_input.slice(
      0, communicator_->deviceId(), communicator_->deviceId() + 1);
  at::Tensor output = at::empty(unsharded_input_sizes, options);
  auto ref_output = unsharded_input * 2;

  auto outputs = hie.runWithInput(
      {{post_compute->inputs().back(), input},
       {communication->outputs().back(), output}});

  // validate the obtained results
  EXPECT_TRUE(torch::allclose(ref_output, outputs.back()));
}

INSTANTIATE_TEST_SUITE_P(
    Manual,
    MultiDeviceHostIrTest,
    testing::Combine(testing::Bool(), testing::Bool()),
    [](const testing::TestParamInfo<MultiDeviceHostIrTestParams>& info)
        -> std::string {
      std::string s;
      s += std::get<0>(info.param) ? "useFusionExecutorCache"
                                   : "useFusionExecutor";
      s += "_";
      s += std::get<1>(info.param) ? "withShardingAnnotations"
                                   : "withoutShardingAnnotations";
      return s;
    });

} // namespace hir

} // namespace nvfuser
