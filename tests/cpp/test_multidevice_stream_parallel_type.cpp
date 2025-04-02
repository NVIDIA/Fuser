// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <cuda_profiler_api.h>
#include <fusion.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <preseg_passes/reorder_sharded_axis.h>
#include <tests/cpp/multidevice.h>

namespace nvfuser {


using MultiDeviceStreamParallelTypeTest = MultiDeviceTest;

TEST_F(MultiDeviceStreamParallelTypeTest, Allgather) {

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  const DeviceMesh mesh = DeviceMesh::createForNumDevices(communicator_->size());
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);
  tv0->axis(1)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), *communicator_);

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 2);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<ForLoop>());

  auto options = at::TensorOptions().device(at::kCUDA, communicator_->deviceId());
  at::Tensor unsharded_input = at::rand({4, communicator_->size()}, options);
  at::Tensor input = shardTensor(unsharded_input, /*axis=*/1, mesh);
  auto output =
      executor.runWithInput(KernelArgumentHolder({input}))[0].as<at::Tensor>();

  EXPECT_TRUE(torch::allclose(output, unsharded_input, 1e-2, 1e-2))
      << "Output: " << output << "\nExpected: " << unsharded_input;
}

TEST_F(MultiDeviceStreamParallelTypeTest, Allreduce) {

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(3);
  TensorView* tv1 = sum(tv0, {1});
  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  const DeviceMesh mesh = DeviceMesh::createForNumDevices(communicator_->size());
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);
  tv0->axis(1)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), *communicator_);

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 2);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<ForLoop>());

  auto options = at::TensorOptions().device(at::kCUDA, communicator_->deviceId());
  at::Tensor unsharded_input = at::rand({4, communicator_->size(), 8}, options);
  at::Tensor input = shardTensor(unsharded_input, /*axis=*/1, mesh);
  auto output =
      executor.runWithInput(KernelArgumentHolder({input}))[0].as<at::Tensor>();

  auto expected_output = unsharded_input.sum(1);
  EXPECT_TRUE(torch::allclose(output, expected_output, 1e-2, 1e-2))
      << "Output: " << output << "\nExpected: " << expected_output;
}

TEST_F(MultiDeviceStreamParallelTypeTest, ReduceScatter) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(4);
  TensorView* tv1 = sum(tv0, {1});
  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  const DeviceMesh mesh = DeviceMesh::createForNumDevices(communicator_->size());
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);
  tv0->axis(1)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::Stream);
  tv1->axis(2)->parallelize(ParallelType::DIDx);

  MultiDeviceExecutor executor(std::move(fusion), *communicator_);

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_EQ(container->topLevelExprs().size(), 2);
  EXPECT_TRUE(container->topLevelExprs().at(0)->isA<kir::Allocate>());
  EXPECT_TRUE(container->topLevelExprs().at(1)->isA<ForLoop>());

  auto options = at::TensorOptions().device(at::kCUDA, communicator_->deviceId());
  at::Tensor unsharded_input = at::rand({4, communicator_->size(), communicator_->size(), 8}, options);
  at::Tensor input = shardTensor(unsharded_input, /*axis=*/1, mesh);
  auto output =
      executor.runWithInput(KernelArgumentHolder({input}))[0].as<at::Tensor>();

  auto expected_output = shardTensor(unsharded_input.sum(1), /*axis=*/1, mesh);
  EXPECT_TRUE(torch::allclose(output, expected_output, 1e-2, 1e-2))
      << "Output: " << output << "\nExpected: " << expected_output;
}

} // namespace nvfuser