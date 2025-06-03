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
#include <tests/cpp/validator.h>

namespace nvfuser {

using testing::ElementsAre;
using testing::SizeIs;

using MultiDeviceStreamParallelTypeTest = MultiDeviceTest;

TEST_F(MultiDeviceStreamParallelTypeTest, Allgather) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  const DeviceMesh mesh =
      DeviceMesh::createForNumDevices(communicator_->size());
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);
  tv0->axis(1)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), *communicator_);

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_THAT(
      container->topLevelExprs(),
      ElementsAre(
          IsA<kir::Allocate>(),
          IsA<hir::GetCurrentStream>(),
          IsA<ForLoop>(),
          IsA<ForLoop>()));

  auto options =
      at::TensorOptions().device(at::kCUDA, communicator_->deviceId());
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

  const DeviceMesh mesh =
      DeviceMesh::createForNumDevices(communicator_->size());
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);
  tv0->axis(1)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), *communicator_);

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_THAT(
      container->topLevelExprs(),
      ElementsAre(
          IsA<kir::Allocate>(),
          IsA<hir::GetCurrentStream>(),
          IsA<ForLoop>(),
          IsA<ForLoop>()));

  auto options =
      at::TensorOptions().device(at::kCUDA, communicator_->deviceId());
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

  const DeviceMesh mesh =
      DeviceMesh::createForNumDevices(communicator_->size());
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);
  tv0->axis(1)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::Stream);
  tv1->axis(2)->parallelize(ParallelType::DIDx);

  MultiDeviceExecutor executor(std::move(fusion), *communicator_);

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_THAT(
      container->topLevelExprs(),
      ElementsAre(
          IsA<kir::Allocate>(),
          IsA<hir::GetCurrentStream>(),
          IsA<ForLoop>(),
          IsA<ForLoop>()));

  auto options =
      at::TensorOptions().device(at::kCUDA, communicator_->deviceId());
  at::Tensor unsharded_input =
      at::rand({4, communicator_->size(), communicator_->size(), 8}, options);
  at::Tensor input = shardTensor(unsharded_input, /*axis=*/1, mesh);
  auto output =
      executor.runWithInput(KernelArgumentHolder({input}))[0].as<at::Tensor>();

  auto expected_output = shardTensor(unsharded_input.sum(1), /*axis=*/1, mesh);
  EXPECT_TRUE(torch::allclose(output, expected_output, 1e-2, 1e-2))
      << "Output: " << output << "\nExpected: " << expected_output;
}

TEST_F(MultiDeviceStreamParallelTypeTest, AG_matmul) {
  constexpr int64_t M = 32768;
  constexpr int64_t K = 32768;
  constexpr int64_t N = 1024;
  constexpr int64_t S = 8;
  const int64_t D = communicator_->size();
  if (M % (D * S) != 0) {
    GTEST_SKIP() << "M must be a multiple of D * S, but got M = " << M
                 << ", D = " << D << ", S = " << S;
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(4); //[S, DIDx(D), M/(S*D), K]
  TensorView* tv1 = makeContigTensor(2); //[K, N]
  TensorView* tv2 = matmul(tv0, tv1); //[Stream(S), D, M/(S*D), N]

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv2);

  auto mesh = DeviceMesh::createForNumDevices(D);
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);
  tv2->setDeviceMesh(mesh);

  tv0->axis(1)->parallelize(ParallelType::DIDx);
  tv2->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), *communicator_);

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_THAT(
      container->topLevelExprs(),
      ElementsAre(
          IsA<kir::Allocate>(),
          IsA<kir::Allocate>(),
          IsA<hir::GetCurrentStream>(),
          IsA<ForLoop>(),
          IsA<ForLoop>()));

  auto tensor_options =
      at::TensorOptions().dtype(at::kFloat).device(communicator_->device());
  auto t0_unsharded = at::randn({S, D, M / (S * D), K}, tensor_options);
  auto t0 = t0_unsharded.slice(
      1, communicator_->deviceId(), communicator_->deviceId() + 1);
  auto t1 = at::randn({K, N}, tensor_options);

  auto t2 = executor.runWithInput({t0, t1})[0].as<at::Tensor>();

  auto t2_ref = at::matmul(t0_unsharded, t1);
  EXPECT_TRUE(torch::allclose(t2_ref, t2, 1e-2, 1e-2));
}

TEST_F(MultiDeviceStreamParallelTypeTest, matmul_AR) {
  constexpr int64_t M = 32768;
  constexpr int64_t K = 32768;
  constexpr int64_t N = 1024;
  constexpr int64_t S = 8;
  const int64_t D = communicator_->size();
  if (M % S != 0) {
    GTEST_SKIP() << "M must be a multiple of S, but got M = " << M
                 << ", S = " << S;
  }
  if (K % D != 0) {
    GTEST_SKIP() << "K must be a multiple of D, but got K = " << K
                 << ", D = " << D;
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(4); //[S, DIDx(D), M/S, K/D]
  TensorView* tv1 = makeContigTensor(3); //[DIDx(D), K/D, N]
  TensorView* tv2_unreduced = matmul(tv0, tv1); //[Stream(S), DIDx(D), M/S, N]
  TensorView* tv2 = sum(tv2_unreduced, {1}); //[Stream(S), r(D), M/S, N]

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv2);

  auto mesh = DeviceMesh::createForNumDevices(D);
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);
  tv2_unreduced->setDeviceMesh(mesh);
  tv2->setDeviceMesh(mesh);

  tv0->axis(1)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);
  tv2_unreduced->axis(0)->parallelize(ParallelType::Stream);
  tv2_unreduced->axis(1)->parallelize(ParallelType::DIDx);
  tv2->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), *communicator_);

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_THAT(
      container->topLevelExprs(),
      ElementsAre(
          IsA<kir::Allocate>(),
          IsA<kir::Allocate>(),
          IsA<hir::GetCurrentStream>(),
          IsA<ForLoop>(),
          IsA<ForLoop>()));

  auto tensor_options =
      at::TensorOptions().dtype(at::kFloat).device(communicator_->device());
  auto t0_unsharded = at::randn({S, D, M / S, K / D}, tensor_options);
  auto t1_unsharded = at::randn({D, K / D, N}, tensor_options);
  auto t0 = shardTensor(t0_unsharded, /*axis=*/1, mesh);
  auto t1 = shardTensor(t1_unsharded, /*axis=*/0, mesh);

  auto t2 = executor.runWithInput({t0, t1})[0].as<at::Tensor>();

  auto t2_ref = at::sum(at::matmul(t0_unsharded, t1_unsharded), {1});
  EXPECT_TRUE(torch::allclose(t2_ref, t2, 1e-2, 1e-2));
}

TEST_F(MultiDeviceStreamParallelTypeTest, matmul_RS_through_bcast) {
  constexpr int64_t M = 32;
  constexpr int64_t K = 8;
  constexpr int64_t N = 2;
  constexpr int64_t S = 4;
  const int64_t D = communicator_->size();
  if (M % (S * D) != 0) {
    GTEST_SKIP() << "M must be a multiple of S * D, but got M = " << M
                 << ", S = " << S << ", D = " << D;
  }
  if (K % D != 0) {
    GTEST_SKIP() << "K must be a multiple of D, but got K = " << K
                 << ", D = " << D;
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(5); //[S, DIDx(D), D, M/(S*D), K/D]
  TensorView* tv1 = makeContigTensor(3); //[DIDx(D), K/D, N]
  TensorView* tv1b = broadcast(
      tv1, {true, false, true, false, false}); //[1, DIDx(D), 1, K/D, N]
  TensorView* tv2_unreduced =
      matmul(tv0, tv1b); //[Stream(S), DIDx(D), D, M/S, N]
  TensorView* tv2 =
      sum(tv2_unreduced, {1}); //[Stream(S), r(D), DIDx(D), M/S, N]

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv2);

  auto mesh = DeviceMesh::createForNumDevices(D);
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);
  tv2_unreduced->setDeviceMesh(mesh);
  tv2->setDeviceMesh(mesh);

  tv0->axis(1)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);
  tv2_unreduced->axis(0)->parallelize(ParallelType::Stream);
  tv2_unreduced->axis(1)->parallelize(ParallelType::DIDx);
  tv2->axis(0)->parallelize(ParallelType::Stream);
  tv2->axis(2)->parallelize(ParallelType::DIDx);

  MultiDeviceExecutor executor(std::move(fusion), *communicator_);

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_THAT(
      container->topLevelExprs(),
      ElementsAre(
          IsA<hir::PostOnStream>(),
          IsA<kir::Allocate>(),
          IsA<kir::Allocate>(),
          IsA<hir::GetCurrentStream>(),
          IsA<ForLoop>(),
          IsA<ForLoop>()));

  auto tensor_options =
      at::TensorOptions().dtype(at::kFloat).device(communicator_->device());
  auto t0_unsharded = at::randn({S, D, D, M / (S * D), K / D}, tensor_options);
  auto t1_unsharded = at::randn({D, K / D, N}, tensor_options);
  auto t0 = shardTensor(t0_unsharded, /*axis=*/1, mesh);
  auto t1 = shardTensor(t1_unsharded, /*axis=*/0, mesh);

  auto t2 = executor.runWithInput({t0, t1})[0].as<at::Tensor>();

  auto t1b_unsharded =
      t1_unsharded.unsqueeze(1).unsqueeze(0); // {1, D, 1, K / D, N}
  auto t2_unreduced_unsharded =
      at::matmul(t0_unsharded, t1b_unsharded); // {S, D, D, M / (S * D), N}
  auto t2_unreduced =
      at::sum(t2_unreduced_unsharded, {1}); // {S, D, M / (S * D), N}
  auto t2_ref =
      shardTensor(t2_unreduced, /*axis=*/1, mesh); // {S, M / (S * D), N}
  EXPECT_TRUE(torch::allclose(t2_ref, t2, 1e-1, 1e-1))
      << "Output: " << t2 << " Expected: " << t2_ref;
}

TEST_F(MultiDeviceStreamParallelTypeTest, AllgatherP2p) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  const DeviceMesh mesh =
      DeviceMesh::createForNumDevices(communicator_->size());
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);
  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), *communicator_);

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_THAT(
      container->topLevelExprs(),
      ElementsAre(
          IsA<kir::Allocate>(),
          IsA<hir::GetCurrentStream>(),
          IsA<ForLoop>(),
          IsA<ForLoop>()));

  auto options =
      at::TensorOptions().device(at::kCUDA, communicator_->deviceId());
  at::Tensor unsharded_input = at::rand({communicator_->size(), 4}, options);
  at::Tensor input = shardTensor(unsharded_input, /*axis=*/0, mesh);
  auto output =
      executor.runWithInput(KernelArgumentHolder({input}))[0].as<at::Tensor>();

  EXPECT_TRUE(torch::allclose(output, unsharded_input, 1e-2, 1e-2))
      << "Output: " << output << "\nExpected: " << unsharded_input;
}

TEST_F(MultiDeviceStreamParallelTypeTest, AG_matmul_P2p) {
  constexpr int64_t M = 32768;
  constexpr int64_t K = 32768;
  constexpr int64_t N = 1024;
  const int64_t D = communicator_->size();
  if (M % D != 0) {
    GTEST_SKIP() << "M must be a multiple of D, but got M = " << M
                 << ", D = " << D;
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(3); //[DIDx(D), M/D, K]
  TensorView* tv1 = makeContigTensor(2); //[K, N]
  TensorView* tv2 = matmul(tv0, tv1); //[Stream(D), M/D, N]

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv2);

  auto mesh = DeviceMesh::createForNumDevices(D);
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);
  tv2->setDeviceMesh(mesh);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv2->axis(0)->parallelize(ParallelType::Stream);

  MultiDeviceExecutor executor(std::move(fusion), *communicator_);

  hir::HostIrContainer* container = executor.hostIrEvaluator()->container();
  EXPECT_THAT(
      container->topLevelExprs(),
      ElementsAre(
          IsA<kir::Allocate>(),
          IsA<kir::Allocate>(),
          IsA<hir::GetCurrentStream>(),
          IsA<ForLoop>(),
          IsA<ForLoop>()));

  auto tensor_options =
      at::TensorOptions().dtype(at::kFloat).device(communicator_->device());
  auto t0_unsharded = at::randn({D, M / D, K}, tensor_options);
  auto t0 = t0_unsharded.slice(
      0, communicator_->deviceId(), communicator_->deviceId() + 1);
  auto t1 = at::randn({K, N}, tensor_options);

  auto t2 = executor.runWithInput({t0, t1})[0].as<at::Tensor>();

  auto t2_ref = at::matmul(t0_unsharded, t1);
  EXPECT_TRUE(torch::allclose(t2_ref, t2, 1e-2, 1e-2));
}

} // namespace nvfuser
