// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef NVFUSER_DISTRIBUTED
#include <disjoint_set.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <gtest/gtest.h>
#include <multidevice/executor.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <test/multidevice.h>
#include <test/validator.h>

namespace nvfuser {

// Test single device computation with device parallelized axis is handled
// properly.
TEST(NVFuserTest, TestShardedComputeIndex) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DeviceMesh mesh({0});

  TensorView* a = makeConcreteTensor({4, 1, 2, 5});
  TensorView* b = sum(a, {0});
  TensorView* c = add(a, a);
  TensorView* d = permute(a, {1, 0, 2, 3});

  fusion->addInput(a);
  fusion->addOutput(b);
  fusion->addOutput(c);
  fusion->addOutput(d);

  a->setDeviceMesh(mesh);
  b->setDeviceMesh(mesh);
  c->setDeviceMesh(mesh);
  d->setDeviceMesh(mesh);
  a->axis(1)->parallelize(ParallelType::DIDx);
  b->axis(1)->parallelize(ParallelType::DIDx);
  c->axis(1)->parallelize(ParallelType::DIDx);
  d->axis(0)->parallelize(ParallelType::DIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto a_ = at::randn({4, 1, 2, 5}, options);
  auto b_ = at::sum(a_, {0});
  auto c_ = a_ + a_;
  auto d_ = at::permute(a_, {1, 0, 2, 3});
  std::vector<c10::IValue> inputs_ = {a_};
  std::vector<at::Tensor> outputs_ = {b_, c_, d_};

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), inputs_);
  auto outputs = fe.runFusion(inputs_);
  testValidate(fusion.get(), outputs, inputs_, outputs_, __LINE__, __FILE__);
}

class ShardingTest : public MultiDeviceTest,
                     public ::testing::WithParamInterface<bool> {};

// Test memory allocation of multidevice fusion with unsharded inputs
// and sharded intermediates, outputs.
TEST_P(ShardingTest, UnshardedGlobalInput) {
  auto concreteTv = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  DeviceMesh mesh(devices);
  std::vector<int64_t> input_size = {num_devices, 3};

  TensorView* tv0 =
      concreteTv ? makeConcreteTensor(input_size) : makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = add(tv1, tv1);
  TensorView* tv3 = sum(tv2, {1});
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);

  tv1->axis(0)->parallelize(ParallelType::DIDx);
  tv2->axis(0)->parallelize(ParallelType::DIDx);
  tv3->axis(0)->parallelize(ParallelType::DIDx);

  std::vector<TensorView*> tvs = {tv0, tv1, tv2, tv3};
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
  }

  auto x0 = at::randn(input_size, tensor_options);
  std::vector<c10::IValue> inputs = {x0};
  auto x1 = shardTensor(x0, mesh, communicator->deviceId());
  auto x2 = x1 + x1;
  auto x3 = at::sum(x2, {1});

  MultiDeviceExecutor runtime(std::move(fusion), *communicator);
  auto outputs = runtime.runWithInput(inputs);

  testValidate(
      runtime.completeFusion(),
      outputs,
      inputs,
      {x1, x2, x3},
      __LINE__,
      __FILE__);
}

// Test memory allocation of multidevice fusion with sharded input
// and replicated intermediates and output.
TEST_P(ShardingTest, ShardGlobalInput) {
  auto concreteTv = GetParam();
  int sharded_dim = 0;
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  DeviceMesh mesh(devices);
  std::vector<int64_t> unsharded_input_size = {num_devices, 3, 2};

  TensorView* tv0 = concreteTv ? makeConcreteTensor(unsharded_input_size)
                               : makeContigTensor(3);
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = add(tv1, tv1);
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  fusion->addOutput(tv2);

  tv0->axis(sharded_dim)->parallelize(ParallelType::DIDx);

  std::vector<TensorView*> tvs = {tv0, tv1, tv2};
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
  }

  auto x1 = at::randn(unsharded_input_size, tensor_options);
  std::vector<c10::IValue> inputs = {
      shardTensor(x1, mesh, communicator->deviceId())};
  auto x2 = x1 * 2;

  MultiDeviceExecutor runtime(std::move(fusion), *communicator);
  auto outputs = runtime.runWithInput(inputs);
  testValidate(
      runtime.completeFusion(), outputs, inputs, {x1, x2}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(ConcreteInput, ShardingTest, ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(SymbolicInput, ShardingTest, ::testing::Values(false));

} // namespace nvfuser
#endif
