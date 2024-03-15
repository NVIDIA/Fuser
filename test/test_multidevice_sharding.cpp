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

// params: concrete vs symbolic input, sharded axis
class ShardingTest
    : public MultiDeviceTest,
      public ::testing::WithParamInterface<std::tuple<bool, int>> {};

// Test memory allocation of multidevice fusion with unsharded inputs
// and sharded intermediates, outputs.
TEST_P(ShardingTest, UnshardedGlobalInput) {
  auto [concreteTv, sharded_dim] = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  DeviceMesh mesh(devices);
  std::vector<int64_t> input_size = {2, 3, 2, 4};
  input_size[sharded_dim] = num_devices;
  input_size[sharded_dim + 1] = num_devices;

  TensorView* tv0 =
      concreteTv ? makeConcreteTensor(input_size) : makeSymbolicTensor(4);
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = add(tv1, tv1);
  TensorView* tv3 = sum(tv2, {sharded_dim});

  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);

  tv1->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  tv2->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  tv3->axis(sharded_dim + 1)->parallelize(ParallelType::DIDx);

  std::vector<TensorView*> tvs = {tv0, tv1, tv2, tv3};
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
  }

  auto x0 = at::randn(input_size, tensor_options);
  std::vector<c10::IValue> inputs = {x0};
  auto x1 = shardTensor(x0, tv1, communicator->deviceId());
  auto x2 = x1 + x1;
  auto x3 = shardTensor(
      at::sum(x0 + x0, {sharded_dim}), tv3, communicator->deviceId());
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
  auto [concreteTv, sharded_dim] = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  DeviceMesh mesh(devices);
  std::vector<int64_t> unsharded_input_size = {3, 2, 5};
  unsharded_input_size[sharded_dim] = num_devices;

  TensorView* tv0 = concreteTv ? makeConcreteTensor(unsharded_input_size)
                               : makeSymbolicTensor(unsharded_input_size.size());
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = sum(tv1, {1});
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
      shardTensor(x1, tv0, communicator->deviceId())};
  auto x2 = at::sum(x1, {1});
  MultiDeviceExecutor runtime(std::move(fusion), *communicator);
  auto outputs = runtime.runWithInput(inputs);
  testValidate(
      runtime.completeFusion(), outputs, inputs, {x1, x2}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    ConcreteInput_OutermostShard,
    ShardingTest,
    ::testing::Values(std::make_tuple(true, 0)));

INSTANTIATE_TEST_SUITE_P(
    SymbolicInput_OutermostShard,
    ShardingTest,
    ::testing::Values(std::make_tuple(false, 0)));

} // namespace nvfuser
#endif
