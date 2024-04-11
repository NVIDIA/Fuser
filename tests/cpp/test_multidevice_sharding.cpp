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
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

// TODO: This test checks that isSharded generates an error when a split/merged
// axis is parallelized with DIDx. Update when this restriction is lifted.
TEST_F(NVFuserTest, TestIsSharded) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

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

class ShardedComputeTest : public NVFuserTest,
                           public testing::WithParamInterface<bool> {};

TEST_P(ShardedComputeTest, ComputeIndex) {
  auto creates_concrete_tensor = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DeviceMesh mesh({0, 1, 2});

  TensorView* a = creates_concrete_tensor ? makeConcreteTensor({4, 2, 3, 5})
                                          : makeSymbolicTensor(4);
  TensorView* b = sum(a, {0});
  TensorView* c = add(a, a);
  TensorView* d = permute(a, {{2, 0}});

  fusion->addInput(a);
  fusion->addOutput(b);
  fusion->addOutput(c);
  fusion->addOutput(d);

  a->setDeviceMesh(mesh);
  b->setDeviceMesh(mesh);
  c->setDeviceMesh(mesh);
  d->setDeviceMesh(mesh);
  a->axis(2)->parallelize(ParallelType::DIDx);
  b->axis(2)->parallelize(ParallelType::DIDx);
  c->axis(2)->parallelize(ParallelType::DIDx);
  d->axis(0)->parallelize(ParallelType::DIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto a_ = at::randn({4, 2, 1, 5}, options);
  auto b_ = at::sum(a_, {0});
  auto c_ = a_ + a_;
  auto d_ = at::permute(a_, {2, 0, 1, 3});
  std::vector<at::Tensor> outputs_ = {b_, c_, d_};

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {a_});
  auto outputs = fe.runFusion({a_});
  testValidate(fusion.get(), outputs, {a_}, outputs_, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(InputType, ShardedComputeTest, testing::Bool());

// params: concrete vs symbolic input, sharded axis
class ShardingTest : public MultiDeviceTest,
                     public testing::WithParamInterface<std::tuple<bool, int>> {
};

// Test memory allocation of multidevice fusion with unsharded inputs
// and sharded intermediates, outputs.
TEST_P(ShardingTest, UnshardedGlobalInput) {
  auto [creates_concrete_tensor, sharded_dim] = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  DeviceMesh mesh(devices);
  std::vector<int64_t> input_size = {2, 3, 2, 4};
  int sharded_output_dim = 3;
  input_size[sharded_dim] = num_devices;
  input_size[sharded_output_dim] = num_devices;

  TensorView* tv0 = creates_concrete_tensor ? makeConcreteTensor(input_size)
                                            : makeSymbolicTensor(4);
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = add(tv1, tv1);
  TensorView* tv3 = sum(tv2, {sharded_dim});

  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);

  tv1->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  tv2->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  tv3->axis(sharded_output_dim)->parallelize(ParallelType::DIDx);

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
  auto [creates_concrete_tensor, sharded_dim] = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  DeviceMesh mesh(devices);
  std::vector<int64_t> unsharded_input_size = {3, 2, 5};
  unsharded_input_size[sharded_dim] = num_devices;

  TensorView* tv0 = creates_concrete_tensor
      ? makeConcreteTensor(unsharded_input_size)
      : makeSymbolicTensor(unsharded_input_size.size());
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
      shardTensor(x1, tv0, communicator->deviceId())};
  auto x2 = x1 * 2;
  MultiDeviceExecutor runtime(std::move(fusion), *communicator);
  auto outputs = runtime.runWithInput(inputs);
  testValidate(
      runtime.completeFusion(), outputs, inputs, {x1, x2}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    OutermostShard,
    ShardingTest,
    testing::Combine(testing::Bool(), testing::Values(0)));

INSTANTIATE_TEST_SUITE_P(
    InnermostShard,
    ShardingTest,
    testing::Combine(testing::Bool(), testing::Values(1)));

} // namespace nvfuser
#endif
