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

// Test memory allocation of multidevice fusion with unsharded inputs
// and sharded intermediates, outputs.

TEST(NVFuserTest, TestContiguousShardSet) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DeviceMesh mesh({0, 1, 2, 3, 4});
  TensorView* a = makeContigConcreteTensor({4, 4, 4});
  TensorView* b = set(a);
  TensorView* c = set(a);

  a->setDeviceMesh(mesh);
  b->setDeviceMesh(mesh);
  c->setDeviceMesh(mesh);
  b->axis(0)->parallelize(ParallelType::DIDx);
  c->axis(1)->parallelize(ParallelType::DIDx);

  ASSERT_TRUE(isContiguousShard(b, a));
  ASSERT_FALSE(isContiguousShard(c, a));
}

TEST(NVFuserTest, TestContiguousShardReduce) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DeviceMesh mesh({0, 1, 2, 3, 4});
  TensorView* a = makeContigConcreteTensor({4, 4, 4});
  TensorView* b = sum(a, {0});
  TensorView* c = sum(a, {0});

  a->setDeviceMesh(mesh);
  b->setDeviceMesh(mesh);
  c->setDeviceMesh(mesh);
  a->axis(0)->parallelize(ParallelType::DIDx);
  c->axis(1)->parallelize(ParallelType::DIDx);

  ASSERT_TRUE(isContiguousShard(b, a));
  ASSERT_TRUE(isContiguousShard(c, a));
}

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
  std::vector<at::Tensor> outputs_ = {b_, c_, d_};

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {a_});
  auto outputs = fe.runFusion({a_});
  testValidate(fusion.get(), outputs, {a_}, outputs_, __LINE__, __FILE__);
}

// params: concrete vs symbolic input, sharded axis
class ShardingTest : public MultiDeviceTest,
                     public ::testing::WithParamInterface<std::tuple<bool, int>> {};

// Test memory allocation of multidevice fusion with unsharded inputs
// and sharded intermediates, outputs.
TEST_P(ShardingTest, UnshardedGlobalInput) {
  auto [concreteTv, sharded_axis] = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  DeviceMesh mesh(devices);
  std::vector<int64_t> input_size = {2, 3, 4, 3};
  input_size[sharded_axis] = num_devices;
  input_size[sharded_axis+1] = num_devices;

  TensorView* tv0 =
      concreteTv ? makeConcreteTensor(input_size) : makeContigTensor(4);
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = add(tv1, tv1);
  TensorView* tv3 = sum(tv2, {sharded_axis}); 
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);

  tv1->axis(sharded_axis)->parallelize(ParallelType::DIDx);
  tv2->axis(sharded_axis)->parallelize(ParallelType::DIDx);
  tv3->axis(2)->parallelize(ParallelType::DIDx);

  std::vector<TensorView*> tvs = {tv0, tv1, tv2, tv3};
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
  }

  auto x0 = at::randn(input_size, tensor_options);
  std::vector<c10::IValue> inputs = {x0};
  auto x1 = shardTensor(x0, tv1, communicator->deviceId());
  auto x2 = x1 + x1;
  auto x3 = shardTensor(at::sum(x0+x0, {sharded_axis}), tv3, communicator->deviceId());
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
  std::vector<int64_t> unsharded_input_size = {3,4,2,5};
  unsharded_input_size[sharded_dim] = num_devices;

  TensorView* tv0 = concreteTv ? makeConcreteTensor(unsharded_input_size)
                               : makeContigTensor(4);
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
  std::cout << "Input shape " << shardTensor(x1, tv0, communicator->deviceId()).sizes() << std::endl;
  auto x2 = x1 * 2;
  MultiDeviceExecutor runtime(std::move(fusion), *communicator);
  auto outputs = runtime.runWithInput(inputs);
  testValidate(
      runtime.completeFusion(), outputs, inputs, {x1, x2}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(ConcreteInput_InnerShard, ShardingTest, 
  ::testing::Values(std::make_tuple(true, 1)));

INSTANTIATE_TEST_SUITE_P(SymbolicInput_InnerShard, ShardingTest, 
  ::testing::Values(std::make_tuple(false, 1)));

  INSTANTIATE_TEST_SUITE_P(ConcreteInput_OutermostShard, ShardingTest, 
  ::testing::Values(std::make_tuple(true, 0)));

INSTANTIATE_TEST_SUITE_P(SymbolicInput_OutermostShard, ShardingTest, 
  ::testing::Values(std::make_tuple(false, 0)));

} // namespace nvfuser
#endif
