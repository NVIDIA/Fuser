// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
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
TEST_F(MultiDeviceTest, ShardOuterAxisConcrete) {
  int sharded_dim = 0;
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> ranks(num_devices);
  std::iota(ranks.begin(), ranks.end(), 0);
  DeviceMesh mesh(ranks);

  TensorView* tv0 = makeConcreteTensor({num_devices, 3});
  TensorView* tv1 = add(tv0, tv0);
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = add(tv2, tv2);
  TensorView* tv4 = set(tv3);
  TensorView* tv5 = sum(tv4, {0});
  fusion->addInput(tv0);
  fusion->addOutput(tv5);

  // TODO: split
  // tv3->split(sharded_dim, num_devices, false);
  tv2->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  // tv3->split(sharded_dim, num_devices, false);
  tv3->axis(sharded_dim)->parallelize(ParallelType::DIDx);

  std::vector<TensorView*> tvs = {tv0, tv1, tv2, tv3, tv4, tv5};
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
  }

  auto x = at::randn({num_devices, 3}, tensor_options);
  std::vector<c10::IValue> inputs = {x};
  auto ref_outputs = at::sum(x * 4, {0});

  MultiDeviceExecutor runtime(std::move(fusion), *communicator);
  auto outputs = runtime.runWithInput(inputs);
  testValidate(
      runtime.fusion(), outputs, inputs, {ref_outputs}, __LINE__, __FILE__);
}
TEST_F(MultiDeviceTest, ShardOuterAxis) {
  int sharded_dim = 0;
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> ranks(num_devices);
  std::iota(ranks.begin(), ranks.end(), 0);
  DeviceMesh mesh(ranks);

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = add(tv0, tv0);
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = add(tv2, tv2);
  TensorView* tv4 = set(tv3);
  TensorView* tv5 = sum(tv4, {0});
  fusion->addInput(tv0);
  fusion->addOutput(tv5);

  // TODO: split
  // tv3->split(sharded_dim, num_devices, false);
  tv2->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  // tv3->split(sharded_dim, num_devices, false);
  tv3->axis(sharded_dim)->parallelize(ParallelType::DIDx);

  std::vector<TensorView*> tvs = {tv0, tv1, tv2, tv3, tv4, tv5};
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
  }

  auto x = at::randn({num_devices, 3}, tensor_options);
  std::vector<c10::IValue> inputs = {x};
  auto ref_outputs = at::sum(x * 4, {0});

  MultiDeviceExecutor runtime(std::move(fusion), *communicator);
  auto outputs = runtime.runWithInput(inputs);
  testValidate(
      runtime.fusion(), outputs, inputs, {ref_outputs}, __LINE__, __FILE__);
}

inline at::Tensor shardInputTensor(at::Tensor tensor, int deviceId) {
  return tensor.index({deviceId, "..."}).unsqueeze(0);
}

TEST_F(MultiDeviceTest, ShardGlobalInput) {
  int sharded_dim = 0;
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> ranks(num_devices);
  std::iota(ranks.begin(), ranks.end(), 0);
  DeviceMesh mesh(ranks);

  TensorView* tv0 = makeContigTensor(3);
  TensorView* tv1 = sum(tv0, {0});
  TensorView* tv2 = add(tv1, tv1);
  fusion->addInput(tv0);
  fusion->addOutput(tv2);

  tv0->axis(sharded_dim)->parallelize(ParallelType::DIDx);

  std::vector<TensorView*> tvs = {tv0, tv1, tv2};
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
  }

  auto x = at::randn({num_devices, 3, 2}, tensor_options);
  // Sharded input shape [1, 3, 2]
  std::vector<c10::IValue> inputs = {
      shardInputTensor(x, communicator->deviceId())};
  auto ref_outputs = at::sum(x, {0}) * 2;

  MultiDeviceExecutor runtime(std::move(fusion), *communicator);
  auto outputs = runtime.runWithInput(inputs);
  testValidate(
      runtime.fusion(), outputs, inputs, {ref_outputs}, __LINE__, __FILE__);
}

} // namespace nvfuser
#endif
