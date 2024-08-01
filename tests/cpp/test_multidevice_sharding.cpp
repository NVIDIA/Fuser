// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <fusion.h>
#include <gtest/gtest.h>
#include <multidevice/executor.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

// params: concrete vs symbolic input, sharded axis
class MultideviceShardingTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<std::tuple<bool, int>> {};

// Test memory allocation of multidevice fusion with unsharded inputs
// and sharded intermediates, outputs.
TEST_P(MultideviceShardingTest, UnshardedGlobalInput) {
  auto [creates_concrete_tensor, sharded_dim] = GetParam();
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(num_devices);
  std::vector<int64_t> input_size = {2, 3, 2, 4};
  int sharded_output_dim = 3;
  input_size[sharded_dim] = num_devices;
  input_size[sharded_output_dim] = num_devices;

  TensorView* tv0 = creates_concrete_tensor
      ? makeContigConcreteTensor(input_size)
      : makeContigTensor(4);
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
  auto x1 = shardTensor(x0, tv1);
  auto x2 = x1 + x1;
  auto x3 = shardTensor(at::sum(x0 + x0, {sharded_dim}), tv3);
  MultiDeviceExecutor runtime(std::move(fusion), *communicator_);
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
TEST_P(MultideviceShardingTest, ShardGlobalInput) {
  auto [creates_concrete_tensor, sharded_dim] = GetParam();
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(num_devices);
  std::vector<int64_t> unsharded_input_size = {3, 2, 5};
  unsharded_input_size[sharded_dim] = num_devices;

  TensorView* tv0 = creates_concrete_tensor
      ? makeContigConcreteTensor(unsharded_input_size)
      : makeContigTensor(unsharded_input_size.size());
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
  std::vector<c10::IValue> inputs = {shardTensor(x1, tv0)};
  auto x2 = x1 * 2;
  MultiDeviceExecutor runtime(std::move(fusion), *communicator_);
  auto outputs = runtime.runWithInput(inputs);
  testValidate(
      runtime.completeFusion(), outputs, inputs, {x1, x2}, __LINE__, __FILE__);
}

TEST_F(MultideviceShardingTest, Slice) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());

  std::vector<int64_t> input_shape = {communicator_->size(), 8, 8};
  TensorView* x = makeContigConcreteTensor(input_shape);
  TensorView* x_slice0 = slice(x, {0, 0, 0}, {communicator_->size(), 8, 4});
  TensorView* x_slice1 = slice(x, {0, 0, 4}, {communicator_->size(), 8, 8});

  fusion->addInput(x);
  fusion->addOutput(x_slice0);
  fusion->addOutput(x_slice1);

  for (auto tv : {x, x_slice0, x_slice1}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  const auto options = at::TensorOptions().device(communicator_->device());
  auto aten_x = at::randn(input_shape, options);
  auto expected_out = aten_x.split(4, 2);
  std::vector<c10::IValue> inputs = {{shardTensor(aten_x, x)}};

  MultiDeviceExecutor runtime(std::move(fusion), *communicator_);
  auto out = runtime.runWithInput(inputs);
  testValidate(
      runtime.completeFusion(),
      out,
      inputs,
      {shardTensor(expected_out[0], x), shardTensor(expected_out[1], x)},
      __LINE__,
      __FILE__);
}

TEST_F(MultideviceShardingTest, LayerNorm) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());

  std::vector<int64_t> input_shape = {1024, 32, 256};
  std::vector<int64_t> norm_shape{256};
  TensorView* x = makeContigConcreteTensor(input_shape);
  fusion->addInput(x);

  constexpr float kEps = 1e-5;
  Val* eps_ptr = IrBuilder::create<Val>(kEps);
  auto result = layer_norm(x, norm_shape, nullptr, nullptr, eps_ptr);
  fusion->addOutput(result.output);
  fusion->addOutput(result.mean);
  fusion->addOutput(result.invstd);

  x->setDeviceMesh(mesh);

  auto options = at::TensorOptions().device(communicator_->device());
  auto aten_x = at::randn(input_shape, options);
  c10::optional<at::Tensor> aten_weight = c10::nullopt;
  c10::optional<at::Tensor> aten_bias = c10::nullopt;
  auto aten_outputs =
      at::native_layer_norm(aten_x, norm_shape, aten_weight, aten_bias, kEps);

  hir::HostIrExecutorParams executor_params{
      .use_fusion_executor_cache = true,
      .skip_auto_scheduling = false,
      .cache_fusion_executor = false};
  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params);
  auto out = runtime.runWithInput({aten_x});

  testValidate(
      runtime.completeFusion(),
      out,
      {aten_x},
      {std::get<0>(aten_outputs),
       std::get<1>(aten_outputs),
       std::get<2>(aten_outputs)},
      __LINE__,
      __FILE__);
}

TEST_F(MultideviceShardingTest, ReduceScatter_Allgather) {
  // Allreduce = ReduceScatter + Allgather
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(num_devices);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh(mesh);
  TensorView* reduce_scattered = sum(in, {0});
  TensorView* allgathered = set(reduce_scattered);
  fusion->addInput(in);
  fusion->addOutput(allgathered);

  in->axis(0)->parallelize(ParallelType::DIDx);
  reduce_scattered->axis(1)->parallelize(ParallelType::DIDx);

  at::Tensor unsharded_in_tensor =
      at::randn({num_devices, num_devices, 4}, tensor_options);
  at::Tensor in_tensor = shardTensor(unsharded_in_tensor, in);

  hir::HostIrExecutorParams executor_params{
      .use_fusion_executor_cache = true,
      .skip_auto_scheduling = false,
      .cache_fusion_executor = false};
  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params);
  auto outputs = runtime.runWithInput({in_tensor});
  testValidate(
      runtime.completeFusion(),
      outputs,
      {in_tensor},
      {unsharded_in_tensor.sum(0)},
      __LINE__,
      __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    MultideviceShardingTest,
    testing::Combine(testing::Bool(), testing::Values(0, 1)),
    [](const testing::TestParamInfo<std::tuple<bool, int>>& info)
        -> std::string {
      // Not sure why the following doesn't work:
      //   auto [creates_concrete_tensor, sharded_dim] = info.param;
      bool creates_concrete_tensor;
      int sharded_dim;
      std::tie(creates_concrete_tensor, sharded_dim) = info.param;
      std::ostringstream os;
      os << (creates_concrete_tensor ? "concrete" : "symbolic")
         << "_sharded_along_dim_" << sharded_dim;
      return os.str();
    });

} // namespace nvfuser
