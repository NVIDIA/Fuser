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

TEST_F(MultiDeviceTest, DID_Split) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(num_devices);
  std::vector<int64_t> input_size = {4 * num_devices, 3};

  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = add(tv1, tv1);
  // TensorView* tv3 = sum(tv2, {sharded_dim});

  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  fusion->addOutput(tv2);
  // fusion->addOutput(tv3);

  std::vector<TensorView*> sharded_tvs = {tv1, tv2};
  for (auto tv : sharded_tvs) {
    tv->split(0, num_devices, false);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }
  // tv3->axis(sharded_dim + 1)->parallelize(ParallelType::DIDx);

  std::vector<TensorView*> tvs = {tv0, tv1, tv2};
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }

  // fusion->printKernel();

  auto x0 = at::randn(input_size, tensor_options);
  std::vector<c10::IValue> inputs = {x0};
  std::cout << "Input tensor " << x0.sizes() << " getShardedAxis" << getShardedAxis(tv1) << std::endl;
  auto x1 = shardTensor(x0, 0, mesh).squeeze(0);
  std::cout << "Sharded tensor " << x1.sizes() << std::endl;
  auto x2 = x1 + x1;
  std::cout << "Input x0 " << x0 << std::endl;
  std::cout << "Expected x1 " << x1 << std::endl;
  std::cout << "Expected x2 " << x2 << std::endl;

  MultiDeviceExecutor runtime(std::move(fusion), *communicator_);
  auto outputs = runtime.runWithInput(inputs);
  std::cout << "Calculated x1 " << outputs[0] << std::endl;
  std::cout << "Calculated x2 " << outputs[1] << std::endl;
  testValidate(
      runtime.completeFusion(),
      outputs,
      inputs,
      {x1, x2},
      __LINE__,
      __FILE__);
}

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
      __FILE__,
      "");
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
