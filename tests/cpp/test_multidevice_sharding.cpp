// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using testing::UnorderedElementsAre;

// params: concrete vs symbolic input, sharded axis
class MultiDeviceReductionTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<std::tuple<bool, int>> {};

// Test multidevice fusion with unsharded inputs and sharded intermediates,
// outputs.
TEST_P(MultiDeviceReductionTest, UnshardedInput_ShardedOutput) {
  auto [creates_concrete_tensor, sharded_input_dim] = GetParam();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const int num_devices = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(num_devices);
  std::vector<int64_t> input_shape = {2, 3, 2, num_devices};
  input_shape[sharded_input_dim] = num_devices;

  TensorView* tv0 = creates_concrete_tensor
      ? makeContigConcreteTensor(input_shape)
      : makeContigTensor(input_shape.size());
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = add(tv1, tv1);
  TensorView* tv3 = sum(tv2, {sharded_input_dim});

  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);

  tv1->axis(sharded_input_dim)->parallelize(ParallelType::DIDx);
  tv2->axis(sharded_input_dim)->parallelize(ParallelType::DIDx);
  tv3->axis(-1)->parallelize(ParallelType::DIDx);

  std::vector<TensorView*> tvs = {tv0, tv1, tv2, tv3};
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
  }

  auto x0 = at::randn(input_shape, tensor_options);
  std::vector<c10::IValue> inputs = {x0};
  auto x1 = shardTensor(x0, tv1);
  auto x2 = x1 + x1;
  auto x3 = shardTensor(at::sum(x0 + x0, {sharded_input_dim}), tv3);
  FusionExecutorCache fec(std::move(fusion));
  auto outputs = fec.runFusionWithInputs(inputs);

  testValidate(fec.fusion(), outputs, inputs, {x1, x2, x3}, __LINE__, __FILE__);
}

// Test multidevice fusion with sharded input and replicated intermediates and
// output.
TEST_P(MultiDeviceReductionTest, ShardedInput_ReplicatedOutput) {
  auto [creates_concrete_tensor, sharded_dim] = GetParam();
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(num_devices);
  std::vector<int64_t> unsharded_input_shape = {3, 2, 5};
  unsharded_input_shape[sharded_dim] = num_devices;

  TensorView* tv0 = creates_concrete_tensor
      ? makeContigConcreteTensor(unsharded_input_shape)
      : makeContigTensor(unsharded_input_shape.size());
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

  auto x1 = at::randn(unsharded_input_shape, tensor_options);
  std::vector<c10::IValue> inputs = {shardTensor(x1, tv0)};
  auto x2 = x1 * 2;
  FusionExecutorCache fec(std::move(fusion));
  auto outputs = fec.runFusionWithInputs(inputs);
  testValidate(fec.fusion(), outputs, inputs, {x1, x2}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    MultiDeviceReductionTest,
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

TEST_F(MultiDeviceTest, Slice) {
  auto fusion = std::make_unique<Fusion>();
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

  FusionExecutorCache fec(std::move(fusion));
  auto outputs = fec.runFusionWithInputs(inputs);
  testValidate(
      fec.fusion(),
      outputs,
      inputs,
      {shardTensor(expected_out[0], x), shardTensor(expected_out[1], x)},
      __LINE__,
      __FILE__);
}

TEST_F(MultiDeviceTest, LayerNorm) {
  auto fusion = std::make_unique<Fusion>();
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

  FusionExecutorCache fec(std::move(fusion));
  auto outputs = fec.runFusionWithInputs({aten_x});

  testValidate(
      fec.fusion(),
      outputs,
      {aten_x},
      {std::get<0>(aten_outputs),
       std::get<1>(aten_outputs),
       std::get<2>(aten_outputs)},
      __LINE__,
      __FILE__);
}

TEST_F(MultiDeviceTest, Issue2758) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(num_devices);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh(mesh);
  in->axis(0)->parallelize(ParallelType::DIDx);

  // ReduceScatter
  TensorView* reduce_scattered = sum(in, {0});
  reduce_scattered->axis(1)->parallelize(ParallelType::DIDx);

  // Add the size of dimension 1 of `in`, which is num_devices.
  TensorView* out = add(reduce_scattered, shape(in)[1]);

  fusion->addInput(in);
  fusion->addOutput(out);

  at::Tensor unsharded_in_tensor =
      at::zeros({num_devices, num_devices, 4}, tensor_options);
  at::Tensor in_tensor = shardTensor(unsharded_in_tensor, in);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];

  at::Tensor expected_out_tensor =
      shardTensor(unsharded_in_tensor.sum(0), reduce_scattered) +
      in_tensor.size(1);
  testValidate(
      fec.fusion(),
      {out_tensor},
      {in_tensor},
      {expected_out_tensor},
      __LINE__,
      __FILE__);
}

TEST_F(MultiDeviceTest, Transpose) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(num_devices);

  TensorView* in = makeContigConcreteTensor({num_devices, -1, -1});
  in->setDeviceMesh(mesh);
  in->axis(0)->parallelize(ParallelType::DIDx);
  TensorView* out = transpose(in, 1, 2);
  out->setAllocationDomain({out->axis(0), out->axis(1), out->axis(2)}, true);

  fusion->addInput(in);
  fusion->addOutput(out);

  // Sizes need to be large enough to trigger the transpose scheduler.
  at::Tensor unsharded_in_tensor =
      at::randn({num_devices, 1024, 1024}, tensor_options);
  at::Tensor in_tensor = shardTensor(unsharded_in_tensor, in);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];

  at::Tensor expected_out_tensor =
      shardTensor(unsharded_in_tensor.transpose(1, 2), out);
  testValidate(
      fec.fusion(),
      {out_tensor},
      {in_tensor},
      {expected_out_tensor},
      __LINE__,
      __FILE__);

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(HeuristicIs(SchedulerType::Transpose)));
}

class MultiDeviceBroadcastTest : public MultiDeviceTest,
                                 public testing::WithParamInterface<bool> {};

// This test and the following `ExpandedBroadcast` test verify the expression
// evaluator correctly binds the extent of a broadcast dimension to 1 and the
// expanded extent to the tensor size. There used to be a bug where it
// incorrectly binds the extent(s) to the mesh size.
//
// `b(DID{i0})` and `b(i0)` bear the same semantics. The former is used more
// often due to how parallelizeAllLike is implemented.
TEST_P(MultiDeviceBroadcastTest, NotExpanded) {
  const bool parallelizes_broadcast = GetParam();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(num_devices);

  TensorView* in = TensorViewBuilder()
                       .dtype(DataType::Float)
                       .contiguity({std::nullopt, true})
                       .shape({1, -1})
                       .build();
  in->setDeviceMesh(mesh);
  if (parallelizes_broadcast) {
    in->axis(0)->parallelize(ParallelType::DIDx);
  }
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  FusionExecutorCache fec(std::move(fusion));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({1, 8}, options);
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  testValidate(fec.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
}

TEST_P(MultiDeviceBroadcastTest, Expanded) {
  const bool parallelizes_broadcast = GetParam();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(num_devices);

  TensorView* in = TensorViewBuilder()
                       .dtype(DataType::Float)
                       .contiguity({std::nullopt, true})
                       .shape({3, -1})
                       .expanded({true, false})
                       .build();
  in->setDeviceMesh(mesh);
  if (parallelizes_broadcast) {
    in->axis(0)->parallelize(ParallelType::DIDx);
  }
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  FusionExecutorCache fec(std::move(fusion));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({8}, options).as_strided({3, 8}, {0, 1});
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  testValidate(fec.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(, MultiDeviceBroadcastTest, testing::Bool());

} // namespace nvfuser
