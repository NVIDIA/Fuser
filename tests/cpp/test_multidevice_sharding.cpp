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
#include <ops/all_ops.h>
#include <preseg_passes/finalize_multidevice_domains.h>
#include <preseg_passes/optimization_pass.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using testing::Contains;
using testing::Each;
using testing::ElementsAre;
using testing::Not;
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
  auto x1 = shardTensor(x0, tv1);
  auto x2 = x1 + x1;
  auto x3 = shardTensor(at::sum(x0 + x0, {sharded_input_dim}), tv3);
  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({x0});

  testValidate(
      executor_cache.fusion(), outputs, {x0}, {x1, x2, x3}, __LINE__, __FILE__);
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
  KernelArgumentHolder args = {shardTensor(x1, tv0)};
  auto x2 = x1 * 2;
  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs(args);
  testValidate(
      executor_cache.fusion(), outputs, args, {x1, x2}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    MultiDeviceReductionTest,
    testing::Combine(testing::Bool(), testing::Values(0, 1)),
    ([](const testing::TestParamInfo<std::tuple<bool, int>>& info) {
      auto [creates_concrete_tensor, sharded_dim] = info.param;
      std::ostringstream os;
      os << (creates_concrete_tensor ? "concrete" : "symbolic")
         << "_sharded_along_dim_" << sharded_dim;
      return os.str();
    }));

TEST_F(MultiDeviceTest, Reduction) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());

  TensorView* in = makeContigTensor(2);
  TensorView* out = sum(in, {0});

  fusion->addInput(in);
  fusion->addOutput(out);

  in->setDeviceMesh(mesh);
  in->axis(0)->parallelize(ParallelType::DIDx);

  auto unsharded_in_tensor = at::randn({mesh.size(), 4}, tensor_options);
  auto in_tensor = shardTensor(unsharded_in_tensor, in);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {in_tensor},
      {unsharded_in_tensor.sum(0)},
      __LINE__,
      __FILE__);
}

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
  KernelArgumentHolder args = {shardTensor(aten_x, x)};

  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs(args);
  testValidate(
      executor_cache.fusion(),
      outputs,
      args,
      {shardTensor(expected_out[0], x), shardTensor(expected_out[1], x)},
      __LINE__,
      __FILE__);
}

TEST_F(MultiDeviceTest, BackpropMeshes) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();

  TensorView* x = makeContigConcreteTensor({num_devices, -1});
  TensorView* y = uniform(
      shape(x),
      fusion->zeroVal(DataType::Float),
      fusion->oneVal(DataType::Float),
      DataType::Float);
  TensorView* z = add(x, y);
  fusion->addInput(x);
  fusion->addOutput(z);

  auto mesh = DeviceMesh::createForNumDevices(num_devices);
  x->setDeviceMesh(mesh);
  x->axis(0)->parallelize(ParallelType::DIDx);

  at::Tensor unsharded_x_tensor = at::randn({num_devices, 4}, tensor_options);
  at::Tensor x_tensor = shardTensor(unsharded_x_tensor, x);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor z_tensor =
      executor_cache.runFusionWithInputs({x_tensor})[0].as<at::Tensor>();
  EXPECT_THAT(z_tensor.sizes(), ElementsAre(1, 4))
      << "Due to sharding propagation, z is supposed to "
      << "be sharded in the same way as x.";
}

TEST_F(MultiDeviceTest, DivideBySum) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int64_t d = communicator_->size();

  // [b, h, s, s]
  TensorView* x = makeContigTensor(4);
  TensorView* sum_x = sum(x, {-1});
  TensorView* sum_x_broadcasted = broadcast(sum_x, {false, false, false, true});
  TensorView* y = div(x, sum_x_broadcasted);
  fusion->addInput(x);
  fusion->addOutput(y);

  auto mesh = DeviceMesh::createForNumDevices(d);
  for (auto* tv : {x, sum_x, sum_x_broadcasted, y}) {
    tv->setDeviceMesh(mesh);
    tv->outer_split(1, d);
    tv->axis(1)->parallelize(ParallelType::DIDx);
    tv->reorder({{1, 0}});
  }

  const int64_t b = 2;
  const int64_t h = d * 3;
  const int64_t s = 5;
  at::Tensor unsharded_x_tensor = at::randint(5, {b, h, s, s}, tensor_options);
  at::Tensor x_tensor = shardTensor(unsharded_x_tensor, x);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor y_tensor =
      executor_cache.runFusionWithInputs({x_tensor})[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(),
      {y_tensor},
      {x_tensor},
      {x_tensor / x_tensor.sum(-1, true)},
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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({aten_x});

  testValidate(
      executor_cache.fusion(),
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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();

  at::Tensor expected_out_tensor =
      shardTensor(unsharded_in_tensor.sum(0), reduce_scattered) +
      in_tensor.size(1);
  testValidate(
      executor_cache.fusion(),
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

  TensorView* in = makeSymbolicTensor(2);
  TensorView* out = transpose(in, 0, 1);
  // Set allocation domain to test Transpose scheduler.
  out->setAllocationDomain(out->getLoopDomain(), true);

  in->setDeviceMesh(mesh);
  in->outer_split(0, num_devices);
  in->axis(0)->parallelize(ParallelType::DIDx);

  fusion->addInput(in);
  fusion->addOutput(out);

  // Sizes need to be large enough to trigger the transpose scheduler.
  at::Tensor in_tensor = at::randn({1024, 1024}, tensor_options);
  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();

  testValidate(
      executor_cache.fusion(),
      {out_tensor},
      {in_tensor},
      {in_tensor.transpose(0, 1)},
      __LINE__,
      __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(HeuristicIs(SchedulerType::Transpose)));
}

TEST_F(MultiDeviceTest, LoopSplit) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(num_devices);

  TensorView* in = makeContigConcreteTensor({num_devices * 3});
  in->setDeviceMesh(mesh);
  fusion->addInput(in);
  TensorView* out = set(in);
  fusion->addOutput(out);

  for (auto* tv : {in, out}) {
    tv->outer_split(0, num_devices);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  at::Tensor in_tensor = at::randn({3}, tensor_options);
  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();

  testValidate(
      executor_cache.fusion(),
      {out_tensor},
      {in_tensor},
      {in_tensor},
      __LINE__,
      __FILE__);
}

TEST_F(MultiDeviceTest, LoopSplitWithReorder) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(num_devices);

  TensorView* in = makeContigConcreteTensor({2, num_devices * 3});
  in->setDeviceMesh(mesh);
  fusion->addInput(in);

  TensorView* out = set(in);
  fusion->addOutput(out);

  // logical: i{2}, i{3D}
  // allocation: iDIDx{D}, i{3}, i{2}
  in->outer_split(1, num_devices);
  in->reorder({{0, -1}});
  in->axis(0)->parallelize(ParallelType::DIDx);
  in->setAllocationDomain(in->getLoopDomain(), true);

  out->outer_split(1, num_devices);
  out->axis(1)->parallelize(ParallelType::DIDx);
  out->setAllocationDomain(out->getLoopDomain(), true);

  at::Tensor in_tensor = at::randn({3, 2}, tensor_options).t();
  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();

  testValidate(
      executor_cache.fusion(),
      {out_tensor},
      {in_tensor},
      {in_tensor},
      __LINE__,
      __FILE__);
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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({1, 8}, options);
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
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
                       .shape({num_devices * 3, -1})
                       .expanded({true, false})
                       .build();
  in->setDeviceMesh(mesh);
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  if (parallelizes_broadcast) {
    for (auto* tv : {in, out}) {
      tv->outer_split(0, num_devices);
      tv->axis(0)->parallelize(ParallelType::DIDx);
    }
  }

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({8}, tensor_options)
          .as_strided(
              {parallelizes_broadcast ? 3 : num_devices * 3, 8}, {0, 1});
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(, MultiDeviceBroadcastTest, testing::Bool());

TEST_F(MultiDeviceTest, ShardTensor_OuterSplit) {
  const int d = communicator_->size();

  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv = makeContigConcreteTensor({2, d * 3});
  tv->setDeviceMesh(DeviceMesh::createForNumDevices(d));
  tv->outer_split(1, d);
  tv->axis(1)->parallelize(ParallelType::DIDx);

  fusion.addInput(tv);
  fusion.addOutput(tv);

  at::Tensor unsharded = at::arange(2 * d * 3).view({2, d * 3});
  at::Tensor sharded = shardTensor(unsharded, tv);

  EXPECT_THAT(sharded.sizes(), ElementsAre(2, 3));
  at::Tensor expected = unsharded.view({2, d, 3}).index(
      {at::indexing::Slice(),
       communicator_->deviceId(),
       at::indexing::Slice()});
  EXPECT_TRUE(at::equal(sharded, expected));
}

TEST_F(MultiDeviceTest, ShardTensor_InnerSplit) {
  const int d = communicator_->size();

  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv = makeContigConcreteTensor({d * 3});
  tv->setDeviceMesh(DeviceMesh::createForNumDevices(d));
  tv->outer_split(0, d);
  tv->axis(-1)->parallelize(ParallelType::DIDx);

  fusion.addInput(tv);
  fusion.addOutput(tv);

  at::Tensor unsharded = at::arange(d * 3);
  EXPECT_THAT(
      [&]() { shardTensor(unsharded, tv); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("DID on inner splits")));
}

TEST_F(MultiDeviceTest, BiasAddRelu) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();
  const int b = 2;
  const int s = 128;
  const int h = d * 64;

  TensorView* in = makeContigConcreteTensor({b, s, h});
  TensorView* bias = makeContigConcreteTensor({h});
  TensorView* broadcasted_bias = broadcast(bias, {true, true, false});
  TensorView* add_out = add(in, broadcasted_bias);
  TensorView* out = relu(add_out);

  fusion->addInput(in);
  fusion->addInput(bias);
  fusion->addOutput(out);

  auto mesh = DeviceMesh::createForNumDevices(d);
  for (auto* tv : {in, bias, broadcasted_bias, add_out, out}) {
    tv->setDeviceMesh(mesh);
    tv->outer_split(-1, d);
    tv->axis(-2)->parallelize(ParallelType::DIDx);
    tv->reorder({{-2, 0}});
  }

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({b, s, h / d}, tensor_options);
  at::Tensor bias_tensor = at::randn({h / d}, tensor_options);
  KernelArgumentHolder args = {in_tensor, bias_tensor};
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs(args)[0].as<at::Tensor>();
  testValidate(executor_cache.fusion(), {out_tensor}, args, __LINE__, __FILE__);
}

TEST_F(MultiDeviceTest, ViewWithSplit) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();

  TensorView* in = makeContigConcreteTensor({d * 2, 15});
  TensorView* out = reshape(in, {d * 2, 15}, {d * 2, 3, 5});

  fusion->addInput(in);
  fusion->addOutput(out);

  auto mesh = DeviceMesh::createForNumDevices(d);
  for (auto* tv : {in, out}) {
    tv->setDeviceMesh(mesh);
    tv->outer_split(0, d);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }
  // So the View won't be treated as a meta op and will trigger Pointwise, the
  // purpose of the test.
  in->setAllocationDomain(in->getLoopDomain(), false);
  out->setAllocationDomain(out->getLoopDomain(), true);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 15}, tensor_options);
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(),
      {out_tensor},
      {in_tensor},
      {in_tensor.view({-1, 3, 5})},
      __LINE__,
      __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(HeuristicIs(SchedulerType::PointWise)));
}

TEST_F(MultiDeviceTest, ViewWithMerge) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();

  TensorView* in = makeContigConcreteTensor({d * 2, 3, 5});
  TensorView* out = reshape(in, {d * 2, 3, 5}, {d * 2, 15});

  fusion->addInput(in);
  fusion->addOutput(out);

  auto mesh = DeviceMesh::createForNumDevices(d);
  for (auto* tv : {in, out}) {
    tv->setDeviceMesh(mesh);
    tv->outer_split(0, d);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }
  // contiguity=false so the View won't be treated as a meta op and will
  // trigger Pointwise, the purpose of the test.
  in->setAllocationDomain(in->getLoopDomain(), false);
  out->setAllocationDomain(out->getLoopDomain(), true);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3, 5}, tensor_options);
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(),
      {out_tensor},
      {in_tensor},
      {in_tensor.view({-1, 15})},
      __LINE__,
      __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(HeuristicIs(SchedulerType::PointWise)));
}

TEST_F(MultiDeviceTest, ReorderDIDToFront) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto d = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(d);

  const int64_t b = 2, s = 4, h = 16;
  TensorView* in = makeConcreteTensor({b, s, d * h});
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  for (auto* tv : {in, out}) {
    tv->setDeviceMesh(mesh);
    tv->outer_split(-1, d);
    tv->axis(-2)->parallelize(ParallelType::DIDx);
    reorderDIDToFront(tv);
    NVF_CHECK(tv->axis(0)->isDeviceDim());
  }

  at::Tensor in_tensor = at::randn({b, s, h}, tensor_options);
  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();

  testValidate(
      executor_cache.fusion(),
      {out_tensor},
      {in_tensor},
      {in_tensor},
      __LINE__,
      __FILE__);
}

using InsertReshardingTestParams = std::tuple<bool, bool, bool>;

class InsertReshardingTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<InsertReshardingTestParams> {};

TEST_P(InsertReshardingTest, Execute) {
  auto [is_tv0_tv5_sharded, is_tv1_tv4_sharded, is_tv2_tv3_sharded] =
      GetParam();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(3);
  TensorView* tv1 = mul(tv0, tv0);
  TensorView* tv2 = add(tv0, tv1);
  TensorView* tv3 = sum(tv2, {1});
  TensorView* tv4 = broadcast(tv3, {false, true, false});
  TensorView* tv5 = mul(tv2, tv4);

  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  fusion->addOutput(tv5);

  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());
  for (auto* tv : {tv0, tv1, tv2, tv3, tv4, tv5}) {
    tv->setDeviceMesh(mesh);
  }

  if (is_tv0_tv5_sharded) {
    tv0->axis(1)->parallelize(ParallelType::DIDx);
    tv5->axis(1)->parallelize(ParallelType::DIDx);
  }
  if (is_tv1_tv4_sharded) {
    tv1->axis(1)->parallelize(ParallelType::DIDx);
    tv4->axis(1)->parallelize(ParallelType::DIDx);
  }
  if (is_tv2_tv3_sharded) {
    tv2->axis(1)->parallelize(ParallelType::DIDx);
    tv3->axis(1)->parallelize(ParallelType::DIDx);
  }

  at::Tensor t0 = at::randint(3, {2, mesh.size(), 5}, tensor_options);
  at::Tensor t1 = t0 * t0;
  at::Tensor t2 = t0 + t1;
  at::Tensor t5 = t2 * t2.sum({1}, /*keepdim=*/true);

  FusionExecutorCache executor_cache(std::move(fusion));
  if (is_tv0_tv5_sharded) {
    t0 = shardTensor(t0, 1, mesh);
    t5 = shardTensor(t5, 1, mesh);
  }
  if (is_tv1_tv4_sharded) {
    t1 = shardTensor(t1, 1, mesh);
  }
  auto outs = executor_cache.runFusionWithInputs({t0});
  testValidate(
      executor_cache.fusion(), outs, {t0}, {t1, t5}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    InsertReshardingTest,
    ::testing::Combine(
        ::testing::Bool(),
        ::testing::Bool(),
        ::testing::Bool()));

TEST_F(MultiDeviceTest, TransformPropagatorSplitReshape) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();
  const int64_t b = 2, s = 2, h = 4, e = 3;

  TensorView* tv0 = makeContigConcreteTensor(
      {b, s, d * h * e}); // in: loop domain: {b, s, d*h*e}
  TensorView* tv1 = reshape(
      tv0,
      {b, s, d * h * e},
      {b, s, d * h, e}); // out: loop domain: {b, s, d*h, e}

  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  auto mesh = DeviceMesh::createForNumDevices(d);

  // Propagate transform from reshaped output to input.
  // Without this propagation, the two DID axes on `in` and `out` will not be
  // mapped in together in ID model. This causes scheduling to fail due to
  // resharding.
  TransformPropagator propagator_c2p(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator_c2p);
  // in: loop domain: {b, s, d*h, e} after transform propagation

  // Loop split and parallelize input
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);
  tv0->split(-2, d, /*inner_split=*/false);
  tv0->axis(-3)->parallelize(ParallelType::DIDx);
  // in: loop domain: {b, s, DIDx{d}, h, e}

  // Propagate DID loop split to output
  TransformPropagator propagator_p2c(tv0);
  MaxLogicalDomainInfoSpanningTree(tv0).traverse(&propagator_p2c);
  // out: loop domain: {b, s, d, h, e} after transform propagation

  // Parallelize output
  scheduler_utils::parallelizeAllLike(
      tv0,
      /*pos=*/-1,
      /*selected_tv=*/{tv1});
  // out: loop domain: {b, s, DIDx{d}, h, e} after parallelization

  tv0->setAllocationDomain(tv0->getLoopDomain(), true);
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor inp = at::randn({b, s, d * h * e}, tensor_options);
  at::Tensor sharded_inp = shardTensor(inp, tv0);

  at::Tensor nvf_out =
      executor_cache.runFusionWithInputs({sharded_inp})[0].as<at::Tensor>();

  testValidate(
      executor_cache.fusion(),
      {nvf_out},
      {sharded_inp},
      {sharded_inp.view({b, s, h, e})},
      __LINE__,
      __FILE__);
}

TEST_F(MultiDeviceTest, LoopShardedSplitReshapeIds) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();
  const int64_t b = 2, s = 3, h = 8, e = 4;

  TensorView* tv0 = makeContigConcreteTensor({b, s, d * h * e});
  TensorView* tv1 = reshape(tv0, {b, s, d * h * e}, {b, s, d * h, e});

  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  auto mesh = DeviceMesh::createForNumDevices(d);

  tv0->setDeviceMesh(mesh);
  tv0->split(-1, d, /*inner_split=*/false);
  tv0->axis(-2)->parallelize(ParallelType::DIDx);

  tv1->setDeviceMesh(mesh);
  tv1->split(-2, d, /*inner_split=*/false);
  tv1->axis(-3)->parallelize(ParallelType::DIDx);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor inp = at::randn({b, s, d * h * e}, tensor_options);
  at::Tensor sharded_inp = shardTensor(inp, -1, mesh);

  at::Tensor nvf_out =
      executor_cache.runFusionWithInputs({sharded_inp})[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(),
      {nvf_out},
      {sharded_inp},
      {sharded_inp.view({b, s, h, e})},
      __LINE__,
      __FILE__);
}

TEST_F(MultiDeviceTest, LoopShardedMergeReshapeIds) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();
  const int64_t b = 2, s = 3, h = 8, e = 4;

  TensorView* tv0 = makeContigConcreteTensor({b, s, d * h, e});
  TensorView* tv1 = reshape(tv0, {b, s, d * h, e}, {b, s, d * h * e});

  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  auto mesh = DeviceMesh::createForNumDevices(d);
  tv0->setDeviceMesh(mesh);
  tv0->split(-2, d, /*inner_split=*/false);
  tv0->axis(-3)->parallelize(ParallelType::DIDx);

  tv1->setDeviceMesh(mesh);
  tv1->split(-1, d, /*inner_split=*/false);
  tv1->axis(-2)->parallelize(ParallelType::DIDx);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor inp = at::randn({b, s, d * h, e}, tensor_options);
  at::Tensor sharded_inp = shardTensor(inp, -2, mesh);
  at::Tensor nvf_out =
      executor_cache.runFusionWithInputs({sharded_inp})[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(),
      {nvf_out},
      {sharded_inp},
      {sharded_inp.view({b, s, h * e})},
      __LINE__,
      __FILE__);
}

TEST_F(MultiDeviceTest, MultipleTransformReshape) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();
  const int64_t b = 2, s = 3, h = 8, e = 4;

  TensorView* tv0 = makeContigConcreteTensor({d * b, s, h * e});
  TensorView* tv1 = reshape(tv0, {d * b, s, h * e}, {d * b * s * h, e});
  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  auto mesh = DeviceMesh::createForNumDevices(d);
  tv0->setDeviceMesh(mesh);
  tv0->split(0, d, /*inner_split=*/false);
  tv0->axis(0)->parallelize(ParallelType::DIDx);

  at::Tensor inp = at::randn({d * b, s, h * e}, tensor_options);
  at::Tensor sharded_inp = shardTensor(inp, 0, mesh);
  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor nvf_out =
      executor_cache.runFusionWithInputs({sharded_inp})[0].as<at::Tensor>();
  EXPECT_TRUE(at::allclose(nvf_out, sharded_inp.view({b * s * h, e})));
}

TEST_F(MultiDeviceTest, AliasingRetainsSharding) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(d);

  TensorView* tv0 = makeContigConcreteTensor({3 * d});
  TensorView* tv1 = add(tv0, tv0);
  TensorView* tv2 = set(tv1);
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  fusion->addOutput(tv2);

  tv0->setDeviceMesh(mesh);
  tv0->outer_split(0, d);
  tv0->axis(0)->parallelize(ParallelType::DIDx);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor t0 = at::randn({3 * d}, tensor_options);
  at::Tensor sharded_t0 = shardTensor(t0, 0, mesh);
  at::Tensor nvf_out =
      executor_cache.runFusionWithInputs({sharded_t0})[0].as<at::Tensor>();

  EXPECT_TRUE(at::allclose(nvf_out, sharded_t0 * 2));

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      Each(Not(HeuristicIs(SchedulerType::Communication))));
}

TEST_F(MultiDeviceTest, DecomposeRowParallelLinearWithBias) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(d);
  const int64_t b = 2, s = 3, e = 5;

  TensorView* tv0 = makeContigConcreteTensor({b, s, d * e}, DataType::BFloat16);
  TensorView* tv1 = makeContigConcreteTensor({e, d * e}, DataType::BFloat16);
  TensorView* tv2 = makeContigConcreteTensor({e}, DataType::BFloat16);
  TensorView* tv3 = linear(tv0, tv1, tv2);
  TensorView* tv4 = castOp(DataType::Float, tv3);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addOutput(tv4);

  for (auto* tv : {tv0, tv1}) {
    tv->setDeviceMesh(mesh);
    tv->outer_split(-1, d);
    tv->axis(-2)->parallelize(ParallelType::DIDx);
  }
  tv2->setDeviceMesh(mesh);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor inp = at::ones({b, s, d * e}, tensor_options.dtype(at::kBFloat16));
  at::Tensor weight = at::ones({e, d * e}, tensor_options.dtype(at::kBFloat16));
  at::Tensor bias = at::ones({e}, tensor_options.dtype(at::kBFloat16));
  at::Tensor sharded_inp = shardTensor(inp, -1, mesh);
  at::Tensor sharded_weight = shardTensor(weight, -1, mesh);
  at::Tensor nvf_out =
      executor_cache.runFusionWithInputs({sharded_inp, sharded_weight, bias})[0]
          .as<at::Tensor>();
  EXPECT_TRUE(at::allclose(
      nvf_out, (at::matmul(inp, weight.t()) + bias).to(at::kFloat)));
}

TEST_F(MultiDeviceTest, OuterReductionShardedInnerDimension) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t d = communicator_->size();
  int64_t b = 1, s = 2048, h = 96, e = 12288;
  auto mesh = DeviceMesh::createForNumDevices(d);

  if (h % d != 0) {
    GTEST_SKIP() << "Requires number of devices=" << d
                 << " evenly divide H=" << h;
  }

  TensorView* tv0 =
      makeContigConcreteTensor({b, s, h, e / h}, DataType::BFloat16);
  TensorView* tv1 =
      makeContigConcreteTensor({b, s, h, e / h}, DataType::BFloat16);
  TensorView* tv2 =
      makeContigConcreteTensor({b, s, h, e / h}, DataType::BFloat16);
  TensorView* tv3 = cat({tv0, tv1, tv2}, -1);
  TensorView* tv4 = reshape(tv3, {b, s, h, 3 * e / h}, {b, s, 3 * e});
  TensorView* tv5 = castOp(DataType::Float, tv4);
  TensorView* tv6 = sum(tv5, {0, 1});
  TensorView* tv7 = castOp(DataType::BFloat16, tv6);

  for (TensorView* tv : {tv0, tv1, tv2}) {
    tv->setDeviceMesh(mesh);
    tv->outer_split(2, d);
    tv->axis(2)->parallelize(ParallelType::DIDx);
    fusion->addInput(tv);
  }

  fusion->addOutput(tv7);
  std::vector<at::Tensor> inputs = {
      at::randn({b, s, h, e / h}, tensor_options.dtype(at::kBFloat16)),
      at::randn({b, s, h, e / h}, tensor_options.dtype(at::kBFloat16)),
      at::randn({b, s, h, e / h}, tensor_options.dtype(at::kBFloat16))};

  std::vector<at::Tensor> sharded_inputs = {
      shardTensor(inputs[0], 2, mesh),
      shardTensor(inputs[1], 2, mesh),
      shardTensor(inputs[2], 2, mesh)};

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor nvf_out =
      executor_cache.runFusionWithInputs(sharded_inputs)[0].as<at::Tensor>();

  at::Tensor ref_out =
      at::cat({sharded_inputs[0], sharded_inputs[1], sharded_inputs[2]}, -1)
          .view({b, s, 3 * e / d})
          .sum(c10::IntArrayRef({0, 1}));
  EXPECT_TRUE(at::allclose(nvf_out, ref_out, 1e-3, 1e-3));

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();

  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(HeuristicIs(SchedulerType::Reduction)));
  const ReductionParams* rparams = runtime->schedulerHeuristics()
                                       ->heuristicsList()
                                       .at(0)
                                       ->as<ReductionParams>();
  EXPECT_TRUE(rparams->vectorize_iter_dom);
  EXPECT_EQ(rparams->unroll_factor_iter_dom, 8);

  auto scheduled_fusion = runtime->executors()
                              .at(0)
                              ->as<KernelExecutor>()
                              ->compiledKernel()
                              ->kernel();

  auto is_vectorized = [](const TensorView* tv) {
    return std::any_of(
        tv->getLoopDomain().begin(),
        tv->getLoopDomain().end(),
        [](const IterDomain* id) {
          return id->getParallelType() == ParallelType::Vectorize;
        });
  };

  for (auto* val : scheduled_fusion->outputs()) {
    EXPECT_TRUE(is_vectorized(val->as<TensorView>()));
  }
}

TEST_F(MultiDeviceTest, AllocationPermutationOfLoop) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(d);

  TensorView* tv0 = makeContigConcreteTensor({5, 3 * d});
  TensorView* tv1 = set(tv0);

  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  for (auto* tv : {tv0, tv1}) {
    tv->setDeviceMesh(mesh);
    tv->outer_split(1, d);
    tv->axis(1)->parallelize(ParallelType::DIDx);
    // DIDx is outermost in loop but not in allocation domain
    tv->setAllocationDomain(tv->getLoopDomain(), true);
    reorderDIDToFront(tv);
  }

  // Disable the pass to verify we can run a fusion where allocation domain
  // is a permutation of loop domain. This pass can currently not be modified
  // due to other issues listed in #4381.
  preseg_passes::OptimizationPassGuard<
      preseg_passes::FinalizeMultideviceDomainsPass>
      optimization_guard(false);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor inp = at::randn({5, 3 * d}, tensor_options);
  at::Tensor sharded_inp = shardTensor(inp, -1, mesh);
  at::Tensor nvf_out =
      executor_cache.runFusionWithInputs({sharded_inp})[0].as<at::Tensor>();
  EXPECT_TRUE(at::allclose(nvf_out, sharded_inp));
}

} // namespace nvfuser
