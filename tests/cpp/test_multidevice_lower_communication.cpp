// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <ops/all_ops.h>
#include <preseg_passes/mark_aliases_prepare.h>
#include <preseg_passes/optimization_pass.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using testing::Contains;
using testing::Each;
using testing::Pointer;
using testing::UnorderedElementsAre;

namespace {
void assertIsCompiledToHostIrContainer(
    const FusionExecutorCache& executor_cache) {
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  if (isOptionEnabled(EnableOption::HostIrLowering)) {
    auto hicExprs = runtime->getHostIrEvaluator().container().topLevelExprs();
    EXPECT_THAT(hicExprs, Contains(IsA<Communication>()))
        << "host ir container should have at least one communication";
  } else {
    EXPECT_EQ(runtime->executors().size(), 1);
    EXPECT_THAT(runtime->executors(), Each(Pointer(IsA<HostIrExecutor>())))
        << "failed to compile to a HostIrContainer with Communications";
  }
}
} // namespace

using InOutMesh = std::pair<DeviceMesh, DeviceMesh>;

static constexpr int kTensorSize = 4;

class LowerGatherTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<std::tuple<InOutMesh, bool>> {};

TEST_P(LowerGatherTest, ) {
  const auto& [meshes, enable_host_ir_lowering] = GetParam();
  const auto& [in_mesh, out_mesh] = meshes;

  if (enable_host_ir_lowering) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigTensor(2);
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  in->setDeviceMesh(in_mesh);
  out->setDeviceMesh(out_mesh);
  in->axis(0)->parallelize(ParallelType::DIDx);

  SKIP_IF_NOT_ENOUGH_DEVICES(fusion);

  const auto device_id = communicator_->deviceId();
  at::Tensor unsharded_tensor =
      at::randn({in_mesh.size(), kTensorSize}, tensor_options);
  at::Tensor in_tensor = shardTensor(unsharded_tensor, in);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  assertIsCompiledToHostIrContainer(executor_cache);

  if (out_mesh.has(device_id)) {
    EXPECT_TRUE(at::equal(out_tensor, unsharded_tensor));
  }
}

namespace {
std::string paramToString(
    const testing::TestParamInfo<std::tuple<InOutMesh, bool>>& info) {
  auto&& [meshes, enable_hir] = info.param;
  auto&& [in_mesh, out_mesh] = meshes;

  std::stringstream ss;
  ss << "InMesh";
  for (auto id : in_mesh.vector()) {
    ss << "_" << id;
  }
  ss << "_OutMesh";
  for (auto id : out_mesh.vector()) {
    ss << "_" << id;
  }
  ss << (enable_hir ? "_HostIr" : "_NonHostIr");

  return ss.str();
}
} // namespace

INSTANTIATE_TEST_SUITE_P(
    ,
    LowerGatherTest,
    // Create product of InOutMesh configurations and HostIrLowering options
    testing::Combine(
        testing::ValuesIn(std::vector<InOutMesh>(
            {{{0, 1}, {0}}, {{0, 1}, {1}}, {{1, 2}, {0, 2}}})),
        testing::Bool()),
    paramToString);

class LowerScatterTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<std::tuple<InOutMesh, bool>> {};

TEST_P(LowerScatterTest, ) {
  const auto& [meshes, enable_host_ir_lowering] = GetParam();
  const auto& [in_mesh, out_mesh] = meshes;

  if (enable_host_ir_lowering) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigTensor(2);
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  in->setDeviceMesh(in_mesh);
  out->setDeviceMesh(out_mesh);
  out->axis(0)->parallelize(ParallelType::DIDx);

  SKIP_IF_NOT_ENOUGH_DEVICES(fusion);

  const auto device_id = communicator_->deviceId();
  at::Tensor unsharded_tensor =
      at::randn({out_mesh.size(), kTensorSize}, tensor_options);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({unsharded_tensor})[0]
          .as<at::Tensor>();
  assertIsCompiledToHostIrContainer(executor_cache);

  if (out_mesh.has(device_id)) {
    EXPECT_TRUE(at::equal(out_tensor, shardTensor(unsharded_tensor, out)));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    LowerScatterTest,
    testing::Combine(
        testing::ValuesIn(std::vector<InOutMesh>(
            {{{0}, {0, 1}}, {{1}, {0, 1}}, {{0, 2}, {1, 2}}})),
        testing::Bool()),
    paramToString);

class LowerSendRecvTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<std::tuple<InOutMesh, bool>> {};

TEST_P(LowerSendRecvTest, ) {
  const auto& [meshes, enable_host_ir_lowering] = GetParam();
  const auto& [in_mesh, out_mesh] = meshes;

  if (enable_host_ir_lowering) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigTensor(2);
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  ASSERT_EQ(in_mesh.size(), out_mesh.size());
  in->setDeviceMesh(in_mesh);
  out->setDeviceMesh(out_mesh);
  in->axis(0)->parallelize(ParallelType::DIDx);
  out->axis(0)->parallelize(ParallelType::DIDx);

  SKIP_IF_NOT_ENOUGH_DEVICES(fusion);

  const auto device_id = communicator_->deviceId();
  at::Tensor unsharded_tensor =
      at::randn({in_mesh.size(), kTensorSize}, tensor_options);
  at::Tensor in_tensor = shardTensor(unsharded_tensor, in);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  assertIsCompiledToHostIrContainer(executor_cache);

  if (out_mesh.has(device_id)) {
    EXPECT_TRUE(at::equal(out_tensor, shardTensor(unsharded_tensor, out)));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    LowerSendRecvTest,
    testing::Combine(
        testing::ValuesIn(std::vector<InOutMesh>(
            {{{0}, {1}}, {{1}, {0}}, {{1, 2}, {0, 1}}, {{1, 2}, {1, 0}}})),
        testing::Bool()),
    paramToString);

class LowerCollectiveTest : public MultiDeviceTest,
                            public testing::WithParamInterface<
                                std::tuple<CommunicatorBackend, bool>> {
 protected:
  void SetUp() override;
};

void LowerCollectiveTest::SetUp() {
  MultiDeviceTest::SetUp();

  const auto& [backend_type, enable_host_ir_lowering] = GetParam();
  if (!communicator_->isBackendAvailable(backend_type)) {
    GTEST_SKIP() << "Backend not available: " << backend_type;
  }
  // getBackendForTeam throws an error if the requested backend type isn't
  // available. Therefore, we call it after the isBackendAvailable check.
  communicator_->setDefaultBackend(backend_type);

  if (enable_host_ir_lowering) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }
}

TEST_P(LowerCollectiveTest, Allgather) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  TensorView* in = makeContigTensor(2);
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  auto mesh = DeviceMesh::createForNumDevices(num_devices);
  in->setDeviceMesh(mesh);
  out->setDeviceMesh(mesh);
  in->axis(0)->parallelize(ParallelType::DIDx);

  at::Tensor unsharded_tensor =
      at::randn({num_devices, kTensorSize}, tensor_options);
  at::Tensor in_tensor = shardTensor(unsharded_tensor, in);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  assertIsCompiledToHostIrContainer(executor_cache);

  EXPECT_TRUE(at::equal(out_tensor, unsharded_tensor));
}

TEST_P(LowerCollectiveTest, Allgather_LoopSplit) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(num_devices);

  TensorView* in = makeContigTensor(1);
  in->setDeviceMesh(mesh);
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  in->split(0, num_devices, /*inner_split=*/false);
  in->axis(0)->parallelize(ParallelType::DIDx);
  in->setAllocationDomain(in->getLoopDomain(), true);

  out->split(0, num_devices, /*inner_split=*/false);
  out->setAllocationDomain(out->getLoopDomain(), true);

  at::Tensor unsharded_tensor =
      at::randn({num_devices * kTensorSize}, at::kFloat);
  at::Tensor in_tensor =
      shardTensor(unsharded_tensor, in).to(communicator_->device());

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor out_tensor =
      fec.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  assertIsCompiledToHostIrContainer(fec);

  EXPECT_TRUE(at::equal(out_tensor.cpu(), unsharded_tensor));
}

TEST_P(LowerCollectiveTest, Broadcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  TensorView* in = makeContigTensor(2);
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  auto mesh = DeviceMesh::createForNumDevices(num_devices);
  constexpr DeviceIdxType kRoot = 0;
  in->setDeviceMesh({kRoot});
  out->setDeviceMesh(mesh);

  at::Tensor unsharded_tensor =
      at::randn({num_devices, kTensorSize}, tensor_options);
  const auto device_id = communicator_->deviceId();
  at::Tensor in_tensor = unsharded_tensor.slice(0, device_id, device_id + 1);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();

  if (num_devices > 1) {
    assertIsCompiledToHostIrContainer(executor_cache);
  }

  EXPECT_TRUE(
      at::equal(out_tensor, unsharded_tensor.slice(0, kRoot, kRoot + 1)));
}

TEST_P(LowerCollectiveTest, Reduce) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  TensorView* in = makeContigTensor(2);
  TensorView* out = sum(in, {0});
  fusion->addInput(in);
  fusion->addOutput(out);

  auto mesh = DeviceMesh::createForNumDevices(num_devices);
  constexpr DeviceIdxType kRoot = 0;
  in->setDeviceMesh(mesh);
  out->setDeviceMesh({kRoot});
  in->axis(0)->parallelize(ParallelType::DIDx);

  at::Tensor unsharded_in_tensor =
      at::randn({num_devices, kTensorSize}, tensor_options);
  const auto device_id = communicator_->deviceId();
  at::Tensor in_tensor = shardTensor(unsharded_in_tensor, in);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  assertIsCompiledToHostIrContainer(executor_cache);

  if (device_id == kRoot) {
    // at::allclose instead of at::equal because addition is involved.
    EXPECT_TRUE(at::allclose(out_tensor, unsharded_in_tensor.sum(0)));
  }
}

TEST_P(LowerCollectiveTest, Allreduce) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  TensorView* in = makeContigTensor(2);
  TensorView* out = sum(in, {0});
  fusion->addInput(in);
  fusion->addOutput(out);

  auto mesh = DeviceMesh::createForNumDevices(num_devices);
  in->setDeviceMesh(mesh);
  out->setDeviceMesh(mesh);
  in->axis(0)->parallelize(ParallelType::DIDx);

  at::Tensor unsharded_in_tensor =
      at::randn({num_devices, kTensorSize}, tensor_options);
  at::Tensor in_tensor = shardTensor(unsharded_in_tensor, in);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  assertIsCompiledToHostIrContainer(executor_cache);

  EXPECT_TRUE(at::allclose(out_tensor, unsharded_in_tensor.sum(0)));
}

TEST_P(LowerCollectiveTest, Allreduce_Concrete) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  TensorView* in = makeContigConcreteTensor({num_devices, kTensorSize});
  // When `num_devices` is 1, the `sum` becomes a `SqueezeOp`, a good test for
  // lowering.
  TensorView* out = sum(in, {0});
  fusion->addInput(in);
  fusion->addOutput(out);

  auto mesh = DeviceMesh::createForNumDevices(num_devices);
  in->setDeviceMesh(mesh);
  out->setDeviceMesh(mesh);
  in->axis(0)->parallelize(ParallelType::DIDx);

  at::Tensor unsharded_in_tensor =
      at::randn({num_devices, kTensorSize}, tensor_options);
  at::Tensor in_tensor = shardTensor(unsharded_in_tensor, in);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  if (num_devices > 1) {
    assertIsCompiledToHostIrContainer(executor_cache);
  }

  EXPECT_TRUE(at::allclose(out_tensor, unsharded_in_tensor.sum(0)));
}

TEST_P(LowerCollectiveTest, ReduceScatter) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto d = communicator_->size();
  TensorView* in = makeContigTensor(2);
  TensorView* out = sum(in, {1});
  fusion->addInput(in);
  fusion->addOutput(out);

  auto mesh = DeviceMesh::createForNumDevices(d);

  in->setDeviceMesh(mesh);
  in->outer_split(1, d);
  in->axis(1)->parallelize(ParallelType::DIDx);
  in->setAllocationDomain(in->getLoopDomain(), true);

  out->setDeviceMesh(mesh);
  out->outer_split(1, d);
  out->axis(1)->parallelize(ParallelType::DIDx);
  out->outer_split(0, d);
  out->axis(0)->parallelize(ParallelType::DIDx);
  out->setAllocationDomain(out->getLoopDomain(), true);

  at::Tensor unsharded_in_tensor = at::randn({d * 2, d * 3}, tensor_options);
  at::Tensor in_tensor = shardTensor(unsharded_in_tensor, in);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();

  at::Tensor unsharded_out_tensor = unsharded_in_tensor.sum(1);
  EXPECT_TRUE(at::allclose(out_tensor, shardTensor(unsharded_out_tensor, out)));

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  const std::vector<SegmentedGroup*>& groups =
      runtime->fusionSegments()->groups();
  EXPECT_THAT(
      groups,
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::Reduction),
          HeuristicIs(SchedulerType::Communication)));
  for (auto* group : groups) {
    if (group->schedulerType() == SchedulerType::Communication) {
      EXPECT_EQ(group->inputs().at(0)->as<TensorView>()->nDims(), 3)
          << "This TV is the output of the local reduction and the input of "
             "the ReduceScatter. Its loop domain should contain three "
             "elements: in no particular order, [i{2d}, iDIDx{d}, r{3}]. "
             "nvFuser used to add an extra split and thus [i{d}, i{2}, "
             "iDIDx{d}, r{3}] as the loop domain. This is in "
             "theory valid but unnecessarily complicated and in practice "
             "caused some schedulers to fail. "
             "Many schedulers panic when they see the input fusion segment "
             "contains non-DID loop splits.";
    }
  }
}

TEST_P(LowerCollectiveTest, ReduceScatter_LogicalSplit) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  TensorView* in = makeContigTensor(3);
  TensorView* out = sum(in, {0});
  fusion->addInput(in);
  fusion->addOutput(out);

  auto mesh = DeviceMesh::createForNumDevices(num_devices);
  in->setDeviceMesh(mesh);
  out->setDeviceMesh(mesh);
  in->axis(0)->parallelize(ParallelType::DIDx);
  out->axis(1)->parallelize(ParallelType::DIDx);

  at::Tensor unsharded_in_tensor =
      at::randn({num_devices, num_devices, kTensorSize}, tensor_options);
  at::Tensor in_tensor = shardTensor(unsharded_in_tensor, in);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  assertIsCompiledToHostIrContainer(executor_cache);

  at::Tensor unsharded_out_tensor = unsharded_in_tensor.sum(0);
  EXPECT_TRUE(at::allclose(out_tensor, shardTensor(unsharded_out_tensor, out)));
}

TEST_P(LowerCollectiveTest, ReduceScatter_Allgather) {
  // Allreduce = ReduceScatter + Allgather
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(num_devices);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh(mesh);
  in->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* out = sum(in, {0});
  out->axis(1)->parallelize(ParallelType::DIDx);

  out = set(out);
  out->axis(0)->parallelize(ParallelType::Serial);

  fusion->addInput(in);
  fusion->addOutput(out);

  at::Tensor unsharded_in_tensor =
      at::randn({num_devices, num_devices, kTensorSize}, tensor_options);
  at::Tensor in_tensor = shardTensor(unsharded_in_tensor, in);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();

  EXPECT_TRUE(at::allclose(out_tensor, unsharded_in_tensor.sum(0)));
}

TEST_P(LowerCollectiveTest, ReduceScatterNoncontig) {
  if (communicator_->size() < 2) {
    GTEST_SKIP() << "This test exercises ReorderShardedAxisPass, and requires "
                    "at least 2 devices.";
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto d = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(d);

  TensorView* tv0 = makeConcreteTensor({5, d * 3, d * 7});
  TensorView* tv1 = sum(tv0, {1});

  tv0->setDeviceMesh(mesh);
  tv0->outer_split(1, d);
  tv0->axis(1)->parallelize(ParallelType::DIDx);

  tv1->setDeviceMesh(mesh);
  tv1->outer_split(1, d);
  tv1->axis(1)->parallelize(ParallelType::DIDx);

  tv1->outer_split(-1, d);
  tv1->axis(-2)->parallelize(ParallelType::DIDx);

  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  at::Tensor unsharded_in_tensor =
      at::randint(2, {5, d * 3, d * 7}, tensor_options);
  at::Tensor in_tensor = shardTensor(unsharded_in_tensor, 1, mesh);

  at::Tensor expected_output =
      shardTensor(unsharded_in_tensor.sum(1), -1, mesh);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();

  EXPECT_TRUE(out_tensor.t().is_contiguous());
  EXPECT_TRUE(at::equal(out_tensor, expected_output));

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::Communication),
          HeuristicIs(SchedulerType::Reduction)));
}

TEST_P(LowerCollectiveTest, AllreduceNoncontig) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto d = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(d);

  TensorView* tv0 = makeConcreteTensor({5, d * 3});
  tv0->setAllocationDomain(tv0->getLogicalDomain(), false);
  TensorView* tv1 = sum(tv0, {1});

  tv0->setDeviceMesh(mesh);
  tv0->outer_split(1, d);
  tv0->axis(1)->parallelize(ParallelType::DIDx);

  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  at::Tensor unsharded_in_tensor = at::randn({5, d * 3}, tensor_options);
  at::Tensor in_tensor = shardTensor(unsharded_in_tensor, 1, mesh);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();

  at::Tensor expected_output = unsharded_in_tensor.sum(1);
  EXPECT_TRUE(at::allclose(out_tensor, expected_output));

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();

  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::Communication),
          HeuristicIs(SchedulerType::Reduction)));
}

TEST_P(LowerCollectiveTest, Allgather_CompliantAllocation) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto d = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(d);

  TensorView* tv0 = makeConcreteTensor({5, d * 3});
  tv0->setAllocationDomain({tv0->axis(1), tv0->axis(0)}, true);

  TensorView* tv1 = set(tv0);
  tv1->setAllocationDomain({tv1->axis(1), tv1->axis(0)}, true);

  tv0->setDeviceMesh(mesh);
  tv0->outer_split(1, d);
  tv0->axis(1)->parallelize(ParallelType::DIDx);

  tv1->setDeviceMesh(mesh);

  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  at::Tensor unsharded_in_tensor = at::randn({d * 3, 5}, tensor_options);
  at::Tensor in_tensor = shardTensor(unsharded_in_tensor, 0, mesh).t();

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();

  EXPECT_TRUE(out_tensor.t().is_contiguous());
  EXPECT_TRUE(at::allclose(out_tensor, unsharded_in_tensor.t()));

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(HeuristicIs(SchedulerType::Communication)));
}

TEST_P(LowerCollectiveTest, Allgather_NonCompliantAllocation) {
  if (communicator_->size() < 2) {
    GTEST_SKIP() << "Should pass with one GPU, but doesn't.";
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto d = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(d);

  TensorView* tv0 = makeConcreteTensor({5, d * 3});
  tv0->setAllocationDomain(tv0->getLogicalDomain(), false);

  TensorView* tv1 = set(tv0);
  tv1->setAllocationDomain(tv1->getLogicalDomain(), true);

  tv0->setDeviceMesh(mesh);
  tv0->outer_split(1, d);
  tv0->axis(1)->parallelize(ParallelType::DIDx);

  tv1->setDeviceMesh(mesh);

  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  at::Tensor unsharded_in_tensor = at::randn({5, d * 3}, tensor_options);
  at::Tensor in_tensor = shardTensor(unsharded_in_tensor, 1, mesh);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();

  EXPECT_TRUE(out_tensor.is_contiguous());
  EXPECT_TRUE(at::allclose(out_tensor, unsharded_in_tensor));

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      Contains(HeuristicIs(SchedulerType::PointWise)).Times(2));
}

TEST_P(LowerCollectiveTest, Allgather_NoncontiguousOutput) {
  if (communicator_->size() < 2) {
    GTEST_SKIP() << "Should pass with one GPU, but doesn't.";
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto d = communicator_->size();
  const auto mesh = DeviceMesh::createForNumDevices(d);

  TensorView* in = makeSymbolicTensor(2);
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  out->setAllocationDomain({out->axis(1), out->axis(0)}, false);

  in->setDeviceMesh(mesh);
  out->setDeviceMesh(mesh);

  in->outer_split(1, d);
  in->axis(1)->parallelize(ParallelType::DIDx);

  at::Tensor unsharded_in_tensor = at::randn({2, d * 3}, tensor_options);
  at::Tensor in_tensor = shardTensor(unsharded_in_tensor, 1, mesh);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  EXPECT_TRUE(at::equal(out_tensor, unsharded_in_tensor));

  EXPECT_LT(out_tensor.stride(0), out_tensor.stride(1))
      << "`out` has been specified to be column major";

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::PointWise),
          HeuristicIs(SchedulerType::Communication)));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    LowerCollectiveTest,
    ::testing::Combine(
        testing::Values(CommunicatorBackend::kNccl),
        testing::Bool()),
    ([](const testing::TestParamInfo<std::tuple<CommunicatorBackend, bool>>&
            info) -> std::string {
      const auto& [backend_type, enable_host_ir_lowering] = info.param;
      std::stringstream ss;
      ss << backend_type;
      ss << (enable_host_ir_lowering ? "_HostIr" : "_NonHostIr");
      return ss.str();
    }));

} // namespace nvfuser