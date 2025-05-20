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
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using testing::Each;
using testing::Pointer;

namespace {
void assertIsCompiledToHostIrContainer(
    const FusionExecutorCache& executor_cache) {
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  if (isOptionEnabled(EnableOption::HostIrLowering)) {
    EXPECT_EQ(runtime->getHostIrEvaluator().canRun(), "");
    auto hicExprs =
        runtime->getHostIrEvaluator().getHostIrContainer().topLevelExprs();
    EXPECT_THAT(hicExprs, Contains(IsA<Communication>()))
        << "host ir container should have at least one communication";
  } else {
    EXPECT_EQ(runtime->executors().size(), 1);
    EXPECT_THAT(runtime->executors(), Each(Pointer(IsA<HostIrExecutor>())))
        << "failed to compile to a HostIrContainer with Communications";
  }
}
} // namespace

// This is made a macro instead of a function, because GTEST_SKIP can only be
// used in individual test cases or `SetUp` methods.
#define SKIP_IF_NOT_ENOUGH_DEVICES(in_mesh, out_mesh)                 \
  do {                                                                \
    const auto num_devices = communicator_->size();                   \
    for (const auto& mesh : {in_mesh, out_mesh}) {                    \
      for (const auto device_id : mesh.vector()) {                    \
        if (device_id >= num_devices) {                               \
          GTEST_SKIP() << "Mesh (" << mesh << ") requires more than " \
                       << num_devices << " devices.";                 \
        }                                                             \
      }                                                               \
    }                                                                 \
  } while (0)

using InOutMesh = std::pair<DeviceMesh, DeviceMesh>;

static constexpr int kTensorSize = 4;

class LowerGatherTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<std::tuple<InOutMesh, bool>> {};

TEST_P(LowerGatherTest, ) {
  EnableOptionsGuard opt_guard;
  const auto& [meshes, enable_host_ir_lowering] = GetParam();
  const auto& [in_mesh, out_mesh] = meshes;

  if (enable_host_ir_lowering) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }

  SKIP_IF_NOT_ENOUGH_DEVICES(in_mesh, out_mesh);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigTensor(2);
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  in->setDeviceMesh(in_mesh);
  out->setDeviceMesh(out_mesh);
  in->axis(0)->parallelize(ParallelType::DIDx);

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
    HostIrLowering,
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
  EnableOptionsGuard opt_guard;
  const auto& [meshes, enable_host_ir_lowering] = GetParam();
  const auto& [in_mesh, out_mesh] = meshes;

  if (enable_host_ir_lowering) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }

  SKIP_IF_NOT_ENOUGH_DEVICES(in_mesh, out_mesh);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigTensor(2);
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  in->setDeviceMesh(in_mesh);
  out->setDeviceMesh(out_mesh);
  out->axis(0)->parallelize(ParallelType::DIDx);

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
    HostIrLowering,
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
  EnableOptionsGuard opt_guard;
  const auto& [meshes, enable_host_ir_lowering] = GetParam();
  const auto& [in_mesh, out_mesh] = meshes;

  if (enable_host_ir_lowering) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }

  SKIP_IF_NOT_ENOUGH_DEVICES(in_mesh, out_mesh);

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
    HostIrLowering,
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

  EnableOptionsGuard enable_options_guard;
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

// This currently fails due to getShardingChanges reads root/logical only:
// https://github.com/NVIDIA/Fuser/blob/1dda106a946adcfd1526b83e4f2d4abebb9e32e4/csrc/multidevice/utils.cpp#L77.
// Will try to fix this in a follow-up PR and reenable the test.
TEST_P(LowerCollectiveTest, DISABLED_Allgather_LoopSplit_Noncontiguous) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(num_devices);

  TensorView* in = makeContigTensor(2);
  in->setDeviceMesh(mesh);
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  in->split(1, num_devices, /*inner_split=*/false);
  in->axis(1)->parallelize(ParallelType::DIDx);
  in->setAllocationDomain(in->getLoopDomain(), true);

  out->split(1, num_devices, /*inner_split=*/false);
  out->setAllocationDomain(out->getLoopDomain(), true);

  at::Tensor unsharded_tensor =
      at::arange(2 * num_devices * 3, at::kFloat).view({2, num_devices * 3});
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

TEST_P(LowerCollectiveTest, AllgatherLoopSplit_Noncontig) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // ProcessGroupNCCL requires the gathered axis to be outermost.
  // We change the allocation of tensorviews to reflect this.
  // We do not modify the logical shape of the tensorview.
  // This would still require one copy on each device if the input tensor is in
  // a different layout.
  const auto d = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(d);

  TensorView* tv0 = makeConcreteTensor({5, d * 3});
  TensorView* tv1 = set(tv0);

  tv0->setDeviceMesh(mesh);
  tv0->outer_split(1, d);
  tv0->axis(1)->parallelize(ParallelType::DIDx);

  tv1->setDeviceMesh(mesh);

  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  at::Tensor unsharded_in_tensor = at::randn({5, d*3}, tensor_options);
  at::Tensor in_tensor =
      shardTensor(unsharded_in_tensor, 1, mesh);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  EXPECT_TRUE(at::allclose(out_tensor, unsharded_in_tensor));
}

TEST_P(LowerCollectiveTest, ScatterLoopSplit) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto d = communicator_->size();
  auto full_mesh = DeviceMesh::createForNumDevices(d);

  DeviceMesh mesh_zero({0});
  TensorView* tv0 = makeConcreteTensor({5, d * 3});
  TensorView* tv1 = set(tv0);

  tv0->setDeviceMesh(mesh_zero);

  tv1->setDeviceMesh(full_mesh);
  tv1->outer_split(1, d);
  tv1->axis(1)->parallelize(ParallelType::DIDx);

  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  at::Tensor unsharded_in_tensor =
      at::randn({5, d * 3}, tensor_options);

  at::Tensor expected_output = shardTensor(unsharded_in_tensor, 1, full_mesh);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({unsharded_in_tensor})[0]
          .as<at::Tensor>();
  EXPECT_TRUE(at::allclose(out_tensor, expected_output));
}

INSTANTIATE_TEST_SUITE_P(
    HostIrLowering,
    LowerCollectiveTest,
    ::testing::Combine(
        testing::Values(CommunicatorBackend::kNccl, CommunicatorBackend::kUcc),
        testing::Values(false)),
    ([](const testing::TestParamInfo<std::tuple<CommunicatorBackend, bool>>&
            info) -> std::string {
      const auto& [backend_type, enable_host_ir_lowering] = info.param;
      std::stringstream ss;
      ss << backend_type;
      ss
          << (enable_host_ir_lowering ? "_HirLowerEnabled"
                                      : "_HirLowerDisabled");
      return ss.str();
    }));
} // namespace nvfuser
