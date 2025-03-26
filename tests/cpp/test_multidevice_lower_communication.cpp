// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gmock/gmock-matchers.h>
#include <gmock/gmock-more-matchers.h>
#include <gtest/gtest.h>

#include <ops/all_ops.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using testing::Each;
using testing::IsTrue;
using testing::Pointer;
using testing::Property;

namespace {
void assertIsCompiledToHostIrContainer(
    const FusionExecutorCache& executor_cache) {
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  if (isOptionEnabled(EnableOption::HostIrLowering)) {
    EXPECT_EQ(runtime->getHostIrEvaluator()->canRun(), "");
  } else {
    EXPECT_EQ(runtime->executors().size(), 1);
    EXPECT_THAT(
        runtime->executors(),
        Each(Pointer(Property(
            "is a HostIrExecutor",
            &ExecutorAbstract::isA<HostIrExecutor>,
            IsTrue()))))
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

class LowerGatherTest : public MultiDeviceTest,
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

INSTANTIATE_TEST_SUITE_P(
    HostIrLowering,
    LowerGatherTest,
    // Create product of InOutMesh configurations and HostIrLowering options
    testing::Combine(
        testing::ValuesIn(std::vector<InOutMesh>(
            {{{0, 1}, {0}},
             {{0, 1}, {1}},
             {{1, 2}, {0, 2}}})),
        testing::Bool()),
    [](const testing::TestParamInfo<std::tuple<InOutMesh, bool>>& info) {
      const auto& meshes = std::get<0>(info.param);
      const auto& in_mesh = meshes.first;
      const auto& out_mesh = meshes.second;
      const auto enable_hir = std::get<1>(info.param);
      
      std::stringstream ss;
      ss << "InMesh";
      for (auto id : in_mesh.vector()) {
        ss << "_" << id;
      }
      ss << "_OutMesh";
      for (auto id : out_mesh.vector()) {
        ss << "_" << id;
      }
      ss << (enable_hir ? "_HirEnabled" : "_HirDisabled");
      
      return ss.str();
    });

class LowerScatterTest : public MultiDeviceTest,
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
            {{{0}, {0, 1}}, //
             {{1}, {0, 1}}, //
             {{0, 2}, {1, 2}}})),
        testing::Bool()),
    [](const testing::TestParamInfo<std::tuple<InOutMesh, bool>>& info) {
      const auto& meshes = std::get<0>(info.param);
      const auto& in_mesh = meshes.first;
      const auto& out_mesh = meshes.second;
      const auto enable_hir = std::get<1>(info.param);
      
      std::stringstream ss;
      ss << "InMesh";
      for (auto id : in_mesh.vector()) {
        ss << "_" << id;
      }
      ss << "_OutMesh";
      for (auto id : out_mesh.vector()) {
        ss << "_" << id;
      }
      ss << (enable_hir ? "_HirEnabled" : "_HirDisabled");
      
      return ss.str();
    });

class LowerSendRecvTest : public MultiDeviceTest,
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
            {{{0}, {1}},
             {{1}, {0}},
             {{1, 2}, {0, 1}},
             {{1, 2}, {1, 0}}})),
        testing::Bool()),
    [](const testing::TestParamInfo<std::tuple<InOutMesh, bool>>& info) {
      const auto& meshes = std::get<0>(info.param);
      const auto& in_mesh = meshes.first;
      const auto& out_mesh = meshes.second;
      const auto enable_hir = std::get<1>(info.param);
      
      std::stringstream ss;
      ss << "InMesh";
      for (auto id : in_mesh.vector()) {
        ss << "_" << id;
      }
      ss << "_OutMesh";
      for (auto id : out_mesh.vector()) {
        ss << "_" << id;
      }
      ss << (enable_hir ? "_HirEnabled" : "_HirDisabled");
      
      return ss.str();
    });

class LowerCollectiveTest : public MultiDeviceTest,
                          public testing::WithParamInterface<bool> {};

TEST_P(LowerCollectiveTest, Allgather) {
  EnableOptionsGuard opt_guard;
  const bool enable_host_ir_lowering = GetParam();

  if (enable_host_ir_lowering) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }
  
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
  EnableOptionsGuard opt_guard;
  const bool enable_host_ir_lowering = GetParam();

  if (enable_host_ir_lowering) {
    // Skip this test when HostIrLowering is enabled
    GTEST_SKIP() << "Disabled for HostIrLowering enabled configuration";
  }
  
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
  EnableOptionsGuard opt_guard;
  const bool enable_host_ir_lowering = GetParam();

  if (enable_host_ir_lowering) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }
  
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
  EnableOptionsGuard opt_guard;
  const bool enable_host_ir_lowering = GetParam();

  if (enable_host_ir_lowering) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }
  
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
  EnableOptionsGuard opt_guard;
  const bool enable_host_ir_lowering = GetParam();

  if (enable_host_ir_lowering) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }
  
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
  EnableOptionsGuard opt_guard;
  const bool enable_host_ir_lowering = GetParam();

  if (enable_host_ir_lowering) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }
  
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
  EnableOptionsGuard opt_guard;
  const bool enable_host_ir_lowering = GetParam();

  if (enable_host_ir_lowering) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }
  
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
  EnableOptionsGuard opt_guard;
  const bool enable_host_ir_lowering = GetParam();

  if (enable_host_ir_lowering) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }
  
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
  EnableOptionsGuard opt_guard;
  const bool enable_host_ir_lowering = GetParam();

  if (enable_host_ir_lowering) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }
  
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

INSTANTIATE_TEST_SUITE_P(
    HostIrLowering,
    LowerCollectiveTest,
    testing::Bool(),
    [](const testing::TestParamInfo<bool>& info) {
      return info.param ? "HirLowerEnabled" : "HirLowerDisabled";
    });

} // namespace nvfuser
