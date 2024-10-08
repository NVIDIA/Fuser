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

namespace {
void assertIsCompiledToHostIrContainer(const FusionExecutorCache& fec) {
  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  const std::vector<FusionExecutor>& fes = runtime->executors();
  EXPECT_THAT(fes, testing::SizeIs(1));
  for (const auto& fe : fes) {
    EXPECT_TRUE(fe.hostIrContainer())
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
                        public testing::WithParamInterface<InOutMesh> {};

TEST_P(LowerGatherTest, ) {
  const auto& [in_mesh, out_mesh] = GetParam();
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

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  assertIsCompiledToHostIrContainer(fec);

  if (out_mesh.has(device_id)) {
    EXPECT_TRUE(at::equal(out_tensor, unsharded_tensor));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    LowerGatherTest,
    // Trick to enforce clang-format to break lines for readability.
    testing::ValuesIn(std::vector<InOutMesh>(
        {{{0, 1}, {0}}, //
         {{0, 1}, {1}}, //
         {{1, 2}, {0, 2}}})));

class LowerScatterTest : public MultiDeviceTest,
                         public testing::WithParamInterface<InOutMesh> {};

TEST_P(LowerScatterTest, ) {
  const auto& [in_mesh, out_mesh] = GetParam();
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

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor out_tensor = fec.runFusionWithInputs({unsharded_tensor})[0];
  assertIsCompiledToHostIrContainer(fec);

  if (out_mesh.has(device_id)) {
    EXPECT_TRUE(at::equal(out_tensor, shardTensor(unsharded_tensor, out)));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    LowerScatterTest,
    testing::ValuesIn(std::vector<InOutMesh>(
        {{{0}, {0, 1}}, //
         {{1}, {0, 1}}, //
         {{0, 2}, {1, 2}}})));

class LowerSendRecvTest : public MultiDeviceTest,
                          public testing::WithParamInterface<InOutMesh> {};

TEST_P(LowerSendRecvTest, ) {
  const auto& [in_mesh, out_mesh] = GetParam();
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

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  assertIsCompiledToHostIrContainer(fec);

  if (out_mesh.has(device_id)) {
    EXPECT_TRUE(at::equal(out_tensor, shardTensor(unsharded_tensor, out)));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    LowerSendRecvTest,
    testing::ValuesIn(std::vector<InOutMesh>(
        {{{0}, {1}}, //
         {{1}, {0}}, //
         {{1, 2}, {0, 1}}, //
         {{1, 2}, {1, 0}}})));

using LowerCollectiveTest = MultiDeviceTest;

TEST_F(LowerCollectiveTest, Allgather) {
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

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  assertIsCompiledToHostIrContainer(fec);

  EXPECT_TRUE(at::equal(out_tensor, unsharded_tensor));
}

TEST_F(LowerCollectiveTest, Broadcast) {
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

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  if (num_devices > 1) {
    assertIsCompiledToHostIrContainer(fec);
  }

  EXPECT_TRUE(
      at::equal(out_tensor, unsharded_tensor.slice(0, kRoot, kRoot + 1)));
}

TEST_F(LowerCollectiveTest, Reduce) {
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

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  assertIsCompiledToHostIrContainer(fec);

  if (device_id == kRoot) {
    // at::allclose instead of at::equal because addition is involved.
    EXPECT_TRUE(at::allclose(out_tensor, unsharded_in_tensor.sum(0)));
  }
}

TEST_F(LowerCollectiveTest, Allreduce) {
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

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  assertIsCompiledToHostIrContainer(fec);

  EXPECT_TRUE(at::allclose(out_tensor, unsharded_in_tensor.sum(0)));
}

TEST_F(LowerCollectiveTest, Allreduce_Concrete) {
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

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  if (num_devices > 1) {
    assertIsCompiledToHostIrContainer(fec);
  }

  EXPECT_TRUE(at::allclose(out_tensor, unsharded_in_tensor.sum(0)));
}

TEST_F(LowerCollectiveTest, ReduceScatter) {
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

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  assertIsCompiledToHostIrContainer(fec);

  at::Tensor unsharded_out_tensor = unsharded_in_tensor.sum(0);
  EXPECT_TRUE(at::allclose(out_tensor, shardTensor(unsharded_out_tensor, out)));
}

TEST_F(LowerCollectiveTest, ReduceScatter_Allgather) {
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

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];

  EXPECT_TRUE(at::allclose(out_tensor, unsharded_in_tensor.sum(0)));
}

} // namespace nvfuser
