// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <sys/types.h>
#include <unistd.h>
#include <mutex>

#ifdef NVFUSER_DISTRIBUTED
#include <torch/csrc/distributed/c10d/debug.h>
#else
#include <multidevice/c10d_mock.h>
#endif
#include <torch/cuda.h>

#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <options.h>
#include <runtime/allocations.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

void MultiDeviceTestEnvironment::TearDown() {
  Communicator::getInstance().cleanup();
}

MultiDeviceTest::MultiDeviceTest() {
  // Enable logging in c10d so debug messages can be printed out via
  // `TORCH_DISTRIBUTED_DEBUG`.
  c10d::setDebugLevelFromEnvironment();

  communicator_ = &Communicator::getInstance();
  tensor_options =
      at::TensorOptions().dtype(at::kFloat).device(communicator_->device());
  debug_print = getNvFuserEnv("MULTIDEVICE_DEBUG_PRINT") != nullptr;
  disable_skip = getNvFuserEnv("MULTIDEVICE_DISABLE_SKIP") != nullptr;
}

MultiDeviceTest::~MultiDeviceTest() {
  // Force all processes to synchronize at a barrier between tests. It slightly
  // slows the tests down, but makes it much easier to isolate a failing test.
  // Without this, if a test fails such that a subset of processes fail, then
  // some processes will move onto another tests and timeout later.
  if (communicator_->is_available()) {
    communicator_->barrier();
  }
}

void MultiDeviceTest::SetUp() {
  // Set the same random seed for all processes.
  NVFuserTest::SetUp();

  if (!disable_skip && !communicator_->is_available()) {
    GTEST_SKIP() << "This test needs an available communicator.";
  }
}

at::Tensor MultiDeviceTest::shardTensor(at::Tensor tensor, TensorView* tv) {
  if (!isSharded(tv)) {
    return tensor;
  }
  NVF_ERROR(tv->hasDeviceMesh(), "`tv` has no DeviceMesh: ", tv);
  return shardTensor(
      tensor,
      getShardedLogicalAxis(tv, ParallelType::DIDx),
      tv->getDeviceMesh());
}

at::Tensor MultiDeviceTest::shardTensor(
    at::Tensor tensor,
    const int64_t axis,
    const DeviceMesh& mesh) {
  const auto device_id = communicator_->deviceId();
  return nvfuser::shardTensor(tensor, axis, mesh, device_id);
}

} // namespace nvfuser

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::AddGlobalTestEnvironment(new nvfuser::MultiDeviceTestEnvironment());
  return RUN_ALL_TESTS();
}
