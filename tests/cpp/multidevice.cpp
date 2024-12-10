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

  // NVFUSER_MULTIDEVICE_WAIT_DEBUGGER_AT_RANK can be used to attach gdb to one
  // of the processes for debugging.
  //
  // When an mpirun fails, it usually prints out something like
  // ```
  // mpirun detected that one or more processes exited with non-zero status,
  // thus causing the job to be terminated. The first process to do so was:
  //
  //   Process name: [[17665,1],0]
  //   Exit code:    1
  // ```
  // The last bit of the process name (0 in this case) is the rank of the first
  // failing process, and usually the rank to debug.
  //
  // Sometimes, multiple processes fail, and a failed, non-gdb'ed process can
  // cause `mpirun` to terminate the entire job including the process being
  // gdb'ed. For that, I use `mpirun -continuous` so `mpirun` keeps running the
  // process being gdb'ed.
  char* rank_to_debug_str = getNvFuserEnv("MULTIDEVICE_WAIT_DEBUGGER_AT_RANK");
  if (rank_to_debug_str != nullptr) {
    const DeviceIdxType rank_to_debug = std::stol(rank_to_debug_str);

    static std::once_flag once;
    std::call_once(once, [&]() {
      // Catch exceptions so call_once always flips `once` and executes this
      // functor only once.
      try {
        waitForDebuggerAtRank(rank_to_debug);
      } catch (const std::exception& e) {
        TORCH_WARN("Failed to wait for debugger: ", e.what());
      }
    });
  }
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

void MultiDeviceTest::waitForDebuggerAtRank(const DeviceIdxType rank) {
  NVF_CHECK(
      rank >= 0 && rank < communicator_->size(),
      "rank=",
      rank,
      " must be in the range of [0,",
      communicator_->size(),
      ").");

  if (communicator_->deviceId() == rank) {
    volatile bool waiting = true;
    auto pid = getpid();
    std::cerr << "Process " << pid
              << " is waiting for the debugger. To continue debugging, "
              << "start gdb, `attach " << pid
              << "`, `set var waiting=false`, and `fini`." << std::endl;
    while (waiting) { // Please change `waiting` in the debugger.
    }
    std::cerr << "Process " << getpid() << " finished waiting." << std::endl;
  }

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
