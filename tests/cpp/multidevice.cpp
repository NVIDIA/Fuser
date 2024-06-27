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

#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <options.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>
#include <torch/cuda.h>

namespace nvfuser {

MultiDeviceTest::MultiDeviceTest() {
  communicator_ = getOrCreateCommunicator();
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

/*static*/ at::Tensor MultiDeviceTest::shardTensor(
    at::Tensor tensor,
    TensorView* tv,
    DeviceIdxType deviceId) {
  if (!isSharded(tv)) {
    return tensor;
  }
  auto sharded_dim = getShardedAxis(tv);
  auto i = tv->getDeviceMesh().idxOf(deviceId);
  // TODO: returning slice 0 temporarily when device is not in the mesh.
  i = (i < 0) ? 0 : i;
  return tensor.slice(sharded_dim, i, i + 1).contiguous();
}

/*static*/ Communicator* MultiDeviceTest::getOrCreateCommunicator() {
  static Communicator* communicator = new Communicator();
  return communicator;
}

void PipelineTest::validate(bool validate_with_prescribed_values) {
  if (!validate_with_prescribed_values) {
    // execute the fusion on one device without pipeline scheduling
    auto fusion_copy = std::make_unique<Fusion>(*runtime->completeFusion());
    unshard(fusion_copy.get());
    FusionExecutorCache unsharded_fec(std::move(fusion_copy));
    ref_unsharded_outputs = unsharded_fec.runFusionWithInputs(unsharded_inputs);
  }

  if (debug_print) {
    std::stringstream ss;
    std::string indent = "  ";
    ss << "Device " << communicator_->deviceId()
       << "'s expected (unsharded) outputs:{\n";
    for (auto& t : ref_unsharded_outputs) {
      ss << indent << t;
    }
    ss << "\n}";
    std::cout << ss.str() << std::endl;
  }

  ASSERT_EQ(ref_unsharded_outputs.size(), outputs.size());
  for (int i : c10::irange(runtime->completeFusion()->outputs().size())) {
    ASSERT_TRUE(runtime->completeFusion()->outputs().at(i)->isA<TensorView>());
    auto output_tv =
        runtime->completeFusion()->outputs().at(i)->as<TensorView>();
    if (!output_tv->getDeviceMesh().has(communicator_->deviceId())) {
      continue;
    }
    auto ref_output = shardTensor(
        ref_unsharded_outputs.at(i), output_tv, communicator_->deviceId());
    auto obtained_output = outputs.at(i);
    EXPECT_TRUE(torch::allclose(ref_output, obtained_output))
        << "Device " << communicator_->deviceId() << " has unexpected output "
        << i << " corresponding to tv " << output_tv
        << ". Expected values: " << ref_output
        << ", obtained values: " << obtained_output;
  }
}

// Run and validate a pipeline
// with given (possibly sharded) inputs
void PipelineTest::executeAndValidate(bool validate_with_prescribed_values) {
  ASSERT_EQ(unsharded_inputs.size(), fusion->inputs().size());
  for (int i : c10::irange(fusion->inputs().size())) {
    ASSERT_TRUE(fusion->inputs().at(i)->isA<TensorView>());
    auto input_tv = fusion->inputs().at(i)->as<TensorView>();
    auto input = shardTensor(
        unsharded_inputs.at(i).toTensor(), input_tv, communicator_->deviceId());
    inputs.push_back(input);
  }

  if (debug_print) {
    if (!communicator_->deviceId()) {
      fusion->printKernel();
    }
    std::stringstream ss;
    std::string indent = "  ";
    ss << "Device " << communicator_->deviceId() << "'s inputs:{\n";
    for (auto& t : inputs) {
      ss << indent << t;
    }
    ss << "\n}";
    std::cout << ss.str() << std::endl;
  }

  runtime = std::make_unique<MultiDeviceExecutor>(
      std::move(fusion), *communicator_, multi_device_executor_params);
  auto error_msg = runtime->validate();
  if (error_msg != "") {
    GTEST_SKIP() << error_msg;
  }
  outputs = runtime->runWithInput(inputs, l_params);

  if (debug_print) {
    if (!communicator_->deviceId()) {
      runtime->print();
    }
    std::stringstream ss;
    std::string indent = "  ";
    ss << "Device " << communicator_->deviceId() << "'s outputs:{\n";
    for (auto& t : outputs) {
      ss << indent << t;
    }
    ss << "\n}";
    std::cout << ss.str() << std::endl;
  }
  validate(validate_with_prescribed_values);
}

PipelineTest::PipelineTest() {
  fusion = std::make_unique<Fusion>();
  communicator_->setDefaultBackend(CommunicatorBackend::nccl);
}

} // namespace nvfuser
