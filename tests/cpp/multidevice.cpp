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
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <options.h>
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
  return shardTensor(tensor, getShardedAxis(tv), tv->getDeviceMesh());
}

at::Tensor MultiDeviceTest::shardTensor(
    at::Tensor tensor,
    int64_t axis,
    const DeviceMesh& mesh) {
  const auto device_id = communicator_->deviceId();
  auto i = mesh.idxOf(device_id);
  auto extent = tensor.size(axis);
  auto nslices = mesh.size();
  NVF_CHECK(
      extent % nslices == 0, "Sharded axis must be evenly divisble by mesh");
  auto stride = extent / nslices;
  // TODO: returning slice 0 temporarily when device is not in the mesh.
  i = (i < 0) ? 0 : i;
  auto slice = tensor.slice(axis, i * stride, (i + 1) * stride).contiguous();
  // Temporary until https://github.com/NVIDIA/Fuser/issues/2563. Adds DIDx
  // axis in front representing the sharded extent of the tensor.
  if (stride > 1) {
    slice = slice.unsqueeze(0);
  }
  return slice;
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
    auto ref_output = shardTensor(ref_unsharded_outputs.at(i), output_tv);
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
    auto input = shardTensor(unsharded_inputs.at(i).toTensor(), input_tv);
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
      std::move(fusion), *communicator_, host_ir_executor_params);
  auto error_msg = runtime->validate();
  if (error_msg != "") {
    GTEST_SKIP() << error_msg;
  }
  outputs = runtime->runWithInput(inputs);

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

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::AddGlobalTestEnvironment(new nvfuser::MultiDeviceTestEnvironment());
  return RUN_ALL_TESTS();
}
