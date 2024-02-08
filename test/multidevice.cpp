// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef NVFUSER_DISTRIBUTED
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <options.h>
#include <test/multidevice.h>
#include <test/validator.h>
#include <torch/cuda.h>

namespace nvfuser {

auto multidevice_env = static_cast<MultiDeviceEnvironment*>(
    testing::AddGlobalTestEnvironment(new MultiDeviceEnvironment));

void MultiDeviceEnvironment::SetUp() {
  communicator_ = std::make_unique<Communicator>();
  if (getNvFuserEnv("MULTIDEVICE_DEBUG_PRINT")) {
    debug_print_ = true;
  }
  if (getNvFuserEnv("MULTIDEVICE_DEBUG_BARRIER")) {
    do_barrier_at_test_ = true;
  }
  if (getNvFuserEnv("MULTIDEVICE_TIME_PRINT")) {
    time_print_ = true;
  }
  if (getNvFuserEnv("MULTIDEVICE_DISABLE_SKIP")) {
    disable_skip_ = true;
  }
}

void MultiDeviceTest::SetUp() {
  NVFuserTest::SetUp();
  communicator = multidevice_env->communicator();
  debug_print = multidevice_env->debugPrint();
  do_barrier_at_test =
      multidevice_env->doBarrierAtTest() && communicator->is_available();
  time_print = multidevice_env->timePrint() && communicator->is_available();
  bool disable_skip = multidevice_env->disableSkip();
  if (!disable_skip && (!communicator->is_available() || communicator->size() < 2 ||
      torch::cuda::device_count() < 2)) {
    GTEST_SKIP() << "This test needs at least 2 GPUs and 2 ranks";
  }
  tensor_options =
      at::TensorOptions().dtype(at::kFloat).device(communicator->device());
  recordEvent("init");
}

void MultiDeviceTest::TearDown() {
  if (do_barrier_at_test) {
    recordEvent("final barrier");
    communicator->barrier();
  }
  recordEvent("cleanup");
  if (time_print) {
    printTimes();
  }
  NVFuserTest::TearDown();
}

void MultiDeviceTest::recordEvent(const std::string name) {
  times.push_back(
      std::make_pair(name, std::chrono::high_resolution_clock::now()));
}

void MultiDeviceTest::printTimes() {
  std::stringstream ss;
  auto test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  ss << "Rank " << communicator->deviceId() << " -- test "
     << test_info->test_suite_name() << "." << test_info->name()
     << " -- Timestamps: {\n";
  for (auto i : c10::irange(times.size() - 1)) {
    auto [event_name, time] = times[i];
    auto [_, next_time] = times[i + 1];
    ss << "  " << event_name << ": "
       << std::chrono::duration_cast<std::chrono::milliseconds>(
              next_time - time)
              .count()
       << " ms\n";
  }
  ss << "}";
  std::cout << ss.str() << std::endl;
}

void CommunicationTest::SetUp() {
  MultiDeviceTest::SetUp();
  if (!communicator->isBackendAvailable(GetParam())) {
    GTEST_SKIP() << "Backend not available";
  }
  all_ranks = std::vector<DeviceIdxType>(communicator->size());
  std::iota(all_ranks.begin(), all_ranks.end(), 0);
}

void CommunicationTest::validate(at::Tensor obtained, at::Tensor expected) {
  NVF_ERROR(
      obtained.equal(expected),
      "Device ",
      communicator->deviceId(),
      " expected tensor:\n",
      expected,
      "\nbut obtained tensor:\n",
      obtained);
}

void CommunicationTest::resetDstBuffers() {
  for (auto& buf : params.dst_bufs) {
    buf.copy_(at::full(tensor_size, nan(""), tensor_options));
  }
}

// Utility function used for validation in the tests
// It compares the given (possibly sharded) output with the result of the Fusion
// run on a single device with the given (possibly sharded) inputs
void PipelineTest::validate() {
  // execute the fusion on one device without pipeline scheduling
  auto fusion_copy = std::make_unique<Fusion>(*runtime->fusion());
  unshard(fusion_copy.get());
  FusionExecutorCache unsharded_fec(std::move(fusion_copy));
  recordEvent("run unsharded fusion");
  auto ref_unsharded_outputs =
      unsharded_fec.runFusionWithInputs(unsharded_inputs);

  if (debug_print) {
    std::stringstream ss;
    std::string indent = "  ";
    ss << "Device " << communicator->deviceId()
       << "'s expected (unsharded) outputs:{\n";
    for (auto& t : ref_unsharded_outputs) {
      ss << indent << t;
    }
    ss << "\n}";
    std::cout << ss.str() << std::endl;
  }

  recordEvent("validate unsharded fusion");
  GTEST_ASSERT_EQ(ref_unsharded_outputs.size(), outputs.size());
  for (int i : c10::irange(runtime->fusion()->outputs().size())) {
    GTEST_ASSERT_TRUE(runtime->fusion()->outputs().at(i)->isA<TensorView>());
    auto output_tv = runtime->fusion()->outputs().at(i)->as<TensorView>();
    if (!output_tv->getDeviceMesh().has(communicator->deviceId())) {
      continue;
    }
    auto ref_output = isSharded(output_tv) ? shardTensor(
                                                 ref_unsharded_outputs.at(i),
                                                 output_tv->getDeviceMesh(),
                                                 communicator->deviceId())
                                           : ref_unsharded_outputs.at(i);
    auto obtained_output = outputs.at(i);
    GTEST_EXPECT_TRUE(torch::allclose(ref_output, obtained_output))
        << "Device " << communicator->deviceId() << " has unexpected output "
        << i << " corresponding to tv " << output_tv
        << ". Expected values: " << ref_output
        << ", obtained values: " << obtained_output;
  }
}

// Run and validate a pipeline
// with given (possibly sharded) inputs
void PipelineTest::execute() {
  GTEST_ASSERT_EQ(unsharded_inputs.size(), fusion->inputs().size());
  for (int i : c10::irange(fusion->inputs().size())) {
    GTEST_ASSERT_TRUE(fusion->inputs().at(i)->isA<TensorView>());
    auto input_tv = fusion->inputs().at(i)->as<TensorView>();
    auto input = isSharded(input_tv) ? shardTensor(
                                           unsharded_inputs.at(i).toTensor(),
                                           input_tv->getDeviceMesh(),
                                           communicator->deviceId())
                                     : unsharded_inputs.at(i).toTensor();
    inputs.push_back(input);
  }

  if (debug_print) {
    if (!communicator->deviceId()) {
      fusion->printKernel();
    }
    std::stringstream ss;
    std::string indent = "  ";
    ss << "Device " << communicator->deviceId() << "'s inputs:{\n";
    for (auto& t : inputs) {
      ss << indent << t;
    }
    ss << "\n}";
    std::cout << ss.str() << std::endl;
  }

  recordEvent("runtime instantiation");
  runtime =
      std::make_unique<MultiDeviceExecutor>(std::move(fusion), *communicator);
  auto error_msg = runtime->validate();
  if (error_msg != "") {
    GTEST_SKIP() << error_msg;
  }

  recordEvent("run the multidevice fusion");
  outputs = runtime->runWithInput(inputs);

  if (debug_print) {
    std::stringstream ss;
    std::string indent = "  ";
    ss << "Device " << communicator->deviceId() << "'s outputs:{\n";
    for (auto& t : outputs) {
      ss << indent << t;
    }
    ss << "\n}";
    std::cout << ss.str() << std::endl;
  }
}

void PipelineTest::SetUp() {
  MultiDeviceTest::SetUp();
  fusion = std::make_unique<Fusion>();
  communicator->setDefaultBackend(CommunicatorBackend::nccl);
}

} // namespace nvfuser

#endif
