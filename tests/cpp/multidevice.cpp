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
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>
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
  if (getNvFuserEnv("MULTIDEVICE_DISABLE_SKIP")) {
    disable_skip_ = true;
  }
}

void MultiDeviceEnvironment::TearDown() {
  if (communicator_->is_available()) {
    communicator_->barrier();
  }
  communicator_.reset();
}

void MultiDeviceTest::SetUp() {
  NVFuserTest::SetUp();
  communicator = multidevice_env->communicator();
  debug_print = multidevice_env->debugPrint();
  do_barrier_at_test =
      multidevice_env->doBarrierAtTest() && communicator->is_available();
  disable_skip = multidevice_env->disableSkip();
  if (!disable_skip &&
      (!communicator->is_available() || communicator->size() < 2 ||
       torch::cuda::device_count() < 2)) {
    GTEST_SKIP() << "This test needs at least 2 GPUs and 2 ranks";
  }
  tensor_options =
      at::TensorOptions().dtype(at::kFloat).device(communicator->device());
}

void MultiDeviceTest::TearDown() {
  if (do_barrier_at_test && communicator->is_available()) {
    communicator->barrier();
  }
  NVFuserTest::TearDown();
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
    ss << "Device " << communicator->deviceId()
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
    if (!output_tv->getDeviceMesh().has(communicator->deviceId())) {
      continue;
    }
    auto ref_output = shardTensor(
        ref_unsharded_outputs.at(i), output_tv, communicator->deviceId());
    auto obtained_output = outputs.at(i);
    EXPECT_TRUE(torch::allclose(ref_output, obtained_output))
        << "Device " << communicator->deviceId() << " has unexpected output "
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
        unsharded_inputs.at(i).toTensor(), input_tv, communicator->deviceId());
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

  runtime = std::make_unique<MultiDeviceExecutor>(
      std::move(fusion), *communicator, multi_device_executor_params);
  auto error_msg = runtime->validate();
  if (error_msg != "") {
    GTEST_SKIP() << error_msg;
  }
  outputs = runtime->runWithInput(inputs, l_params);

  if (debug_print) {
    if (!communicator->deviceId()) {
      runtime->print();
    }
    std::stringstream ss;
    std::string indent = "  ";
    ss << "Device " << communicator->deviceId() << "'s outputs:{\n";
    for (auto& t : outputs) {
      ss << indent << t;
    }
    ss << "\n}";
    std::cout << ss.str() << std::endl;
  }
  validate(validate_with_prescribed_values);
}

void PipelineTest::SetUp() {
  MultiDeviceTest::SetUp();
  fusion = std::make_unique<Fusion>();
  communicator->setDefaultBackend(CommunicatorBackend::nccl);
}

} // namespace nvfuser

#endif
