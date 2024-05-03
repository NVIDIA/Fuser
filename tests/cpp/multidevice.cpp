// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
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
  communicator = getOrCreateCommunicator();
  tensor_options =
      at::TensorOptions().dtype(at::kFloat).device(communicator->device());
  debug_print = getNvFuserEnv("MULTIDEVICE_DEBUG_PRINT") != nullptr;
  disable_skip = getNvFuserEnv("MULTIDEVICE_DISABLE_SKIP") != nullptr;
}

MultiDeviceTest::~MultiDeviceTest() {
  // Force all processes to synchronize at a barrier between tests. It slightly
  // slows the tests down, but makes it much easier to isolate a failing test.
  // Without this, if a test fails such that a subset of processes fail, then
  // some processes will move onto another tests and timeout later.
  if (communicator->is_available()) {
    communicator->barrier();
  }
}

void MultiDeviceTest::SetUp() {
  // Set the same random seed for all processes.
  NVFuserTest::SetUp();

  if (!disable_skip && !communicator->is_available()) {
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
  int i = 0;
  const auto& devices = tv->getDeviceMesh().vector();
  auto it = std::find(devices.begin(), devices.end(), deviceId);
  if (it != devices.end()) {
    i = std::distance(devices.begin(), it);
  }
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

PipelineTest::PipelineTest() {
  fusion = std::make_unique<Fusion>();
  communicator->setDefaultBackend(CommunicatorBackend::nccl);
}

} // namespace nvfuser
