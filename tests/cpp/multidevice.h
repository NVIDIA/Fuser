// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef NVFUSER_DISTRIBUTED
#pragma once

#include <multidevice/communication.h>
#include <multidevice/communicator.h>
#include <multidevice/executor.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

class MultiDeviceEnvironment : public testing::Environment {
 public:
  void SetUp() override;
  void TearDown() override;

  Communicator* communicator() const {
    NVF_ERROR(communicator_ != nullptr);
    return communicator_.get();
  }

  bool debugPrint() const {
    return debug_print_;
  }

  bool doBarrierAtTest() const {
    return do_barrier_at_test_;
  }

  bool disableSkip() const {
    return disable_skip_;
  }

 private:
  std::unique_ptr<Communicator> communicator_ = nullptr;
  bool debug_print_ = false;
  bool do_barrier_at_test_ = false;
  bool disable_skip_ = false;
};

class MultiDeviceTest : public NVFuserTest {
 public:
  static at::Tensor shardTensor(
      at::Tensor tensor,
      const DeviceMesh& mesh,
      DeviceIdxType deviceId) {
    int i = 0;
    auto devices = mesh.vector();
    auto it = find(devices.begin(), devices.end(), deviceId);
    if (it != devices.end()) {
      i = std::distance(devices.begin(), it);
    }
    return tensor.index({at::indexing::Slice(i, i + 1), "..."});
  }

 protected:
  void SetUp() override;
  void TearDown() override;
  Communicator* communicator;
  c10::TensorOptions tensor_options;
  bool debug_print;
  bool do_barrier_at_test;
  bool disable_skip;
};

class CommunicationTest
    : public MultiDeviceTest,
      public ::testing::WithParamInterface<CommunicatorBackend> {
 protected:
  void SetUp() override;
  void validate(at::Tensor obtained, at::Tensor expected);
  void resetDstBuffers();
  static constexpr DeviceIdxType root = 0;
  static constexpr int tensor_size = 1024;
  static constexpr int number_of_repetitions = 8;
  static constexpr c10d::ReduceOp::RedOpType red_op =
      c10d::ReduceOp::RedOpType::SUM;
  CommParams params;
  std::vector<DeviceIdxType> all_ranks;
};

class PipelineTest : public MultiDeviceTest {
 protected:
  void SetUp() override;
  // Run the multidevice fusion and stores the (sharded) results in "outputs"
  void execute();
  // Utility function used for validation in the tests. It compares the
  // (sharded) outputs with ref_unsharded_outputs. if
  // validate_with_prescribed_values is true, ref_unsharded_outputs is assumed
  // to be set manually in the test body. Otherwise, ref_unsharded_outputs is
  // computed by running a Fusion on a single device with the unsharded_inputs
  void validate(bool validate_with_prescribed_values = false);
  void executeAndValidate(bool validate_with_prescribed_values = false) {
    execute();
    validate(validate_with_prescribed_values);
  }
  std::unique_ptr<MultiDeviceExecutor> runtime;
  std::unique_ptr<Fusion> fusion;
  std::vector<c10::IValue> inputs;
  std::vector<c10::IValue> unsharded_inputs;
  std::vector<at::Tensor> outputs;
  std::vector<at::Tensor> ref_unsharded_outputs;
  MultiDeviceExecutorParams multi_device_executor_params;
};

} // namespace nvfuser

#endif
