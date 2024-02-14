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
#include <test/utils.h>
#include <chrono>

namespace nvfuser {

class MultiDeviceEnvironment : public testing::Environment {
 public:
  void SetUp() override;

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

  bool timePrint() const {
    return time_print_;
  }

  bool disableSkip() const {
    return disable_skip_;
  }

 private:
  std::unique_ptr<Communicator> communicator_ = nullptr;
  bool debug_print_ = false;
  bool do_barrier_at_test_ = false;
  bool time_print_ = false;
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
  void printTimes();
  void recordEvent(std::string);
  Communicator* communicator;
  c10::TensorOptions tensor_options;
  bool debug_print;
  bool do_barrier_at_test;
  bool time_print;
  bool disable_skip;
  std::vector<std::pair<
      const std::string,
      std::chrono::time_point<std::chrono::high_resolution_clock>>>
      times;
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
  void validate();
  void execute();
  void executeAndValidate() {
    execute();
    validate();
  }
  std::unique_ptr<MultiDeviceExecutor> runtime;
  std::unique_ptr<Fusion> fusion;
  std::vector<c10::IValue> inputs;
  std::vector<c10::IValue> unsharded_inputs;
  std::vector<at::Tensor> outputs;
};

} // namespace nvfuser

#endif
