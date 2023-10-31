// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#pragma once

#include <multidevice/communication.h>
#include <multidevice/communicator.h>
#include <multidevice/pipeline.h>
#include <test/utils.h>

namespace nvfuser {

class MultiDeviceEnvironment : public testing::Environment {
 public:
  void SetUp() override;
  void TearDown() override;

  Communicator* communicator() const {
    NVF_ERROR(communicator_ != nullptr);
    return communicator_.get();
  }

 private:
  std::unique_ptr<Communicator> communicator_ = nullptr;
};

class MultiDeviceTest : public NVFuserTest {
 protected:
  void SetUp() override;
  Communicator* communicator;
  c10::TensorOptions tensor_options;
};

class CommunicationTest : public MultiDeviceTest {
 protected:
  void SetUp() override;
  void validate(at::Tensor obtained, at::Tensor expected);
  void resetDstBuffers();
  static constexpr DeviceIdxType root = 0;
  static constexpr int tensor_size = 1024;
  static constexpr int number_of_repetitions = 8;
  CommParams params;
  std::vector<DeviceIdxType> all_ranks;
};

class PipelineTest : public MultiDeviceTest {
 protected:
  void SetUp() override;
  void validate();
  bool print = false;
  std::unique_ptr<Pipeline> pipeline;
  std::unique_ptr<Fusion> fusion;
  std::vector<c10::IValue> inputs;
};

//(first stage's mesh, second stage's mesh, is first stage sharded, is second
// stage sharded)
using PipelineTestTwoStagesParams =
    std::tuple<DeviceMesh, DeviceMesh, bool, bool>;
class PipelineTestTwoStages
    : public PipelineTest,
      public ::testing::WithParamInterface<PipelineTestTwoStagesParams> {
 protected:
  void SetUp() override;
};

} // namespace nvfuser

#endif
