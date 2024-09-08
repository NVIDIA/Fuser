// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <multidevice/communication.h>
#include <multidevice/communicator.h>
#include <multidevice/executor.h>
#include <multidevice/utils.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

// Used to clean up the Communicator singleton.
class MultiDeviceTestEnvironment : public testing::Environment {
 public:
  void TearDown() override;
};

class MultiDeviceTest : public NVFuserTest {
 protected:
  MultiDeviceTest();
  ~MultiDeviceTest();
  void SetUp() override;

  // Returns a shard of the tensor according to the sharding annotation in tv
  // for the deviceId. If tensor is not sharded returns the original tensor.
  // TODO: If deviceId is not part of the mesh this should return an empty
  // tensor. Currently, we don't support this, so for now it returns a slice.
  at::Tensor shardTensor(at::Tensor tensor, TensorView* tv);

  at::Tensor shardTensor(
      at::Tensor tensor,
      int64_t axis,
      const DeviceMesh& mesh);

  Communicator* communicator_;
  c10::TensorOptions tensor_options;
  bool debug_print;
  bool disable_skip;

 private:
  void waitForDebuggerAtRank(DeviceIdxType rank);
};

} // namespace nvfuser
