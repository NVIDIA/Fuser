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

  // A lower-level helper that doesn't require a TensorView.
  at::Tensor shardTensor(
      at::Tensor tensor,
      int64_t axis,
      const DeviceMesh& mesh);

  // Validate the outputs of a fusion against expected outputs.
  static void validate(
      const std::vector<at::Tensor>& expected_outputs,
      const KernelArgumentHolder& outputs,
      const std::vector<double>& atols);

  Communicator* communicator_;
  c10::TensorOptions tensor_options;
  bool debug_print;
  bool disable_skip;
};

// This macro is supposed to be used in a test case of a MultiDeviceTest or its
// `SetUp` method, which have access to GTEST_SKIP and communicator_. It's not
// made a function because that function wouldn't be able to skip the test by
// calling GTEST_SKIP.
#define SKIP_IF_NOT_ENOUGH_DEVICES(fusion)                          \
  do {                                                              \
    const auto num_devices = communicator_->size();                 \
    for (auto* tv : fusion->allTvs()) {                             \
      const DeviceMesh& mesh = tv->getDeviceMesh();                 \
      for (const auto device_id : mesh.vector()) {                  \
        if (device_id >= num_devices) {                             \
          GTEST_SKIP() << tv->toString() << ") requires more than " \
                       << num_devices << " devices.";               \
        }                                                           \
      }                                                             \
    }                                                               \
  } while (0)

} // namespace nvfuser
