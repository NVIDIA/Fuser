// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>

namespace nvfuser {

using LowerCommunicationTest = MultiDeviceTest;

TEST_F(LowerCommunicationTest, AllGather) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const auto num_devices = communicator->size();
  TensorView* in = makeContigConcreteTensor({num_devices, 3});
  TensorView* out = set(in);
  fusion.addInput(in);
  fusion.addOutput(out);

  DeviceMesh mesh(num_devices);
  in->setDeviceMesh(mesh);
  out->setDeviceMesh(mesh);

  in->axis(0)->parallelize(ParallelType::DIDx);

  at::Tensor unsharded_in_tensor = at::randn({num_devices, 3}, tensor_options);
  at::Tensor in_tensor =
      shardTensor(unsharded_in_tensor, in, communicator->deviceId());

  FusionExecutor fe;
  fe.compileFusion(&fusion, {in_tensor});
  at::Tensor out_tensor = fe.runFusion({in_tensor})[0];
  EXPECT_TRUE(at::equal(out_tensor, unsharded_in_tensor));
}

} // namespace nvfuser
