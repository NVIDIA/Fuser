// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <cuda_profiler_api.h>
#include <fusion.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/multidevice_kernels.h>

namespace nvfuser {

TEST_F(MultiDeviceTest, DummyMultiDeviceKernelTest) {
  LaunchDummyMultiDeviceKernel();
}

} // namespace nvfuser
