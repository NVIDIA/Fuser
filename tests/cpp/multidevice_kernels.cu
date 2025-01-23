// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Warning: this file should not include any header from nvFuser or pytorch
// (except raw headers). Compiling dynamic_type.h with nvcc is not supported.
// Compiling pytorch with nvcc is not supported either.

#include <tests/cpp/multidevice_kernels.h>

namespace nvfuser {

__global__ void DummyMultiDeviceKernel() {}

void LaunchDummyMultiDeviceKernel() {
  DummyMultiDeviceKernel<<<1, 1>>>();
}

} // namespace nvfuser
