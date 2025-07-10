// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cuda_utils.h>
#include <cutlass_utils.h>

namespace nvfuser::cutlass_kernels {

int getSMVersion() {
  int device{-1};
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaDeviceGetAttribute(
      &sm_major, cudaDevAttrComputeCapabilityMajor, device));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaDeviceGetAttribute(
      &sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

int getMultiProcessorCount() {
  static int multi_processor_count = []() {
    int device_id = 0;
    int count = 0;

    // Get the current CUDA device ID
    NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDevice(&device_id));

    // Get the number of multiprocessors for the current device
    NVFUSER_CUDA_RT_SAFE_CALL(cudaDeviceGetAttribute(
        &count, cudaDevAttrMultiProcessorCount, device_id));

    return count; // Initialize the static variable
  }();

  return multi_processor_count; // Return the cached value on subsequent calls
}

} // namespace nvfuser::cutlass_kernels
