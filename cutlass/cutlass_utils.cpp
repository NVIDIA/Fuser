// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cutlass_utils.h>

namespace nvfuser::cutlass_kernels {

int getSMVersion() {
  int device{-1};
  CHECK_CUDA_SUCCESS(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(
      &sm_major, cudaDevAttrComputeCapabilityMajor, device));
  CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(
      &sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

int getMultiProcessorCount() {
  static int multi_processor_count = []() {
    int device_id = 0;
    int count = 0;

    // Get the current CUDA device ID
    CHECK_CUDA_SUCCESS(cudaGetDevice(&device_id));

    // Get the number of multiprocessors for the current device
    CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(
        &count, cudaDevAttrMultiProcessorCount, device_id));

    return count; // Initialize the static variable
  }();

  return multi_processor_count; // Return the cached value on subsequent calls
}

} // namespace nvfuser::cutlass_kernels
