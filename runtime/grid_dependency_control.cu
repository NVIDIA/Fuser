// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

namespace pdl {

// Issuing the launch_dependents instruction hints a dependent kernel to launch
// before the primary kernel is finished. This is a performance optimization.
// Launching a dependent kernel too early means it can compete with current
// kernels for device resources, while launching too late can lead to a long
// latency.
__device__ inline void launchDependentGrid() {
  asm volatile("griddepcontrol.launch_dependents;");
}

// Issuing the wait instruction enforces no global memory access prior to this
// instruction. This ensures the correctness of global memory access when
// launching a dependent kernel.
__device__ inline void waitForPriorGrid() {
  asm volatile("griddepcontrol.wait;");
}

} // namespace pdl

#endif // (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
