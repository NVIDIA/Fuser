// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh
__device__ __inline__ float fast_tanhf(float x) {
#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)
  float y;
  asm volatile("tanh.approx.f32 %0, %1; " : "=f"(y) : "f"(x));
  return y;
#else
  return tanhf(x);
#endif
}
