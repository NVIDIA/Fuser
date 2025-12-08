// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Kernel for multicast copy using multimem instructions
// This is compiled by NVRTC at runtime

extern "C" __global__ void multimem_copy_kernel(
    void* dst,
    const void* src,
    size_t n_bytes) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  char* dst_c = (char*)dst;
  const char* src_c = (const char*)src;

  // Vectorized copy (16 bytes) if pointers are 16-byte aligned
  bool is_aligned = ((size_t)src % 16 == 0) && ((size_t)dst % 16 == 0);
  size_t n_vec = is_aligned ? (n_bytes / 16) : 0;

  for (size_t i = idx; i < n_vec; i += stride) {
    int4 val = ((const int4*)src)[i];
    // Use multimem.st.global.b32
    asm volatile(
        "multimem.st.global.b32 [%0], %1;\n"
        "multimem.st.global.b32 [%0+4], %2;\n"
        "multimem.st.global.b32 [%0+8], %3;\n"
        "multimem.st.global.b32 [%0+12], %4;"
        :
        : "l"((void*)(dst_c + i * 16)),
          "r"(val.x),
          "r"(val.y),
          "r"(val.z),
          "r"(val.w)
        : "memory");
  }

  // Handle tail with standard stores (st.global works on multicast addresses
  // too)
  size_t processed = n_vec * 16;
  for (size_t i = processed + idx; i < n_bytes; i += stride) {
    dst_c[i] = src_c[i];
  }
}
