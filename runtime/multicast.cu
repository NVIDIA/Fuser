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
    void* mc_dst,
    const void* src,
    size_t n_bytes) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  char* mc_dst_c = (char*)mc_dst;
  const char* src_c = (const char*)src;

  // Assume pointers are 16-byte aligned and size is multiple of 16
  size_t n_vec = n_bytes / 16;

  for (size_t i = idx; i < n_vec; i += stride) {
    int4 val = ((const int4*)src)[i];
    // Use multimem.st.global.v4.f32
    asm volatile("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"((void*)(mc_dst_c + i * 16)),
                   "f"(__int_as_float(val.x)),
                   "f"(__int_as_float(val.y)),
                   "f"(__int_as_float(val.z)),
                   "f"(__int_as_float(val.w))
                 : "memory");
  }
}
