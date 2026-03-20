// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Kernel for reduce & allreduce using multimem ld_reduce (NVLink SHARP).
// Loads from the multicast address (aggregating all ranks) with reduction.
// Requires SM90+ (Hopper) and PTX ISA 8.0+.
// Compiled by NVRTC at runtime.

extern "C" __global__ void multimem_ld_reduce_sum_f32_kernel(
    const void* mc_src,
    void* dst,
    size_t n_bytes) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  const char* mc_src_c = (const char*)mc_src;
  char* dst_c = (char*)dst;

  // 16-byte (4xf32) aligned; multimem.ld.reduce.global.v4.f32.sum
  size_t n_vec = n_bytes / 16;

  for (size_t i = idx; i < n_vec; i += stride) {
    float r0, r1, r2, r3;
    const void* addr = mc_src_c + i * 16;
    asm volatile(
        "multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3)
        : "l"(addr)
        : "memory");
    float4 out;
    out.x = r0;
    out.y = r1;
    out.z = r2;
    out.w = r3;
    ((float4*)dst_c)[i] = out;
  }
}
