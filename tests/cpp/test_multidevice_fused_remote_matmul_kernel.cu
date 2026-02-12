// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cuda_fp16.h>

#include "cuda_utils.h"

namespace nvfuser {

namespace {

// Naive fused kernel:
// - A is row-sharded across ranks (axis M)
// - each output row reads from its owner rank shard via remote pointers
// - B is replicated
__global__ void fusedRemoteMatmulKernel(
    const __half* const* a_remote_shards,
    const __half* b_full,
    __half* c_out,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t m_per_rank) {
  const int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }

  // Map global row to the rank-local row in that rank's A shard.
  const int64_t owner_rank = row / m_per_rank;
  const int64_t local_row = row - owner_rank * m_per_rank;
  const __half* a_local = a_remote_shards[owner_rank];

  float acc = 0.0f;
  for (int64_t kk = 0; kk < k; ++kk) {
    const float a = __half2float(a_local[local_row * k + kk]);
    const float b = __half2float(b_full[kk * n + col]);
    acc += a * b;
  }
  c_out[row * n + col] = __float2half(acc);
}

} // namespace

double timeFusedRemoteMatmulMs(
    const __half* const* a_remote_shards,
    const __half* b_full,
    __half* c_out,
    int64_t world_size,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t m_per_rank,
    int64_t warmup_iters,
    int64_t iters,
    cudaStream_t stream) {
  (void)world_size;
  const dim3 block(16, 16);
  const dim3 grid(
      static_cast<uint32_t>((n + block.x - 1) / block.x),
      static_cast<uint32_t>((m + block.y - 1) / block.y));

  auto launch_once = [&]() {
    fusedRemoteMatmulKernel<<<grid, block, 0, stream>>>(
        a_remote_shards, b_full, c_out, m, n, k, m_per_rank);
  };

  for (int64_t i = 0; i < warmup_iters; ++i) {
    launch_once();
  }
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetLastError());
  NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&start));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&stop));

  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(start, stream));
  for (int64_t i = 0; i < iters; ++i) {
    launch_once();
  }
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetLastError());
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(stop, stream));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventElapsedTime(&total_ms, start, stop));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(start));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(stop));
  return static_cast<double>(total_ms) / static_cast<double>(iters);
}

} // namespace nvfuser
