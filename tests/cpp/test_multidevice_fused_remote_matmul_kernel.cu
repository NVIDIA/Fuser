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

// Fused kernel with explicit internal stages:
// 1) Allgather stage: materialize one full A row from remote shard.
// 2) Compute stage: matmul for that row across all output columns.
__global__ void fusedStagedThreadLoadKernel(
    const __half* const* a_remote_shards,
    __half* a_gathered,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t m_per_rank,
    const __half* b_full,
    __half* c_out) {
  for (int64_t row = blockIdx.x; row < m; row += gridDim.x) {
    const int64_t owner_rank = row / m_per_rank;
    const int64_t local_row = row - owner_rank * m_per_rank;
    const __half* a_local = a_remote_shards[owner_rank];

    // Stage 1: gather this row into staged global buffer.
    for (int64_t kk = threadIdx.x; kk < k; kk += blockDim.x) {
      a_gathered[row * k + kk] = a_local[local_row * k + kk];
    }
    __syncthreads();

    // Stage 2: compute this row from staged global A.
    for (int64_t col = threadIdx.x; col < n; col += blockDim.x) {
      float acc = 0.0f;
      for (int64_t kk = 0; kk < k; ++kk) {
        acc += __half2float(a_gathered[row * k + kk]) *
            __half2float(b_full[kk * n + col]);
      }
      c_out[row * n + col] = __float2half(acc);
    }
    __syncthreads();
  }
}

// Same fused structure as above, but stage-1 writes use multimem stores.
__global__ void fusedStagedMultimemKernel(
    const __half* const* a_remote_shards,
    __half* a_gathered_multicast,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t m_per_rank,
    const __half* b_full,
    __half* c_out) {
  for (int64_t row = blockIdx.x; row < m; row += gridDim.x) {
    const int64_t owner_rank = row / m_per_rank;
    const int64_t local_row = row - owner_rank * m_per_rank;
    const __half* a_local = a_remote_shards[owner_rank];
    __half* a_row_stage = a_gathered_multicast + row * k;

    // Stage 1: gather row into multicast staging buffer via 16-byte vectors.
    constexpr int64_t vec_elems = 8; // 8 * half = 16 bytes
    const int64_t n_vec = k / vec_elems;
    for (int64_t vec_i = threadIdx.x; vec_i < n_vec; vec_i += blockDim.x) {
      const uint4 val =
          reinterpret_cast<const uint4*>(a_local + local_row * k)[vec_i];
      char* dst_byte = reinterpret_cast<char*>(a_row_stage) + vec_i * 16;

#if __CUDA_ARCH__ >= 900
      asm volatile("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                   :
                   : "l"((void*)dst_byte),
                     "f"(__int_as_float(static_cast<int>(val.x))),
                     "f"(__int_as_float(static_cast<int>(val.y))),
                     "f"(__int_as_float(static_cast<int>(val.z))),
                     "f"(__int_as_float(static_cast<int>(val.w)))
                   : "memory");
#else
      (void)val;
      // Multimem path must never run on non-Hopper architectures.
      asm volatile("trap;");
#endif
    }
    for (int64_t kk = n_vec * vec_elems + threadIdx.x; kk < k;
         kk += blockDim.x) {
      a_row_stage[kk] = a_local[local_row * k + kk];
    }
    __syncthreads();

    // Stage 2: compute from staged multicast-backed row.
    for (int64_t col = threadIdx.x; col < n; col += blockDim.x) {
      float acc = 0.0f;
      for (int64_t kk = 0; kk < k; ++kk) {
        acc +=
            __half2float(a_row_stage[kk]) * __half2float(b_full[kk * n + col]);
      }
      c_out[row * n + col] = __float2half(acc);
    }
    __syncthreads();
  }
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
    int64_t block_x,
    int64_t block_y,
    int64_t warmup_iters,
    int64_t iters,
    cudaStream_t stream) {
  (void)world_size;
  const dim3 block(static_cast<uint32_t>(block_x), static_cast<uint32_t>(block_y));
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

double timeSeparatedAllgatherMatmulThreadLoadMs(
    const __half* const* a_remote_shards,
    const __half* b_full,
    __half* c_out,
    __half* a_gathered,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t m_per_rank,
    int64_t block_threads,
    int64_t grid_blocks,
    int64_t warmup_iters,
    int64_t iters,
    cudaStream_t stream) {
  const dim3 block(static_cast<uint32_t>(block_threads));
  const dim3 grid(static_cast<uint32_t>(grid_blocks <= 0 ? m : grid_blocks));

  auto launch_once = [&]() {
    fusedStagedThreadLoadKernel<<<grid, block, 0, stream>>>(
        a_remote_shards, a_gathered, m, n, k, m_per_rank, b_full, c_out);
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

double timeSeparatedAllgatherMatmulMultimemMs(
    const __half* const* a_remote_shards,
    const __half* b_full,
    __half* c_out,
    __half* a_gathered_multicast,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t m_per_rank,
    int64_t block_threads,
    int64_t grid_blocks,
    int64_t warmup_iters,
    int64_t iters,
    cudaStream_t stream) {
  const dim3 block(static_cast<uint32_t>(block_threads));
  const dim3 grid(static_cast<uint32_t>(grid_blocks <= 0 ? m : grid_blocks));

  auto launch_once = [&]() {
    fusedStagedMultimemKernel<<<grid, block, 0, stream>>>(
        a_remote_shards,
        a_gathered_multicast,
        m,
        n,
        k,
        m_per_rank,
        b_full,
        c_out);
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
