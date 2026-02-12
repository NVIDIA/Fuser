// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cuda_fp16.h>
#include <limits>

#include <ATen/core/Tensor.h>

#include "cuda_utils.h"
#include "runtime/matmul_tma.h"

namespace nvfuser {

namespace {

constexpr int64_t kSemaphoreVecWidth = 4;
constexpr int64_t kMaxSemaphorePollIters = 1LL << 26;

__device__ inline void publishEpochToAllRanks(
    int32_t* const* remote_semaphore_ptrs,
    int32_t* local_semaphore,
    int64_t writer_rank,
    int64_t row,
    int64_t m,
    int64_t world_size,
    int32_t epoch) {
  int32_t* my_row_local =
      local_semaphore + (writer_rank * m + row) * kSemaphoreVecWidth;
  for (int64_t vec_i = 0; vec_i < kSemaphoreVecWidth; ++vec_i) {
    my_row_local[vec_i] = epoch;
  }
  __threadfence_system();
  for (int64_t peer = 0; peer < world_size; ++peer) {
    int32_t* peer_row_remote =
        remote_semaphore_ptrs[peer] + (writer_rank * m + row) * kSemaphoreVecWidth;
    for (int64_t vec_i = 0; vec_i < kSemaphoreVecWidth; ++vec_i) {
      peer_row_remote[vec_i] = epoch;
    }
  }
  __threadfence_system();
}

__device__ inline void publishEpochToRank(
    int32_t* remote_semaphore_for_target_rank,
    int64_t writer_rank,
    int64_t row,
    int64_t m,
    int32_t epoch) {
  int32_t* row_remote =
      remote_semaphore_for_target_rank + (writer_rank * m + row) * kSemaphoreVecWidth;
  for (int64_t vec_i = 0; vec_i < kSemaphoreVecWidth; ++vec_i) {
    row_remote[vec_i] = epoch;
  }
  __threadfence_system();
}

__device__ inline void setLocalEpoch(
    int32_t* local_semaphore,
    int64_t writer_rank,
    int64_t row,
    int64_t m,
    int32_t epoch) {
  int32_t* row_local =
      local_semaphore + (writer_rank * m + row) * kSemaphoreVecWidth;
  for (int64_t vec_i = 0; vec_i < kSemaphoreVecWidth; ++vec_i) {
    row_local[vec_i] = epoch;
  }
  __threadfence_system();
}

__device__ inline void waitForEpochFromRank(
    int32_t* local_semaphore,
    int64_t row,
    int64_t m,
    int64_t writer_rank,
    int32_t epoch) {
  auto* rank_epoch_ptr = reinterpret_cast<unsigned int*>(
      local_semaphore + (writer_rank * m + row) * kSemaphoreVecWidth);
  int64_t spins = 0;
  while (atomicAdd(rank_epoch_ptr, 0U) < static_cast<unsigned int>(epoch)) {
    ++spins;
    if (spins > kMaxSemaphorePollIters) {
      asm volatile("trap;");
    }
  }
}

__device__ inline void waitForEpochFromAllRanks(
    int32_t* local_semaphore,
    int64_t row,
    int64_t m,
    int64_t world_size,
    int32_t epoch) {
  for (int64_t rank = 0; rank < world_size; ++rank) {
    auto* rank_epoch_ptr = reinterpret_cast<unsigned int*>(
        local_semaphore + (rank * m + row) * kSemaphoreVecWidth);
    int64_t spins = 0;
    while (atomicAdd(rank_epoch_ptr, 0U) < static_cast<unsigned int>(epoch)) {
      ++spins;
      if (spins > kMaxSemaphorePollIters) {
        asm volatile("trap;");
      }
    }
  }
}

template <typename LaunchFn>
double timeKernelLaunchesMs(
    int64_t warmup_iters,
    int64_t iters,
    cudaStream_t stream,
    LaunchFn&& launch_once) {
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

template <typename CommLaunchFn>
double timeCommThenCutlassMs(
    int64_t warmup_iters,
    int64_t iters,
    cudaStream_t stream,
    const at::Tensor& a_comm,
    const at::Tensor& b_full,
    at::Tensor& c_out,
    CommLaunchFn&& launch_comm_once) {
  NVF_CHECK(
      canRunMatmulTma(a_comm, b_full),
      "CUTLASS TMA compute requires Hopper+ and compatible half inputs.");
  auto launch_once = [&]() {
    launch_comm_once();
    // Rebind output tensor to TMA matmul result (avoids an extra device copy).
    c_out = matmulTma(a_comm, b_full);
  };
  return timeKernelLaunchesMs(warmup_iters, iters, stream, launch_once);
}

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

__global__ void fusedStagedThreadLoadSynchronizedKernel(
    const __half* const* a_remote_shards,
    __half* a_gathered,
    int32_t* const* ready_semaphore_remote_ptrs,
    int32_t* ready_semaphore_local,
    int32_t* const* done_semaphore_remote_ptrs,
    int32_t* done_semaphore_local,
    int64_t my_rank,
    int64_t world_size,
    int32_t launch_epoch_base,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t m_per_rank,
    const __half* b_full,
    __half* c_out) {
  const int32_t launch_epoch = launch_epoch_base + 1;
  for (int64_t row = blockIdx.x; row < m; row += gridDim.x) {
    const int64_t owner_rank = row / m_per_rank;
    const int64_t local_row = row - owner_rank * m_per_rank;
    const __half* a_local = a_remote_shards[owner_rank];
    if (threadIdx.x == 0) {
      // Owner publishes "ready"; non-owners wait only on owner readiness.
      if (my_rank == owner_rank) {
        publishEpochToAllRanks(
            ready_semaphore_remote_ptrs,
            ready_semaphore_local,
            my_rank,
            row,
            m,
            world_size,
            launch_epoch);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0 && my_rank != owner_rank) {
      waitForEpochFromRank(
          ready_semaphore_local, row, m, owner_rank, launch_epoch);
    }
    __syncthreads();

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

    if (threadIdx.x == 0) {
      // Readers ack completion only to owner; owner waits on all readers.
      if (my_rank == owner_rank) {
        setLocalEpoch(done_semaphore_local, my_rank, row, m, launch_epoch);
      } else {
        publishEpochToRank(
            done_semaphore_remote_ptrs[owner_rank],
            my_rank,
            row,
            m,
            launch_epoch);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0 && my_rank == owner_rank) {
      waitForEpochFromAllRanks(
          done_semaphore_local, row, m, world_size, launch_epoch);
    }
    __syncthreads();
  }
}

// Same fused structure as above, but stage-1 writes use multimem stores.
__global__ void fusedStagedMultimemKernel(
    const __half* const* a_remote_shards,
    __half* a_gathered_multicast,
    int32_t* const* stage_semaphore_remote_ptrs,
    int32_t* stage_semaphore_local,
    int64_t my_rank,
    int64_t world_size,
    int32_t launch_epoch_base,
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

    // Owner materializes the multicast row; non-owners wait on owner readiness.
    constexpr int64_t vec_elems = 8; // 8 * half = 16 bytes
    const int64_t n_vec = k / vec_elems;
    if (my_rank == owner_rank) {
      for (int64_t vec_i = threadIdx.x; vec_i < n_vec; vec_i += blockDim.x) {
        const uint4 val =
            reinterpret_cast<const uint4*>(a_local + local_row * k)[vec_i];
#if __CUDA_ARCH__ >= 900
        asm volatile("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                     :
                     : "l"((void*)(a_row_stage + vec_i * vec_elems)),
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
    }
    __syncthreads();

#if __CUDA_ARCH__ >= 900
    // Cross-device barrier between stage-1 stores and stage-2 reads.
    // Each rank publishes one epoch for (my_rank,row), then waits until
    // all writer ranks have published the same epoch for that row.
    const int32_t launch_epoch = launch_epoch_base + 1;

    if (threadIdx.x == 0) {
      if (my_rank == owner_rank) {
        publishEpochToAllRanks(
            stage_semaphore_remote_ptrs,
            stage_semaphore_local,
            my_rank,
            row,
            m,
            world_size,
            launch_epoch);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0 && my_rank != owner_rank) {
      // Non-owners only need owner readiness before reading multicast row.
      waitForEpochFromRank(
          stage_semaphore_local, row, m, owner_rank, launch_epoch);
    }
    __syncthreads();
#else
    (void)stage_semaphore_remote_ptrs;
    (void)stage_semaphore_local;
    (void)my_rank;
    (void)world_size;
    (void)launch_epoch_base;
    asm volatile("trap;");
#endif

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
  return timeKernelLaunchesMs(warmup_iters, iters, stream, launch_once);
}

double timeNaiveRemoteMatmulCutlassMs(
    const __half* const* a_remote_shards,
    at::Tensor& a_gathered,
    const at::Tensor& b_full,
    at::Tensor& c_out,
    int64_t m,
    int64_t k,
    int64_t m_per_rank,
    int64_t block_threads,
    int64_t grid_blocks,
    int64_t warmup_iters,
    int64_t iters,
    cudaStream_t stream) {
  const dim3 block(static_cast<uint32_t>(block_threads));
  const dim3 grid(static_cast<uint32_t>(grid_blocks <= 0 ? m : grid_blocks));
  const __half* b_ptr =
      reinterpret_cast<const __half*>(b_full.data_ptr<at::Half>());
  __half* c_ptr = reinterpret_cast<__half*>(c_out.data_ptr<at::Half>());
  __half* a_gathered_ptr =
      reinterpret_cast<__half*>(a_gathered.data_ptr<at::Half>());
  auto launch_comm_once = [&]() {
    // Keep communication as remote thread-load gather.
    fusedStagedThreadLoadKernel<<<grid, block, 0, stream>>>(
        a_remote_shards,
        a_gathered_ptr,
        m,
        /*n=*/0,
        k,
        m_per_rank,
        b_ptr,
        c_ptr);
  };
  return timeCommThenCutlassMs(
      warmup_iters,
      iters,
      stream,
      a_gathered,
      b_full,
      c_out,
      launch_comm_once);
}

double timeSeparatedAllgatherMatmulThreadLoadSynchronizedMs(
    const __half* const* a_remote_shards,
    const __half* b_full,
    __half* c_out,
    __half* a_gathered,
    int32_t* const* ready_semaphore_remote_ptrs,
    int32_t* ready_semaphore_local,
    int32_t* const* done_semaphore_remote_ptrs,
    int32_t* done_semaphore_local,
    int64_t my_rank,
    int64_t world_size,
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
  int64_t launch_epoch_base = 0;

  auto launch_once = [&]() {
    NVF_CHECK(
        launch_epoch_base < std::numeric_limits<int32_t>::max(),
        "ThreadLoad synchronized semaphore epoch overflow.");
    fusedStagedThreadLoadSynchronizedKernel<<<grid, block, 0, stream>>>(
        a_remote_shards,
        a_gathered,
        ready_semaphore_remote_ptrs,
        ready_semaphore_local,
        done_semaphore_remote_ptrs,
        done_semaphore_local,
        my_rank,
        world_size,
        static_cast<int32_t>(launch_epoch_base),
        m,
        n,
        k,
        m_per_rank,
        b_full,
        c_out);
    ++launch_epoch_base;
  };
  return timeKernelLaunchesMs(warmup_iters, iters, stream, launch_once);
}

double timeSeparatedAllgatherMatmulThreadLoadSynchronizedCutlassMs(
    const __half* const* a_remote_shards,
    at::Tensor& a_gathered,
    int32_t* const* ready_semaphore_remote_ptrs,
    int32_t* ready_semaphore_local,
    int32_t* const* done_semaphore_remote_ptrs,
    int32_t* done_semaphore_local,
    int64_t my_rank,
    int64_t world_size,
    const at::Tensor& b_full,
    at::Tensor& c_out,
    int64_t m,
    int64_t k,
    int64_t m_per_rank,
    int64_t block_threads,
    int64_t grid_blocks,
    int64_t warmup_iters,
    int64_t iters,
    cudaStream_t stream) {
  const dim3 block(static_cast<uint32_t>(block_threads));
  const dim3 grid(static_cast<uint32_t>(grid_blocks <= 0 ? m : grid_blocks));
  int64_t launch_epoch_base = 0;
  const __half* b_ptr =
      reinterpret_cast<const __half*>(b_full.data_ptr<at::Half>());
  __half* c_ptr = reinterpret_cast<__half*>(c_out.data_ptr<at::Half>());
  __half* a_gathered_ptr =
      reinterpret_cast<__half*>(a_gathered.data_ptr<at::Half>());
  auto launch_comm_once = [&]() {
    NVF_CHECK(
        launch_epoch_base < std::numeric_limits<int32_t>::max(),
        "ThreadLoad synchronized CUTLASS semaphore epoch overflow.");
    fusedStagedThreadLoadSynchronizedKernel<<<grid, block, 0, stream>>>(
        a_remote_shards,
        a_gathered_ptr,
        ready_semaphore_remote_ptrs,
        ready_semaphore_local,
        done_semaphore_remote_ptrs,
        done_semaphore_local,
        my_rank,
        world_size,
        static_cast<int32_t>(launch_epoch_base),
        m,
        /*n=*/0,
        k,
        m_per_rank,
        b_ptr,
        c_ptr);
    ++launch_epoch_base;
  };
  return timeCommThenCutlassMs(
      warmup_iters,
      iters,
      stream,
      a_gathered,
      b_full,
      c_out,
      launch_comm_once);
}

double timeSeparatedAllgatherMatmulMultimemMs(
    const __half* const* a_remote_shards,
    const __half* b_full,
    __half* c_out,
    __half* a_gathered_multicast,
    int32_t* const* stage_semaphore_remote_ptrs,
    int32_t* stage_semaphore_local,
    int64_t my_rank,
    int64_t world_size,
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
  int64_t launch_epoch_base = 0;

  auto launch_once = [&]() {
    NVF_CHECK(
        launch_epoch_base < std::numeric_limits<int32_t>::max(),
        "Multimem semaphore epoch overflow.");
    fusedStagedMultimemKernel<<<grid, block, 0, stream>>>(
        a_remote_shards,
        a_gathered_multicast,
        stage_semaphore_remote_ptrs,
        stage_semaphore_local,
        my_rank,
        world_size,
        static_cast<int32_t>(launch_epoch_base),
        m,
        n,
        k,
        m_per_rank,
        b_full,
        c_out);
    ++launch_epoch_base;
  };
  return timeKernelLaunchesMs(warmup_iters, iters, stream, launch_once);
}

double timeSeparatedAllgatherMatmulMultimemCutlassMs(
    const __half* const* a_remote_shards,
    __half* a_gathered_multicast_ptr,
    const at::Tensor& a_gathered_local,
    int32_t* const* stage_semaphore_remote_ptrs,
    int32_t* stage_semaphore_local,
    int64_t my_rank,
    int64_t world_size,
    const at::Tensor& b_full,
    at::Tensor& c_out,
    int64_t m,
    int64_t k,
    int64_t m_per_rank,
    int64_t block_threads,
    int64_t grid_blocks,
    int64_t warmup_iters,
    int64_t iters,
    cudaStream_t stream) {
  const dim3 block(static_cast<uint32_t>(block_threads));
  const dim3 grid(static_cast<uint32_t>(grid_blocks <= 0 ? m : grid_blocks));
  int64_t launch_epoch_base = 0;
  const __half* b_ptr =
      reinterpret_cast<const __half*>(b_full.data_ptr<at::Half>());
  __half* c_ptr = reinterpret_cast<__half*>(c_out.data_ptr<at::Half>());
  auto launch_comm_once = [&]() {
    NVF_CHECK(
        launch_epoch_base < std::numeric_limits<int32_t>::max(),
        "Multimem CUTLASS semaphore epoch overflow.");
    fusedStagedMultimemKernel<<<grid, block, 0, stream>>>(
        a_remote_shards,
        a_gathered_multicast_ptr,
        stage_semaphore_remote_ptrs,
        stage_semaphore_local,
        my_rank,
        world_size,
        static_cast<int32_t>(launch_epoch_base),
        m,
        /*n=*/0,
        k,
        m_per_rank,
        b_ptr,
        c_ptr);
    ++launch_epoch_base;
  };
  return timeCommThenCutlassMs(
      warmup_iters,
      iters,
      stream,
      a_gathered_local,
      b_full,
      c_out,
      launch_comm_once);
}

} // namespace nvfuser
