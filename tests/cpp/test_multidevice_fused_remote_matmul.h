// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <ATen/core/Tensor.h>

namespace c10d {
class Backend;
}

namespace nvfuser {

class Communicator;
class Communication;
class SymMemForAllgather;

// =========================================================================
// Distributed Matmul Benchmark -- shared types
//
// Computes C[M,N] = A[M,K] x B[K,N] where A is row-sharded across ranks
// on axis M and B is replicated.  Each implementation varies the
// communication strategy (how A shards are gathered) and the compute
// strategy (how the matmul is performed).
//
// Performance on 8xH100 DGX (M=N=K=1024, half precision):
//
//   Implementation                         | TFLOP/s
//   ---------------------------------------+--------
//   baselineNcclAllgatherMatmul            |  39.9
//   baselineCudaAllgatherMatmul            |  24.1
//   naiveRemoteRead                        |   2.67
//   threadloadGatherScalarCompute          |   4.05
//   multimemGatherScalarCompute            |   3.69
//   threadloadGatherThenCutlass            |  50.3
//   multimemGatherThenCutlass              |  50.1
// =========================================================================

enum class DistributedMatmulImpl {
  // -- Baselines: separate allgather then PyTorch eager matmul --
  baselineNcclAllgatherMatmul,
  baselineCudaAllgatherMatmul,
  // -- Fused kernels: comm + scalar matmul in one kernel --
  naiveRemoteRead,
  threadloadGatherScalarCompute,
  multimemGatherScalarCompute,
  // -- Two-kernel: separate comm kernel then CUTLASS TMA matmul
  //    (NOT truly fused -- two kernel launches on the same stream) --
  threadloadGatherThenCutlass,
  multimemGatherThenCutlass,
};

enum class TimeMeasurementMode { CudaEvents, CpuClock };

struct BenchmarkConfig {
  int64_t warmup_iters;
  int64_t iters;
  TimeMeasurementMode time_mode;
  bool barrier_at_each_iteration;
};

// All data any implementation may need.  Unused fields are
// null/undefined for a given implementation.
struct DistributedMatmulContext {
  // Problem dimensions
  int64_t m = 0, n = 0, k = 0, m_per_rank = 0;
  int64_t my_rank = 0, world_size = 0;

  // Remote A shard pointers (device array of const __half*)
  const __half* const* device_remote_ptrs = nullptr;

  // Input / output tensors
  at::Tensor a_local_half;   // [m_per_rank, k]
  at::Tensor b_full_half;    // [k, n]
  at::Tensor c_out_half;     // [m, n]

  // Staging buffers for gather-then-compute paths
  at::Tensor a_gathered;           // [m, k] threadload staging
  at::Tensor a_gathered_multimem;  // [m, k] multicast-backed
  __half* multicast_ptr = nullptr;

  // Threadload semaphores (ready / done handshake)
  int32_t* const* ready_sem_remote = nullptr;
  int32_t* ready_sem_local = nullptr;
  int32_t* const* done_sem_remote = nullptr;
  int32_t* done_sem_local = nullptr;

  // Multimem semaphores (stage barrier)
  int32_t* const* stage_sem_remote = nullptr;
  int32_t* stage_sem_local = nullptr;

  // Baseline-only resources
  c10d::Backend* nccl_backend = nullptr;
  Communication* cuda_comm = nullptr;
  SymMemForAllgather* cuda_handle = nullptr;
  at::Tensor a_allgathered_cuda;

  // Runtime
  Communicator* communicator = nullptr;
  cudaStream_t stream = nullptr;
};

// --- Defined in .cu (kernel launchers, CUTLASS wrapper) ---
void launchNaiveRemoteRead(DistributedMatmulContext& ctx);
void launchThreadloadGather(
    DistributedMatmulContext& ctx,
    int32_t epoch,
    bool compute);
void launchMultimemGather(
    DistributedMatmulContext& ctx,
    int32_t epoch,
    bool compute);
void matmulTma(
    at::Tensor& out,
    const at::Tensor& a,
    const at::Tensor& b);
bool canRunCutlassCompute(
    const at::Tensor& a,
    const at::Tensor& b);
const char* implName(DistributedMatmulImpl impl);
bool isMulticastSupported(int64_t device_id);

} // namespace nvfuser
