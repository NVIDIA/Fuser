// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cuda_fp16.h>

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <chrono>

#include "fusion.h"
#include "host_ir/container.h"
#include "ir/builder.h"
#include "multidevice/communication.h"
#include "multidevice/communicator.h"
#include "multidevice/cuda_p2p.h"
#include "multidevice/ipc_handle.h"
#include "multidevice/symmetric_tensor.h"
#include "runtime/matmul_tma.h"
#include "tests/cpp/multidevice.h"

namespace nvfuser {

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
    cudaStream_t stream);

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
    cudaStream_t stream);

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
    cudaStream_t stream);

double timeSeparatedAllgatherMatmulThreadLoadCutlassMs(
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
    cudaStream_t stream);

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
    cudaStream_t stream);

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
    cudaStream_t stream);

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
    cudaStream_t stream);

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
    cudaStream_t stream);

namespace {

// Implementations compared by this benchmark:
// - naiveFusedKernel:
//   A rank-local, handwritten remote-pointer matmul path. A is sharded on M and
//   each output row pulls its owner shard directly from symmetric remote memory.
//   This models a fused comm+compute style where data movement is embedded in
//   the kernel access pattern.
// - baselinePytorchEagerNccl:
//   Reference eager path using PyTorch process group NCCL allgather to rebuild
//   full A on every rank, then regular eager matmul_out(A_full, B_full).
// - baselinePytorchEagerCuda:
//   Same eager compute structure as the NCCL baseline, but communication uses
//   nvFuser's CUDA backend allgather primitives (post/wait with symmetric-memory
//   handles from cuda_p2p.h) before eager matmul_out.
// - gpuAllgatherFusedNaiveCompute:
//   Single fused kernel with explicit internal stages:
//   (1) stage full A-row from remote pointers with regular thread loads,
//   (2) compute matmul for that row. Uses fused ready/done semaphores.
// - gpuAllgatherMultimemFusedNaiveCompute:
//   Same fused-kernel staged structure, but stage-1 writes use multimem store
//   instructions on a multicast pointer before compute, with fused semaphores.
// - *CutlassCompute variants:
//   Keep communication path semantics and replace compute with Hopper CUTLASS
//   TMA matmul.
enum class DistributedMatmulImpl {
  naiveFusedKernel,
  naiveFusedKernelCutlassCompute,
  baselinePytorchEagerNccl,
  baselinePytorchEagerCuda,
  gpuAllgatherFusedNaiveCompute,
  gpuAllgatherTmaCompute,
  gpuAllgatherMultimemTmaCompute,
  gpuAllgatherMultimemFusedNaiveCompute
};

enum class TimeMeasurementMode { CudaEvents, CpuClock };

// Runtime kernel-launch knobs shared by all implementations.
enum class RuntimeParams {
  NaiveBlockX,
  NaiveBlockY,
  StagedBlockThreads,
  StagedGridBlocks
};

int64_t runtimeParam(RuntimeParams param) {
  switch (param) {
    case RuntimeParams::NaiveBlockX:
      return 16;
    case RuntimeParams::NaiveBlockY:
      return 16;
    case RuntimeParams::StagedBlockThreads:
      return 256;
    case RuntimeParams::StagedGridBlocks:
      // <= 0 means auto-select from M in the kernel launcher.
      return 0;
  }
  NVF_ERROR(false, "Unknown runtime parameter enum value.");
  return 0;
}

// Centralized benchmark knobs used by every implementation path.
struct BenchmarkConfig {
  int64_t warmup_iters;
  int64_t iters;
  TimeMeasurementMode time_mode;
  bool barrier_at_each_iteration;
};

// Optional runtime objects required by specific implementations.
struct BenchmarkResources {
  c10d::Backend* nccl_backend = nullptr;
  std::unique_ptr<hir::HostIrContainer> cuda_hic;
  Communication* cuda_allgather_communication = nullptr;
  std::unique_ptr<SymMemForAllgather> cuda_allgather_handle;
  at::Tensor a_allgathered_half_cuda;
  at::Tensor a_gathered_threadload;
  at::Tensor a_gathered_multimem;
  std::unique_ptr<SymmetricTensor> a_gathered_multimem_sym;
  at::Tensor stage_semaphore_multimem;
  std::unique_ptr<SymmetricTensor> stage_semaphore_multimem_sym;
  at::Tensor threadload_ready_semaphore;
  std::unique_ptr<SymmetricTensor> threadload_ready_semaphore_sym;
  at::Tensor threadload_done_semaphore;
  std::unique_ptr<SymmetricTensor> threadload_done_semaphore_sym;
};

const char* implName(DistributedMatmulImpl impl) {
  switch (impl) {
    case DistributedMatmulImpl::naiveFusedKernel:
      return "naiveFusedKernel";
    case DistributedMatmulImpl::naiveFusedKernelCutlassCompute:
      return "naiveFusedKernelCutlassCompute";
    case DistributedMatmulImpl::baselinePytorchEagerNccl:
      return "baselinePytorchEagerNccl";
    case DistributedMatmulImpl::baselinePytorchEagerCuda:
      return "baselinePytorchEagerCuda";
    case DistributedMatmulImpl::gpuAllgatherFusedNaiveCompute:
      return "gpuAllgatherFusedNaiveCompute";
    case DistributedMatmulImpl::gpuAllgatherTmaCompute:
      return "gpuAllgatherTmaCompute";
    case DistributedMatmulImpl::gpuAllgatherMultimemTmaCompute:
      return "gpuAllgatherMultimemTmaCompute";
    case DistributedMatmulImpl::gpuAllgatherMultimemFusedNaiveCompute:
      return "gpuAllgatherMultimemFusedNaiveCompute";
  }
  NVF_ERROR(false, "Unknown implementation enum value: ", static_cast<int>(impl));
}

bool isMulticastSupported(int64_t device_id) {
  int is_multicast_supported = 0;
  auto result = cuDeviceGetAttribute(
      &is_multicast_supported,
      CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
      static_cast<CUdevice>(device_id));
  return result == CUDA_SUCCESS && is_multicast_supported != 0;
}

bool canRunHopperCutlassCompute(const at::Tensor& a, const at::Tensor& b) {
  if (!canRunMatmulTma(a, b)) {
    return false;
  }
  auto* props = at::cuda::getDeviceProperties(a.get_device());
  // Restrict CUTLASS-compute benchmark variants to Hopper in this test.
  return props->major == 9 && props->minor == 0;
}

template <typename Fn>
double benchmarkLoopMs(
    const BenchmarkConfig& config,
    Communicator* communicator,
    cudaStream_t stream,
    Fn&& run_once) {
  NVF_CHECK(config.iters > 0, "iters must be > 0, got ", config.iters);

  // Warmup segment (not timed).
  for (int64_t i = 0; i < config.warmup_iters; ++i) {
    if (config.barrier_at_each_iteration) {
      communicator->barrier();
    }
    run_once();
  }
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetLastError());
  NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(stream));

  // Timed segment with device-side timestamps.
  if (config.time_mode == TimeMeasurementMode::CudaEvents) {
    // Time each iteration independently so optional barriers can remain outside
    // the measured region while preserving per-iteration MAX reduction semantics.
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&start));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&stop));
    float total_ms = 0.0f;
    for (int64_t i = 0; i < config.iters; ++i) {
      if (config.barrier_at_each_iteration) {
        communicator->barrier();
      }
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(start, stream));
      run_once();
      NVFUSER_CUDA_RT_SAFE_CALL(cudaGetLastError());
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(stop, stream));
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(stop));
      float iter_ms = 0.0f;
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventElapsedTime(&iter_ms, start, stop));
      total_ms += iter_ms;
    }
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(start));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(stop));
    return static_cast<double>(total_ms) / static_cast<double>(config.iters);
  }

  // Timed segment with host-side timestamps (includes stream sync cost).
  double total_ms = 0.0;
  for (int64_t i = 0; i < config.iters; ++i) {
    if (config.barrier_at_each_iteration) {
      communicator->barrier();
    }
    NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(stream));
    auto start = std::chrono::high_resolution_clock::now();
    run_once();
    NVFUSER_CUDA_RT_SAFE_CALL(cudaGetLastError());
    NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(stream));
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;
    total_ms += elapsed.count();
  }
  return total_ms / static_cast<double>(config.iters);
}

BenchmarkResources initBenchmarkResources(
    DistributedMatmulImpl impl,
    Communicator* communicator,
    const Team& all_devices,
    int64_t world_size,
    int64_t m,
    int64_t k) {
  BenchmarkResources resources;
  // NCCL eager baseline resources.
  if (impl == DistributedMatmulImpl::baselinePytorchEagerNccl) {
    if (!communicator->isBackendAvailable(CommunicatorBackend::kNccl)) {
      return resources;
    }
    resources.nccl_backend =
        communicator->getBackendForTeam(all_devices, CommunicatorBackend::kNccl);
  }

  // CUDA backend eager baseline resources (symmetric allgather handle).
  if (impl == DistributedMatmulImpl::baselinePytorchEagerCuda) {
    resources.cuda_hic = std::make_unique<hir::HostIrContainer>();
    FusionGuard fg(resources.cuda_hic.get());
    auto* in_tv = makeContigTensor(2);
    auto* out_tv = makeContigTensor(2);
    DeviceMesh mesh = DeviceMesh::createForNumDevices(world_size);
    in_tv->setDeviceMesh(mesh);
    out_tv->setDeviceMesh(mesh);
    resources.cuda_allgather_communication = IrBuilder::create<Communication>(
        CommunicationType::Allgather,
        out_tv,
        in_tv,
        all_devices,
        /*root=*/-1,
        RedOpType::UNUSED,
        CommunicatorBackend::kCuda);
    resources.a_allgathered_half_cuda = SymmetricTensor::allocate(
        {m, k}, at::ScalarType::Half, communicator->device());
    resources.cuda_allgather_handle = std::make_unique<SymMemForAllgather>(
        resources.cuda_allgather_communication, resources.a_allgathered_half_cuda);
  }

  if (impl == DistributedMatmulImpl::gpuAllgatherFusedNaiveCompute ||
      impl == DistributedMatmulImpl::naiveFusedKernelCutlassCompute ||
      impl == DistributedMatmulImpl::gpuAllgatherTmaCompute) {
    resources.a_gathered_threadload = at::empty(
        {m, k},
        at::TensorOptions()
            .dtype(at::kHalf)
            .device(communicator->device())
            .layout(at::kStrided));
  }

  if (impl == DistributedMatmulImpl::gpuAllgatherFusedNaiveCompute ||
      impl == DistributedMatmulImpl::gpuAllgatherTmaCompute) {
    // Per-rank [writer_rank, row, vec4-int] semaphores for fused ready/done.
    resources.threadload_ready_semaphore = SymmetricTensor::allocate(
        {world_size, m, 4}, at::ScalarType::Int, communicator->device());
    resources.threadload_ready_semaphore.zero_();
    resources.threadload_ready_semaphore_sym = std::make_unique<SymmetricTensor>(
        resources.threadload_ready_semaphore);
    resources.threadload_ready_semaphore_sym->setupRemoteHandles(
        "fused_remote_matmul_threadload_ready");

    resources.threadload_done_semaphore = SymmetricTensor::allocate(
        {world_size, m, 4}, at::ScalarType::Int, communicator->device());
    resources.threadload_done_semaphore.zero_();
    resources.threadload_done_semaphore_sym = std::make_unique<SymmetricTensor>(
        resources.threadload_done_semaphore);
    resources.threadload_done_semaphore_sym->setupRemoteHandles(
        "fused_remote_matmul_threadload_done");
  }

  if (impl == DistributedMatmulImpl::gpuAllgatherMultimemFusedNaiveCompute ||
      impl == DistributedMatmulImpl::gpuAllgatherMultimemTmaCompute) {
    resources.a_gathered_multimem = SymmetricTensor::allocate(
        {m, k}, at::ScalarType::Half, communicator->device());
    resources.a_gathered_multimem_sym =
        std::make_unique<SymmetricTensor>(resources.a_gathered_multimem);
    resources.a_gathered_multimem_sym->setupMulticast(
        /*exporter_rank=*/0, "fused_remote_matmul_staged_multimem");

    // Per-rank semaphore rows used by the fused multimem kernel barrier.
    // Shape is [writer_rank, row, vec4-int] so each writer can publish one
    // epoch per row, and each reader can wait on all writers for that row.
    resources.stage_semaphore_multimem = SymmetricTensor::allocate(
        {world_size, m, 4}, at::ScalarType::Int, communicator->device());
    resources.stage_semaphore_multimem.zero_();
    resources.stage_semaphore_multimem_sym =
        std::make_unique<SymmetricTensor>(resources.stage_semaphore_multimem);
    resources.stage_semaphore_multimem_sym->setupRemoteHandles(
        "fused_remote_matmul_stage_semaphore");
  }
  return resources;
}

double reduceMaxTimeMs(Communicator* communicator, double local_ms_per_iter) {
  // Reduce per-rank timing with MAX so throughput reflects slowest rank.
  at::Tensor max_time_tensor = at::tensor(
      {static_cast<float>(local_ms_per_iter)},
      at::TensorOptions().dtype(at::kFloat).device(communicator->device()));
  std::vector<at::Tensor> time_tensors = {max_time_tensor};
  communicator->getWorld()->allreduce(time_tensors, {c10d::ReduceOp::MAX})->wait();
  return static_cast<double>(max_time_tensor.item<float>());
}

double timeBaselinePytorchEagerMs(
    c10d::Backend* backend,
    at::Tensor& a_local_half,
    const at::Tensor& b_full_half,
    at::Tensor& c_out_half,
    const BenchmarkConfig& config,
    Communicator* communicator,
    cudaStream_t stream) {
  at::Tensor a_allgathered_half = at::empty(
      {a_local_half.size(0) * backend->getSize(), a_local_half.size(1)},
      a_local_half.options());
  auto run_once = [&]() {
    backend->_allgather_base(a_allgathered_half, a_local_half)->wait();
    at::matmul_out(c_out_half, a_allgathered_half, b_full_half);
  };
  return benchmarkLoopMs(config, communicator, stream, run_once);
}

double timeBaselinePytorchEagerCudaMs(
    Communication* communication,
    SymMemForAllgather* allgather_handle,
    at::Tensor& a_local_half,
    at::Tensor& a_allgathered_half,
    const at::Tensor& b_full_half,
    at::Tensor& c_out_half,
    const BenchmarkConfig& config,
    Communicator* communicator,
    cudaStream_t stream) {
  auto run_once = [&]() {
    postWithCudaBackend(
        communication,
        a_local_half,
        allgather_handle,
        (CUstream)stream,
        /*root=*/-1);
    waitWithCudaBackend(
        communication,
        allgather_handle,
        (CUstream)stream,
        /*root=*/-1);
    at::matmul_out(c_out_half, a_allgathered_half, b_full_half);
  };
  return benchmarkLoopMs(config, communicator, stream, run_once);
}

double timeImplementationMs(
    DistributedMatmulImpl impl,
    const BenchmarkConfig& config,
    Communicator* communicator,
    c10d::Backend* nccl_backend,
    Communication* cuda_allgather_communication,
    SymMemForAllgather* cuda_allgather_handle,
    const __half* const* device_remote_ptrs,
    at::Tensor& a_local_half,
    at::Tensor& a_allgathered_half_cuda,
    at::Tensor& a_gathered_threadload,
    at::Tensor& a_gathered_multimem,
    SymmetricTensor* threadload_ready_semaphore_sym,
    SymmetricTensor* threadload_done_semaphore_sym,
    SymmetricTensor* a_gathered_multimem_sym,
    SymmetricTensor* stage_semaphore_multimem_sym,
    int32_t* const* threadload_ready_semaphore_remote_ptrs,
    int32_t* const* threadload_done_semaphore_remote_ptrs,
    int32_t* const* stage_semaphore_remote_ptrs,
    const at::Tensor& b_full_half,
    at::Tensor& c_out_half,
    int64_t my_rank,
    int64_t world_size,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t m_per_rank,
    cudaStream_t stream) {
  // Dispatch to implementation-specific execution path.
  (void)a_gathered_multimem;
  const int64_t naive_block_x = runtimeParam(RuntimeParams::NaiveBlockX);
  const int64_t naive_block_y = runtimeParam(RuntimeParams::NaiveBlockY);
  const int64_t staged_block_threads =
      runtimeParam(RuntimeParams::StagedBlockThreads);
  const int64_t staged_grid_blocks =
      runtimeParam(RuntimeParams::StagedGridBlocks);
  switch (impl) {
    case DistributedMatmulImpl::naiveFusedKernel: {
      auto run_once = [&]() {
        timeFusedRemoteMatmulMs(
            device_remote_ptrs,
            reinterpret_cast<const __half*>(b_full_half.data_ptr()),
            reinterpret_cast<__half*>(c_out_half.data_ptr()),
            world_size,
            m,
            n,
            k,
            m_per_rank,
            naive_block_x,
            naive_block_y,
            /*warmup_iters=*/0,
            /*iters=*/1,
            stream);
      };
      return benchmarkLoopMs(config, communicator, stream, run_once);
    }
    case DistributedMatmulImpl::naiveFusedKernelCutlassCompute:
      return timeNaiveRemoteMatmulCutlassMs(
          device_remote_ptrs,
          a_gathered_threadload,
          b_full_half,
          c_out_half,
          m,
          k,
          m_per_rank,
          staged_block_threads,
          staged_grid_blocks,
          config.warmup_iters,
          config.iters,
          stream);
    case DistributedMatmulImpl::baselinePytorchEagerNccl:
      NVF_CHECK(
          nccl_backend != nullptr,
          "baselinePytorchEagerNccl requires a valid NCCL process group backend.");
      return timeBaselinePytorchEagerMs(
          nccl_backend,
          a_local_half,
          b_full_half,
          c_out_half,
          config,
          communicator,
          stream);
    case DistributedMatmulImpl::baselinePytorchEagerCuda:
      NVF_CHECK(
          cuda_allgather_communication != nullptr && cuda_allgather_handle != nullptr,
          "baselinePytorchEagerCuda requires initialized CUDA allgather resources.");
      return timeBaselinePytorchEagerCudaMs(
          cuda_allgather_communication,
          cuda_allgather_handle,
          a_local_half,
          a_allgathered_half_cuda,
          b_full_half,
          c_out_half,
          config,
          communicator,
          stream);
    case DistributedMatmulImpl::gpuAllgatherFusedNaiveCompute:
      NVF_CHECK(
          threadload_ready_semaphore_sym != nullptr &&
              threadload_done_semaphore_sym != nullptr &&
              threadload_ready_semaphore_remote_ptrs != nullptr &&
              threadload_done_semaphore_remote_ptrs != nullptr,
          "gpuAllgatherFusedNaiveCompute requires semaphore resources.");
      return timeSeparatedAllgatherMatmulThreadLoadSynchronizedMs(
          device_remote_ptrs,
          reinterpret_cast<const __half*>(b_full_half.data_ptr()),
          reinterpret_cast<__half*>(c_out_half.data_ptr()),
          reinterpret_cast<__half*>(a_gathered_threadload.data_ptr()),
          threadload_ready_semaphore_remote_ptrs,
          reinterpret_cast<int32_t*>(
              threadload_ready_semaphore_sym->localTensor().data_ptr()),
          threadload_done_semaphore_remote_ptrs,
          reinterpret_cast<int32_t*>(
              threadload_done_semaphore_sym->localTensor().data_ptr()),
          my_rank,
          world_size,
          m,
          n,
          k,
          m_per_rank,
          staged_block_threads,
          staged_grid_blocks,
          config.warmup_iters,
          config.iters,
          stream);
    case DistributedMatmulImpl::gpuAllgatherTmaCompute:
      NVF_CHECK(
          threadload_ready_semaphore_sym != nullptr &&
              threadload_done_semaphore_sym != nullptr &&
              threadload_ready_semaphore_remote_ptrs != nullptr &&
              threadload_done_semaphore_remote_ptrs != nullptr,
          "gpuAllgatherTmaCompute requires semaphore resources.");
      return timeSeparatedAllgatherMatmulThreadLoadSynchronizedCutlassMs(
          device_remote_ptrs,
          a_gathered_threadload,
          threadload_ready_semaphore_remote_ptrs,
          reinterpret_cast<int32_t*>(
              threadload_ready_semaphore_sym->localTensor().data_ptr()),
          threadload_done_semaphore_remote_ptrs,
          reinterpret_cast<int32_t*>(
              threadload_done_semaphore_sym->localTensor().data_ptr()),
          my_rank,
          world_size,
          b_full_half,
          c_out_half,
          m,
          k,
          m_per_rank,
          staged_block_threads,
          staged_grid_blocks,
          config.warmup_iters,
          config.iters,
          stream);
    case DistributedMatmulImpl::gpuAllgatherMultimemTmaCompute:
      NVF_CHECK(
          a_gathered_multimem_sym != nullptr &&
              stage_semaphore_multimem_sym != nullptr &&
              stage_semaphore_remote_ptrs != nullptr,
          "gpuAllgatherMultimemTmaCompute requires staging and "
          "semaphore tensors.");
      return timeSeparatedAllgatherMatmulMultimemCutlassMs(
          device_remote_ptrs,
          reinterpret_cast<__half*>(a_gathered_multimem_sym->multicastPtr()),
          a_gathered_multimem,
          stage_semaphore_remote_ptrs,
          reinterpret_cast<int32_t*>(
              stage_semaphore_multimem_sym->localTensor().data_ptr()),
          my_rank,
          world_size,
          b_full_half,
          c_out_half,
          m,
          k,
          m_per_rank,
          staged_block_threads,
          staged_grid_blocks,
          config.warmup_iters,
          config.iters,
          stream);
    case DistributedMatmulImpl::gpuAllgatherMultimemFusedNaiveCompute:
      NVF_CHECK(
          a_gathered_multimem_sym != nullptr &&
              stage_semaphore_multimem_sym != nullptr,
          "gpuAllgatherMultimemFusedNaiveCompute requires staging and semaphore tensors.");
      return timeSeparatedAllgatherMatmulMultimemMs(
          device_remote_ptrs,
          reinterpret_cast<const __half*>(b_full_half.data_ptr()),
          reinterpret_cast<__half*>(c_out_half.data_ptr()),
          reinterpret_cast<__half*>(a_gathered_multimem_sym->multicastPtr()),
          stage_semaphore_remote_ptrs,
          reinterpret_cast<int32_t*>(
              stage_semaphore_multimem_sym->localTensor().data_ptr()),
          my_rank,
          world_size,
          m,
          n,
          k,
          m_per_rank,
          staged_block_threads,
          staged_grid_blocks,
          config.warmup_iters,
          config.iters,
          stream);
  }
  NVF_ERROR(false, "Unsupported implementation enum: ", static_cast<int>(impl));
}

} // namespace

class FusedRemoteMatmulTest : public MultiDeviceTest,
                              public testing::WithParamInterface<
                                  DistributedMatmulImpl> {
 protected:
  static constexpr BenchmarkConfig kBenchmarkConfig = {
      /*warmup_iters=*/8,
      /*iters=*/30,
      /*time_mode=*/TimeMeasurementMode::CpuClock,
      /*barrier_at_each_iteration=*/false};
  static constexpr BenchmarkConfig kCorrectnessConfig = {
      /*warmup_iters=*/3,
      /*iters=*/1,
      /*time_mode=*/TimeMeasurementMode::CpuClock,
      /*barrier_at_each_iteration=*/false};
};

// Benchmark context:
// - A is sharded on M across ranks, B is replicated.
// - We compare three execution paths under identical setup/validation:
//   fused remote-pointer kernel, NCCL allgather+eager matmul, and CUDA-backend
//   allgather+eager matmul.
// - Rank 0 reports throughput using MAX latency reduced across ranks.
TEST_P(FusedRemoteMatmulTest, DistributedMatmulRemotePointerFused) {
  // ---------- Preconditions ----------
  if (!communicator_->is_available()) {
    GTEST_SKIP() << "Communicator is unavailable.";
  }
  if (communicator_->size() == 1) {
    GTEST_SKIP() << "Needs at least 2 devices.";
  }

  const int64_t world_size = communicator_->size();
  const int64_t my_rank = communicator_->deviceId();
  const auto impl = GetParam();

  if ((impl == DistributedMatmulImpl::gpuAllgatherMultimemFusedNaiveCompute ||
       impl == DistributedMatmulImpl::gpuAllgatherMultimemTmaCompute) &&
      !isMulticastSupported(my_rank)) {
    GTEST_SKIP() << "Multicast is not supported on this device.";
  }

  // ---------- Problem shape ----------
  Team all_devices(world_size);
  std::iota(all_devices.begin(), all_devices.end(), 0);

  constexpr int64_t m = 1024;
  constexpr int64_t k = 1024;
  constexpr int64_t n = 1024;
  NVF_ERROR(m % world_size == 0, "M must be divisible by world size.");
  const int64_t m_per_rank = m / world_size;

  // ---------- Inputs ----------
  const auto cpu_float_opts =
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  const auto gpu_half_opts =
      at::TensorOptions().dtype(at::kHalf).device(communicator_->device());

  // Deterministic inputs on every rank for fair cross-impl comparison.
  at::manual_seed(0);
  at::Tensor a_full_cpu = at::randn({m, k}, cpu_float_opts);
  at::Tensor b_full_cpu = at::randn({k, n}, cpu_float_opts);

  at::Tensor a_local_half = a_full_cpu
                                .slice(0, my_rank * m_per_rank, (my_rank + 1) * m_per_rank)
                                .to(gpu_half_opts.device(), at::kHalf);
  at::Tensor b_full_half = b_full_cpu.to(gpu_half_opts.device(), at::kHalf);

  at::Tensor a_local_sym = SymmetricTensor::allocate(
      {m_per_rank, k}, at::ScalarType::Half, communicator_->device());
  a_local_sym.copy_(a_local_half);
  SymmetricTensor symmetric_a(a_local_sym);
  symmetric_a.setupRemoteHandles("fused_remote_matmul_a");

  const __half* const* device_remote_ptrs =
      reinterpret_cast<const __half* const*>(symmetric_a.devicePeerPointers());

  // ---------- Outputs and stream ----------
  at::Tensor c_out_half = at::zeros({m, n}, gpu_half_opts);
  at::Tensor a_allgathered_half_cuda;
  at::Tensor a_gathered_threadload;
  at::Tensor a_gathered_multimem;
  c10::cuda::CUDAStream test_stream = c10::cuda::getStreamFromPool(
      /*isHighPriority=*/false,
      static_cast<int>(communicator_->device().index()));
  c10::cuda::CUDAStreamGuard stream_guard(test_stream);
  cudaStream_t stream = test_stream.stream();

  auto resources =
      initBenchmarkResources(impl, communicator_, all_devices, world_size, m, k);
  if (impl == DistributedMatmulImpl::baselinePytorchEagerNccl &&
      resources.nccl_backend == nullptr) {
    GTEST_SKIP() << "NCCL backend unavailable for baselinePytorchEagerNccl.";
  }
  a_allgathered_half_cuda = resources.a_allgathered_half_cuda;
  a_gathered_threadload = resources.a_gathered_threadload;
  a_gathered_multimem = resources.a_gathered_multimem;

  int32_t* const* device_stage_semaphore_remote_ptrs = nullptr;
  int32_t* const* device_threadload_ready_semaphore_remote_ptrs = nullptr;
  int32_t* const* device_threadload_done_semaphore_remote_ptrs = nullptr;
  if (impl == DistributedMatmulImpl::gpuAllgatherFusedNaiveCompute ||
      impl == DistributedMatmulImpl::gpuAllgatherTmaCompute) {
    NVF_CHECK(
        resources.threadload_ready_semaphore_sym != nullptr &&
            resources.threadload_done_semaphore_sym != nullptr,
        "Missing synchronized threadload semaphore resources.");
    device_threadload_ready_semaphore_remote_ptrs =
        reinterpret_cast<int32_t* const*>(
            resources.threadload_ready_semaphore_sym->devicePeerPointers());
    device_threadload_done_semaphore_remote_ptrs =
        reinterpret_cast<int32_t* const*>(
            resources.threadload_done_semaphore_sym->devicePeerPointers());
  }

  if (impl == DistributedMatmulImpl::gpuAllgatherMultimemFusedNaiveCompute ||
      impl == DistributedMatmulImpl::gpuAllgatherMultimemTmaCompute) {
    NVF_CHECK(
        resources.stage_semaphore_multimem_sym != nullptr,
        "Missing staged multimem semaphore resources.");
    device_stage_semaphore_remote_ptrs =
        reinterpret_cast<int32_t* const*>(
            resources.stage_semaphore_multimem_sym->devicePeerPointers());
  }

  const bool needs_cutlass_compute = impl ==
          DistributedMatmulImpl::naiveFusedKernelCutlassCompute ||
      impl == DistributedMatmulImpl::gpuAllgatherTmaCompute ||
      impl == DistributedMatmulImpl::gpuAllgatherMultimemTmaCompute;
  if (needs_cutlass_compute &&
      !canRunHopperCutlassCompute(
          a_gathered_threadload.defined() ? a_gathered_threadload : a_gathered_multimem,
          b_full_half)) {
    GTEST_SKIP()
        << "CUTLASS-compute variants require Hopper SM90 with TMA support.";
  }

  auto run_implementation = [&](const BenchmarkConfig& config) {
    return timeImplementationMs(
        impl,
        config,
        communicator_,
        resources.nccl_backend,
        resources.cuda_allgather_communication,
        resources.cuda_allgather_handle.get(),
        device_remote_ptrs,
        a_local_half,
        a_allgathered_half_cuda,
        a_gathered_threadload,
        a_gathered_multimem,
        resources.threadload_ready_semaphore_sym.get(),
        resources.threadload_done_semaphore_sym.get(),
        resources.a_gathered_multimem_sym.get(),
        resources.stage_semaphore_multimem_sym.get(),
        device_threadload_ready_semaphore_remote_ptrs,
        device_threadload_done_semaphore_remote_ptrs,
        device_stage_semaphore_remote_ptrs,
        b_full_half,
        c_out_half,
        my_rank,
        world_size,
        m,
        n,
        k,
        m_per_rank,
        stream);
  };

  // ---------- Correctness ----------
  // Run once before validation to execute the selected implementation path.
  (void)run_implementation(kCorrectnessConfig);

  at::Tensor c_ref_cpu = at::matmul(a_full_cpu, b_full_cpu);
  at::Tensor c_out_cpu = c_out_half.cpu().to(at::kFloat);
  EXPECT_TRUE(c_out_cpu.allclose(c_ref_cpu, 2e-1, 2e-1))
      << "Fused remote-pointer matmul output mismatch.";

  // ---------- Benchmark ----------
  communicator_->barrier();
  const double local_ms_per_iter = run_implementation(kBenchmarkConfig);
  communicator_->barrier();
  // Distributed throughput is constrained by the slowest rank.
  const double global_ms_per_iter =
      reduceMaxTimeMs(communicator_, local_ms_per_iter);

  // ---------- Reporting ----------
  const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
      static_cast<double>(k);
  const double tflops = flops / (global_ms_per_iter * 1.0e9);
  if (my_rank == 0) {
    std::cout << "[perf] fused_remote_matmul impl=" << implName(impl)
              << " M=" << m << " N=" << n << " K=" << k
              << " world_size=" << world_size << " : " << global_ms_per_iter
              << " ms/iter, " << tflops << " TFLOP/s" << std::endl;
  }

}

INSTANTIATE_TEST_SUITE_P(
    ,
    FusedRemoteMatmulTest,
    testing::Values(
        DistributedMatmulImpl::naiveFusedKernel,
        DistributedMatmulImpl::naiveFusedKernelCutlassCompute,
        DistributedMatmulImpl::baselinePytorchEagerNccl,
        DistributedMatmulImpl::baselinePytorchEagerCuda,
        DistributedMatmulImpl::gpuAllgatherFusedNaiveCompute,
        DistributedMatmulImpl::gpuAllgatherTmaCompute,
        DistributedMatmulImpl::gpuAllgatherMultimemTmaCompute,
        DistributedMatmulImpl::gpuAllgatherMultimemFusedNaiveCompute),
    [](const testing::TestParamInfo<DistributedMatmulImpl>& info) {
      return implName(info.param);
    });

} // namespace nvfuser
