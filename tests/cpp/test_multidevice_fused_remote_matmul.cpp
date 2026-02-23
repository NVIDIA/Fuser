// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
//
// =========================================================================
// Distributed Matmul Benchmark -- Test Harness
//
// Measures allgather + matmul throughput for C = A * B where A is
// row-sharded on M across ranks and B is replicated.  Compares
// baseline (NCCL / CUDA P2P) and fused kernel implementations.
//
// See test_multidevice_fused_remote_matmul.h for the performance
// summary and implementation descriptions.
// =========================================================================

#include "test_multidevice_fused_remote_matmul.h"

#include <cuda_fp16.h>

#include <chrono>
#include <numeric>

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "fusion.h"
#include "host_ir/container.h"
#include "ir/builder.h"
#include "multidevice/communication.h"
#include "multidevice/communicator.h"
#include "multidevice/cuda_p2p.h"
#include "multidevice/ipc_handle.h"
#include "multidevice/symmetric_tensor.h"
#include "tests/cpp/multidevice.h"

namespace nvfuser {

namespace {

// =========================================================================
// Timing helpers
// =========================================================================

// Batched GPU timing: one cuda-event pair around all iterations.
// No per-iteration host sync, matching the original kernel timing
// methodology.  Avoids ~15us cudaStreamSynchronize overhead that
// dominates sub-100us kernels like CUTLASS TMA matmuls.
template <typename Fn>
double batchedKernelTimeMs(
    int64_t warmup_iters,
    int64_t iters,
    cudaStream_t stream,
    Fn&& run_once) {
  for (int64_t i = 0; i < warmup_iters; ++i)
    run_once();
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetLastError());
  NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(stream));
  cudaEvent_t start, stop;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&start));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&stop));
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaEventRecord(start, stream));
  for (int64_t i = 0; i < iters; ++i)
    run_once();
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetLastError());
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaEventRecord(stop, stream));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(stop));
  float total_ms;
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaEventElapsedTime(&total_ms, start, stop));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(start));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(stop));
  return static_cast<double>(total_ms) / iters;
}

// Per-iteration timing for baselines with host-blocking waits.
template <typename Fn>
double benchmarkLoopMs(
    const BenchmarkConfig& config,
    Communicator* communicator,
    cudaStream_t stream,
    Fn&& run_once) {
  NVF_CHECK(config.iters > 0, "iters must be > 0");
  for (int64_t i = 0; i < config.warmup_iters; ++i) {
    if (config.barrier_at_each_iteration)
      communicator->barrier();
    run_once();
  }
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetLastError());
  NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(stream));

  if (config.time_mode == TimeMeasurementMode::CudaEvents) {
    cudaEvent_t start, stop;
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&start));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&stop));
    float total_ms = 0.f;
    for (int64_t i = 0; i < config.iters; ++i) {
      if (config.barrier_at_each_iteration)
        communicator->barrier();
      NVFUSER_CUDA_RT_SAFE_CALL(
          cudaEventRecord(start, stream));
      run_once();
      NVFUSER_CUDA_RT_SAFE_CALL(cudaGetLastError());
      NVFUSER_CUDA_RT_SAFE_CALL(
          cudaEventRecord(stop, stream));
      NVFUSER_CUDA_RT_SAFE_CALL(
          cudaEventSynchronize(stop));
      float ms;
      NVFUSER_CUDA_RT_SAFE_CALL(
          cudaEventElapsedTime(&ms, start, stop));
      total_ms += ms;
    }
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(start));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(stop));
    return static_cast<double>(total_ms) / config.iters;
  }

  double total_ms = 0.0;
  for (int64_t i = 0; i < config.iters; ++i) {
    if (config.barrier_at_each_iteration)
      communicator->barrier();
    NVFUSER_CUDA_RT_SAFE_CALL(
        cudaStreamSynchronize(stream));
    auto t0 = std::chrono::high_resolution_clock::now();
    run_once();
    NVFUSER_CUDA_RT_SAFE_CALL(cudaGetLastError());
    NVFUSER_CUDA_RT_SAFE_CALL(
        cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    total_ms +=
        std::chrono::duration<double, std::milli>(t1 - t0)
            .count();
  }
  return total_ms / config.iters;
}

// =========================================================================
// Resource helpers
// =========================================================================

bool needsThreadloadRes(DistributedMatmulImpl impl) {
  using I = DistributedMatmulImpl;
  return impl == I::threadloadGatherScalarCompute ||
      impl == I::threadloadGatherThenCutlass;
}

bool needsMultimemRes(DistributedMatmulImpl impl) {
  using I = DistributedMatmulImpl;
  return impl == I::multimemGatherScalarCompute ||
      impl == I::multimemGatherThenCutlass;
}

bool needsCutlass(DistributedMatmulImpl impl) {
  using I = DistributedMatmulImpl;
  return impl == I::threadloadGatherThenCutlass ||
      impl == I::multimemGatherThenCutlass;
}

struct OwnedResources {
  std::unique_ptr<SymmetricTensor> a_sym;
  std::unique_ptr<SymmetricTensor> ready_sym;
  std::unique_ptr<SymmetricTensor> done_sym;
  std::unique_ptr<SymmetricTensor> stage_sym;
  std::unique_ptr<SymmetricTensor> multimem_sym;
  std::unique_ptr<hir::HostIrContainer> cuda_hic;
  std::unique_ptr<SymMemForAllgather> cuda_ag_handle;
  at::Tensor ready_t, done_t, stage_t;
};

void initResources(
    DistributedMatmulImpl impl,
    Communicator* comm,
    const Team& team,
    int64_t ws, int64_t m, int64_t k,
    OwnedResources& res,
    DistributedMatmulContext& ctx) {
  using I = DistributedMatmulImpl;
  auto dev = comm->device();

  if (impl == I::baselineNcclAllgatherMatmul) {
    if (comm->isBackendAvailable(CommunicatorBackend::kNccl))
      ctx.nccl_backend = comm->getBackendForTeam(
          team, CommunicatorBackend::kNccl);
  }

  if (impl == I::baselineCudaAllgatherMatmul) {
    res.cuda_hic =
        std::make_unique<hir::HostIrContainer>();
    FusionGuard fg(res.cuda_hic.get());
    auto* itv = makeContigTensor(2);
    auto* otv = makeContigTensor(2);
    auto mesh = DeviceMesh::createForNumDevices(ws);
    itv->setDeviceMesh(mesh);
    otv->setDeviceMesh(mesh);
    auto* cir = IrBuilder::create<Communication>(
        CommunicationType::Allgather, otv, itv,
        team, -1, RedOpType::UNUSED,
        CommunicatorBackend::kCuda);
    ctx.a_allgathered_cuda = SymmetricTensor::allocate(
        {m, k}, at::ScalarType::Half, dev);
    res.cuda_ag_handle =
        std::make_unique<SymMemForAllgather>(
            cir, ctx.a_allgathered_cuda);
    ctx.cuda_comm = cir;
    ctx.cuda_handle = res.cuda_ag_handle.get();
  }

  if (needsThreadloadRes(impl)) {
    ctx.a_gathered = at::empty(
        {m, k},
        at::TensorOptions().dtype(at::kHalf).device(dev));

    auto make_sem = [&](const char* tag)
        -> std::pair<at::Tensor,
                     std::unique_ptr<SymmetricTensor>> {
      at::Tensor t = SymmetricTensor::allocate(
          {ws, m, 4}, at::ScalarType::Int, dev);
      t.zero_();
      auto s = std::make_unique<SymmetricTensor>(t);
      s->setupRemoteHandles(tag);
      return {t, std::move(s)};
    };

    auto [rt, rs] = make_sem("fused_matmul_ready");
    res.ready_t = rt;
    res.ready_sym = std::move(rs);
    ctx.ready_sem_remote =
        reinterpret_cast<int32_t* const*>(
            res.ready_sym->devicePeerPointers());
    ctx.ready_sem_local = reinterpret_cast<int32_t*>(
        res.ready_sym->localTensor().data_ptr());

    auto [dt, ds] = make_sem("fused_matmul_done");
    res.done_t = dt;
    res.done_sym = std::move(ds);
    ctx.done_sem_remote =
        reinterpret_cast<int32_t* const*>(
            res.done_sym->devicePeerPointers());
    ctx.done_sem_local = reinterpret_cast<int32_t*>(
        res.done_sym->localTensor().data_ptr());
  }

  if (needsMultimemRes(impl)) {
    ctx.a_gathered_multimem = SymmetricTensor::allocate(
        {m, k}, at::ScalarType::Half, dev);
    res.multimem_sym = std::make_unique<SymmetricTensor>(
        ctx.a_gathered_multimem);
    res.multimem_sym->setupMulticast(
        0, "fused_matmul_mc");
    ctx.multicast_ptr = reinterpret_cast<__half*>(
        res.multimem_sym->multicastPtr());

    res.stage_t = SymmetricTensor::allocate(
        {ws, m, 4}, at::ScalarType::Int, dev);
    res.stage_t.zero_();
    res.stage_sym =
        std::make_unique<SymmetricTensor>(res.stage_t);
    res.stage_sym->setupRemoteHandles(
        "fused_matmul_stage");
    ctx.stage_sem_remote =
        reinterpret_cast<int32_t* const*>(
            res.stage_sym->devicePeerPointers());
    ctx.stage_sem_local = reinterpret_cast<int32_t*>(
        res.stage_sym->localTensor().data_ptr());
  }
}

double reduceMaxTimeMs(
    Communicator* comm, double local_ms) {
  at::Tensor t = at::tensor(
      {static_cast<float>(local_ms)},
      at::TensorOptions()
          .dtype(at::kFloat)
          .device(comm->device()));
  std::vector<at::Tensor> tv = {t};
  comm->getWorld()
      ->allreduce(tv, {c10d::ReduceOp::MAX})
      ->wait();
  return static_cast<double>(t.item<float>());
}

// =========================================================================
// Implementation dispatcher
//
// Each case builds a run_once lambda and wraps it with
// benchmarkLoopMs.  Kernel launchers live in the .cu file.
// =========================================================================

double runImplementation(
    DistributedMatmulImpl impl,
    DistributedMatmulContext& ctx,
    const BenchmarkConfig& config) {
  using I = DistributedMatmulImpl;
  const int64_t wu = config.warmup_iters;
  const int64_t it = config.iters;
  switch (impl) {
    case I::baselineNcclAllgatherMatmul: {
      at::Tensor a_full = at::empty(
          {ctx.m, ctx.k}, ctx.a_local_half.options());
      auto run = [&]() {
        ctx.nccl_backend
            ->_allgather_base(a_full, ctx.a_local_half)
            ->wait();
        at::matmul_out(
            ctx.c_out_half, a_full, ctx.b_full_half);
      };
      return benchmarkLoopMs(
          config, ctx.communicator, ctx.stream, run);
    }
    case I::baselineCudaAllgatherMatmul: {
      auto run = [&]() {
        postWithCudaBackend(
            ctx.cuda_comm, ctx.a_local_half,
            ctx.cuda_handle,
            (CUstream)ctx.stream, -1);
        waitWithCudaBackend(
            ctx.cuda_comm, ctx.cuda_handle,
            (CUstream)ctx.stream, -1);
        at::matmul_out(
            ctx.c_out_half, ctx.a_allgathered_cuda,
            ctx.b_full_half);
      };
      return benchmarkLoopMs(
          config, ctx.communicator, ctx.stream, run);
    }
    case I::naiveRemoteRead: {
      return batchedKernelTimeMs(
          wu, it, ctx.stream, [&]() {
            launchNaiveRemoteRead(ctx);
          });
    }
    case I::threadloadGatherScalarCompute: {
      int64_t epoch = 0;
      return batchedKernelTimeMs(
          wu, it, ctx.stream, [&]() {
            launchThreadloadGather(
                ctx, static_cast<int32_t>(epoch), true);
            ++epoch;
          });
    }
    case I::threadloadGatherThenCutlass: {
      int64_t epoch = 0;
      return batchedKernelTimeMs(
          wu, it, ctx.stream, [&]() {
            launchThreadloadGather(
                ctx, static_cast<int32_t>(epoch), false);
            matmulTma(
                ctx.c_out_half,
                ctx.a_gathered, ctx.b_full_half);
            ++epoch;
          });
    }
    case I::multimemGatherScalarCompute: {
      int64_t epoch = 0;
      return batchedKernelTimeMs(
          wu, it, ctx.stream, [&]() {
            launchMultimemGather(
                ctx, static_cast<int32_t>(epoch), true);
            ++epoch;
          });
    }
    case I::multimemGatherThenCutlass: {
      int64_t epoch = 0;
      return batchedKernelTimeMs(
          wu, it, ctx.stream, [&]() {
            launchMultimemGather(
                ctx, static_cast<int32_t>(epoch), false);
            matmulTma(
                ctx.c_out_half,
                ctx.a_gathered_multimem,
                ctx.b_full_half);
            ++epoch;
          });
    }
  }
  NVF_ERROR(false, "Unknown implementation.");
}

} // anonymous namespace

// =========================================================================
// Test fixture
// =========================================================================

class FusedRemoteMatmulTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<
          DistributedMatmulImpl> {
 protected:
  static constexpr BenchmarkConfig kConfig = {
      /*warmup_iters=*/8,
      /*iters=*/30,
      /*time_mode=*/TimeMeasurementMode::CpuClock,
      /*barrier_at_each_iteration=*/false};
};

TEST_P(FusedRemoteMatmulTest, DistributedMatmul) {
  if (!communicator_->is_available())
    GTEST_SKIP() << "Communicator unavailable.";
  if (communicator_->size() == 1)
    GTEST_SKIP() << "Needs >= 2 devices.";

  const int64_t ws = communicator_->size();
  const int64_t rank = communicator_->deviceId();
  const auto impl = GetParam();

  if (needsMultimemRes(impl) &&
      !isMulticastSupported(rank))
    GTEST_SKIP() << "Multicast unsupported.";

  // ---- Problem shape ----
  constexpr int64_t m = 1024, k = 1024, n = 1024;
  NVF_ERROR(m % ws == 0);
  const int64_t mpr = m / ws;
  Team team(ws);
  std::iota(team.begin(), team.end(), 0);

  // ---- Inputs ----
  at::manual_seed(0);
  auto cpu_f = at::TensorOptions().dtype(at::kFloat);
  auto gpu_h = at::TensorOptions()
                   .dtype(at::kHalf)
                   .device(communicator_->device());
  at::Tensor a_full = at::randn({m, k}, cpu_f);
  at::Tensor b_full = at::randn({k, n}, cpu_f);
  at::Tensor a_local =
      a_full.slice(0, rank * mpr, (rank + 1) * mpr)
          .to(gpu_h.device(), at::kHalf);
  at::Tensor b_gpu = b_full.to(gpu_h.device(), at::kHalf);

  at::Tensor a_sym = SymmetricTensor::allocate(
      {mpr, k}, at::ScalarType::Half,
      communicator_->device());
  a_sym.copy_(a_local);
  OwnedResources res;
  res.a_sym = std::make_unique<SymmetricTensor>(a_sym);
  res.a_sym->setupRemoteHandles("fused_matmul_a");

  // ---- Build context ----
  DistributedMatmulContext ctx;
  ctx.m = m;
  ctx.n = n;
  ctx.k = k;
  ctx.m_per_rank = mpr;
  ctx.my_rank = rank;
  ctx.world_size = ws;
  ctx.device_remote_ptrs =
      reinterpret_cast<const __half* const*>(
          res.a_sym->devicePeerPointers());
  ctx.a_local_half = a_local;
  ctx.b_full_half = b_gpu;
  ctx.c_out_half = at::zeros({m, n}, gpu_h);
  ctx.communicator = communicator_;

  c10::cuda::CUDAStream test_stream =
      c10::cuda::getStreamFromPool(
          false,
          static_cast<int>(
              communicator_->device().index()));
  c10::cuda::CUDAStreamGuard guard(test_stream);
  ctx.stream = test_stream.stream();

  initResources(
      impl, communicator_, team, ws, m, k, res, ctx);

  // ---- Capability gates ----
  if (impl ==
          DistributedMatmulImpl::
              baselineNcclAllgatherMatmul &&
      ctx.nccl_backend == nullptr)
    GTEST_SKIP() << "NCCL backend unavailable.";

  if (needsCutlass(impl)) {
    at::Tensor ref = ctx.a_gathered.defined()
        ? ctx.a_gathered
        : ctx.a_gathered_multimem;
    if (!canRunCutlassCompute(ref, b_gpu))
      GTEST_SKIP() << "CUTLASS needs Hopper SM90.";
  }

  // ---- Correctness (1 iteration, no warmup) ----
  (void)runImplementation(
      impl, ctx,
      {0, 1, TimeMeasurementMode::CpuClock, false});
  at::Tensor c_ref = at::matmul(a_full, b_full);
  EXPECT_TRUE(
      ctx.c_out_half.cpu().to(at::kFloat).allclose(
          c_ref, 2e-1, 2e-1))
      << "Mismatch for " << implName(impl);

  // ---- Benchmark ----
  communicator_->barrier();
  double local_ms =
      runImplementation(impl, ctx, kConfig);
  communicator_->barrier();
  double global_ms =
      reduceMaxTimeMs(communicator_, local_ms);

  // ---- Report ----
  double tflops =
      2.0 * m * n * k / (global_ms * 1e9);
  if (rank == 0) {
    std::cout << "[perf] fused_remote_matmul"
              << " impl=" << implName(impl)
              << " M=" << m << " N=" << n << " K=" << k
              << " world_size=" << ws << " : "
              << global_ms << " ms/iter, " << tflops
              << " TFLOP/s" << std::endl;
  }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    FusedRemoteMatmulTest,
    testing::Values(
        DistributedMatmulImpl::baselineNcclAllgatherMatmul,
        DistributedMatmulImpl::baselineCudaAllgatherMatmul,
        DistributedMatmulImpl::naiveRemoteRead,
        DistributedMatmulImpl::threadloadGatherScalarCompute,
        DistributedMatmulImpl::multimemGatherScalarCompute,
        DistributedMatmulImpl::threadloadGatherThenCutlass,
        DistributedMatmulImpl::multimemGatherThenCutlass),
    [](const testing::TestParamInfo<
        DistributedMatmulImpl>& info) {
      return implName(info.param);
    });

} // namespace nvfuser
