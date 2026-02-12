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
    int64_t warmup_iters,
    int64_t iters,
    cudaStream_t stream);

namespace {

enum class RemoteMatmulImpl {
  naiveFusedKernel,
  baselinePytorchEagerNccl,
  baselinePytorchEagerCuda
};

enum class TimeMeasurementMode { CudaEvents, CpuClock };

struct BenchmarkConfig {
  int64_t warmup_iters;
  int64_t iters;
  TimeMeasurementMode time_mode;
  bool barrier_at_each_iteration;
};

const char* implName(RemoteMatmulImpl impl) {
  switch (impl) {
    case RemoteMatmulImpl::naiveFusedKernel:
      return "naiveFusedKernel";
    case RemoteMatmulImpl::baselinePytorchEagerNccl:
      return "baselinePytorchEagerNccl";
    case RemoteMatmulImpl::baselinePytorchEagerCuda:
      return "baselinePytorchEagerCuda";
  }
  NVF_ERROR(false, "Unknown implementation enum value: ", static_cast<int>(impl));
}

template <typename Fn>
double benchmarkLoopMs(
    const BenchmarkConfig& config,
    Communicator* communicator,
    cudaStream_t stream,
    Fn&& run_once) {
  for (int64_t i = 0; i < config.warmup_iters; ++i) {
    if (config.barrier_at_each_iteration) {
      communicator->barrier();
    }
    run_once();
  }
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetLastError());
  NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(stream));

  if (config.time_mode == TimeMeasurementMode::CudaEvents) {
    float total_ms = 0.0f;
    for (int64_t i = 0; i < config.iters; ++i) {
      if (config.barrier_at_each_iteration) {
        communicator->barrier();
      }
      cudaEvent_t start = nullptr;
      cudaEvent_t stop = nullptr;
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&start));
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&stop));
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(start, stream));
      run_once();
      NVFUSER_CUDA_RT_SAFE_CALL(cudaGetLastError());
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(stop, stream));
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(stop));
      float iter_ms = 0.0f;
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventElapsedTime(&iter_ms, start, stop));
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(start));
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(stop));
      total_ms += iter_ms;
    }
    return static_cast<double>(total_ms) / static_cast<double>(config.iters);
  }

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
    RemoteMatmulImpl impl,
    const BenchmarkConfig& config,
    Communicator* communicator,
    c10d::Backend* nccl_backend,
    Communication* cuda_allgather_communication,
    SymMemForAllgather* cuda_allgather_handle,
    const __half* const* device_remote_ptrs,
    at::Tensor& a_local_half,
    at::Tensor& a_allgathered_half_cuda,
    const at::Tensor& b_full_half,
    at::Tensor& c_out_half,
    int64_t world_size,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t m_per_rank,
    cudaStream_t stream) {
  switch (impl) {
    case RemoteMatmulImpl::naiveFusedKernel: {
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
            /*warmup_iters=*/0,
            /*iters=*/1,
            stream);
      };
      return benchmarkLoopMs(config, communicator, stream, run_once);
    }
    case RemoteMatmulImpl::baselinePytorchEagerNccl:
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
    case RemoteMatmulImpl::baselinePytorchEagerCuda:
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
  }
  NVF_ERROR(false, "Unsupported implementation enum: ", static_cast<int>(impl));
}

} // namespace

class FusedRemoteMatmulTest : public MultiDeviceTest,
                              public testing::WithParamInterface<RemoteMatmulImpl> {
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

TEST_P(FusedRemoteMatmulTest, DistributedMatmulRemotePointerFused) {
  if (!communicator_->is_available()) {
    GTEST_SKIP() << "Communicator is unavailable.";
  }
  if (communicator_->size() == 1) {
    GTEST_SKIP() << "Needs at least 2 devices.";
  }

  const int64_t world_size = communicator_->size();
  const int64_t my_rank = communicator_->deviceId();
  const auto impl = GetParam();

  Team all_devices(world_size);
  std::iota(all_devices.begin(), all_devices.end(), 0);

  c10d::Backend* nccl_backend = nullptr;
  if (impl == RemoteMatmulImpl::baselinePytorchEagerNccl) {
    if (!communicator_->isBackendAvailable(CommunicatorBackend::kNccl)) {
      GTEST_SKIP() << "NCCL backend unavailable for baselinePytorchEagerNccl.";
    }
    nccl_backend =
        communicator_->getBackendForTeam(all_devices, CommunicatorBackend::kNccl);
  }
  std::unique_ptr<hir::HostIrContainer> cuda_hic;
  Communication* cuda_allgather_communication = nullptr;
  std::unique_ptr<SymMemForAllgather> cuda_allgather_handle;

  constexpr int64_t m = 1024;
  constexpr int64_t k = 1024;
  constexpr int64_t n = 1024;
  NVF_ERROR(m % world_size == 0, "M must be divisible by world size.");
  const int64_t m_per_rank = m / world_size;

  const auto cpu_float_opts =
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  const auto gpu_half_opts =
      at::TensorOptions().dtype(at::kHalf).device(communicator_->device());

  // Every rank builds identical global inputs from the same seed.
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

  std::vector<const __half*> host_remote_ptrs(world_size);
  for (int64_t rank = 0; rank < world_size; ++rank) {
    host_remote_ptrs[rank] =
        reinterpret_cast<const __half*>(symmetric_a.remoteTensor(rank).data_ptr());
  }

  __half** device_remote_ptrs = nullptr;
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaMalloc(&device_remote_ptrs, world_size * sizeof(__half*)));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
      device_remote_ptrs,
      host_remote_ptrs.data(),
      world_size * sizeof(__half*),
      cudaMemcpyHostToDevice));

  at::Tensor c_out_half = at::zeros({m, n}, gpu_half_opts);
  at::Tensor a_allgathered_half_cuda;
  c10::cuda::CUDAStream test_stream = c10::cuda::getStreamFromPool(
      /*isHighPriority=*/false,
      static_cast<int>(communicator_->device().index()));
  c10::cuda::CUDAStreamGuard stream_guard(test_stream);
  cudaStream_t stream = test_stream.stream();

  if (impl == RemoteMatmulImpl::baselinePytorchEagerCuda) {
    cuda_hic = std::make_unique<hir::HostIrContainer>();
    FusionGuard fg(cuda_hic.get());
    auto* in_tv = makeContigTensor(2);
    auto* out_tv = makeContigTensor(2);
    DeviceMesh mesh = DeviceMesh::createForNumDevices(world_size);
    in_tv->setDeviceMesh(mesh);
    out_tv->setDeviceMesh(mesh);
    cuda_allgather_communication = IrBuilder::create<Communication>(
        CommunicationType::Allgather,
        out_tv,
        in_tv,
        all_devices,
        /*root=*/-1,
        RedOpType::UNUSED,
        CommunicatorBackend::kCuda);
    a_allgathered_half_cuda = SymmetricTensor::allocate(
        {m, k}, at::ScalarType::Half, communicator_->device());
    cuda_allgather_handle = std::make_unique<SymMemForAllgather>(
        cuda_allgather_communication, a_allgathered_half_cuda);
  }

  // Correctness check.
  (void)timeImplementationMs(
      impl,
      kCorrectnessConfig,
      communicator_,
      nccl_backend,
      cuda_allgather_communication,
      cuda_allgather_handle.get(),
      const_cast<const __half* const*>(device_remote_ptrs),
      a_local_half,
      a_allgathered_half_cuda,
      b_full_half,
      c_out_half,
      world_size,
      m,
      n,
      k,
      m_per_rank,
      stream);

  at::Tensor c_ref_cpu = at::matmul(a_full_cpu, b_full_cpu);
  at::Tensor c_out_cpu = c_out_half.cpu().to(at::kFloat);
  EXPECT_TRUE(c_out_cpu.allclose(c_ref_cpu, 2e-1, 2e-1))
      << "Fused remote-pointer matmul output mismatch.";

  communicator_->barrier();
  const double local_ms_per_iter = timeImplementationMs(
      impl,
      kBenchmarkConfig,
      communicator_,
      nccl_backend,
      cuda_allgather_communication,
      cuda_allgather_handle.get(),
      const_cast<const __half* const*>(device_remote_ptrs),
      a_local_half,
      a_allgathered_half_cuda,
      b_full_half,
      c_out_half,
      world_size,
      m,
      n,
      k,
      m_per_rank,
      stream);
  communicator_->barrier();

  at::Tensor max_time_tensor = at::tensor(
      {static_cast<float>(local_ms_per_iter)},
      at::TensorOptions().dtype(at::kFloat).device(communicator_->device()));
  std::vector<at::Tensor> time_tensors = {max_time_tensor};
  communicator_->getWorld()->allreduce(time_tensors, {c10d::ReduceOp::MAX})->wait();
  const double global_ms_per_iter = static_cast<double>(max_time_tensor.item<float>());

  const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
      static_cast<double>(k);
  const double tflops = flops / (global_ms_per_iter * 1.0e9);
  if (my_rank == 0) {
    std::cout << "[perf] fused_remote_matmul impl=" << implName(impl)
              << " M=" << m << " N=" << n << " K=" << k
              << " world_size=" << world_size << " : " << global_ms_per_iter
              << " ms/iter, " << tflops << " TFLOP/s" << std::endl;
  }

  NVFUSER_CUDA_RT_SAFE_CALL(cudaFree(device_remote_ptrs));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    FusedRemoteMatmulTest,
    testing::Values(
        RemoteMatmulImpl::naiveFusedKernel,
        RemoteMatmulImpl::baselinePytorchEagerNccl,
        RemoteMatmulImpl::baselinePytorchEagerCuda),
    [](const testing::TestParamInfo<RemoteMatmulImpl>& info) {
      return implName(info.param);
    });

} // namespace nvfuser
