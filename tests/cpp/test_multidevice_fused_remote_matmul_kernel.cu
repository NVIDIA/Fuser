// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
//
// =========================================================================
// Distributed Matmul Kernels
//
// This file contains CUDA kernels and host-side launcher functions for
// the distributed matmul benchmark.  Each kernel combines a
// communication strategy with a compute strategy:
//
// Communication strategies:
//   - Naive remote read: each thread reads A directly from the owner
//     rank via remote pointers.
//   - Threadload gather: cooperative thread loads stage A rows into a
//     local buffer, synchronized via ready/done semaphores.
//   - Multimem gather: owner rank writes A rows to a multicast buffer
//     using multimem.st (Hopper SM90+), synchronized via semaphores.
//
// Compute strategies:
//   - Scalar: each thread accumulates one output element.
//   - CUTLASS TMA: host-launched Hopper GEMM (in the .cpp file, not
//     here -- the kernel is launched with n=0 to skip in-kernel compute).
//
// The CUTLASS TMA matmul wrapper (matmulTma) is also defined here,
// moved from csrc/runtime/matmul_tma.cu for self-containment.
// =========================================================================

#include "test_multidevice_fused_remote_matmul.h"

#include <cuda_fp16.h>

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_utils.h>

// CUTLASS TMA matmul (Hopper SM90)
#if defined(NVFUSER_ENABLE_CUTLASS)
#if !defined(__CUDACC_VER_MAJOR__)
#define __CUDACC_VER_MAJOR__ 13
#define __CUDACC_VER_MINOR__ 0
#endif
#include "cutlass/arch/config.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#endif

namespace nvfuser {

namespace {

// =========================================================================
// Section 1: CUTLASS TMA matmul wrapper
//
// Provides matmulTma() -- a Hopper SM90 GEMM using CUTLASS 3.x with
// TMA loads.  Moved from csrc/runtime/matmul_tma.cu so the benchmark
// is self-contained.
// =========================================================================

bool hasValidTmaShape(
    const at::Tensor& a,
    const at::Tensor& b) {
  if (!a.defined() || !b.defined()) return false;
  if (!a.is_cuda() || !b.is_cuda()) return false;
  if (a.dim() != 2 || b.dim() != 2) return false;
  if (a.scalar_type() != b.scalar_type()) return false;
  if (!(a.scalar_type() == at::ScalarType::Half ||
        a.scalar_type() == at::ScalarType::BFloat16))
    return false;
  if (!a.is_contiguous() || !b.is_contiguous()) return false;
  if (a.size(1) != b.size(0)) return false;
  if (a.get_device() != b.get_device()) return false;
  constexpr int64_t kAlign = 8;
  if (a.size(1) % kAlign != 0 || b.size(1) % kAlign != 0)
    return false;
  return true;
}

#if defined(NVFUSER_ENABLE_CUTLASS) && \
    defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
using namespace cute;

template <typename ElementT>
struct TmaSm90Config {
  using EA = ElementT;
  using EB = ElementT;
  using EC = ElementT;
  using ED = ElementT;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int kAA =
      128 / cutlass::sizeof_bits<EA>::value;
  static constexpr int kAB =
      128 / cutlass::sizeof_bits<EB>::value;
  static constexpr int kAC =
      128 / cutlass::sizeof_bits<EC>::value;
  static constexpr int kAD =
      128 / cutlass::sizeof_bits<ED>::value;
  using Acc = float;
  using Arch = cutlass::arch::Sm90;
  using Op = cutlass::arch::OpClassTensorOp;
  using Tile = Shape<_128, _128, _64>;
  using Cluster = Shape<_1, _1, _1>;
  using SmTile = Shape<_128, _128, _64>;

  using Epi =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          Arch, Op, SmTile, Cluster,
          cutlass::epilogue::collective::EpilogueTileAuto,
          Acc, Acc, EC, LayoutC, kAC, ED, LayoutD, kAD,
          cutlass::epilogue::collective::
              EpilogueScheduleAuto>::CollectiveOp;

  using Main =
      typename cutlass::gemm::collective::CollectiveBuilder<
          Arch, Op, EA, LayoutA, kAA, EB, LayoutB, kAB,
          Acc, Tile, Cluster,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(
                  sizeof(typename Epi::SharedStorage))>,
          cutlass::gemm::collective::
              KernelScheduleAuto>::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Main, Epi, void>;
  using Gemm =
      cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using SA = typename Gemm::GemmKernel::StrideA;
  using SB = typename Gemm::GemmKernel::StrideB;
  using SC = typename Gemm::GemmKernel::StrideC;
  using SD = typename Gemm::GemmKernel::StrideD;
};

template <typename ElementT>
void runGemmSm90(
    at::Tensor& out,
    const at::Tensor& a,
    const at::Tensor& b,
    int64_t m, int64_t n, int64_t k,
    cudaStream_t stream) {
  using C = TmaSm90Config<ElementT>;
  auto sa = cutlass::make_cute_packed_stride(
      typename C::SA{}, {(int)m, (int)k, 1});
  auto sb = cutlass::make_cute_packed_stride(
      typename C::SB{}, {(int)k, (int)n, 1});
  auto sc = cutlass::make_cute_packed_stride(
      typename C::SC{}, {(int)m, (int)n, 1});
  auto sd = cutlass::make_cute_packed_stride(
      typename C::SD{}, {(int)m, (int)n, 1});
  typename C::Kernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {(int)m, (int)n, (int)k, 1},
      {static_cast<const typename C::EA*>(a.data_ptr()), sa,
       static_cast<const typename C::EB*>(b.data_ptr()), sb},
      {{}, nullptr, sc,
       static_cast<typename C::ED*>(out.data_ptr()), sd}};
  typename C::Gemm gemm;
  size_t ws = C::Gemm::get_workspace_size(args);
  auto wt = at::empty(
      {(int64_t)ws},
      at::TensorOptions().dtype(at::kByte).device(
          a.device()));
  NVF_CHECK(
      gemm.can_implement(args) == cutlass::Status::kSuccess,
      "CUTLASS cannot implement this GEMM.");
  NVF_CHECK(
      gemm.initialize(args, wt.data_ptr(), stream) ==
          cutlass::Status::kSuccess,
      "CUTLASS init failed.");
  NVF_CHECK(
      gemm.run(args, wt.data_ptr(), stream, nullptr, true) ==
          cutlass::Status::kSuccess,
      "CUTLASS run failed.");
}

#else

template <typename ElementT>
void runGemmSm90(
    at::Tensor&, const at::Tensor&, const at::Tensor&,
    int64_t, int64_t, int64_t, cudaStream_t) {
  NVF_THROW("CUTLASS SM90 support required for TMA matmul.");
}

#endif

// =========================================================================
// Section 2: Remote semaphore helpers
//
// Device-side helpers for inter-rank synchronization.  Each semaphore
// row stores kVecW int32 epochs.  Owner publishes; readers poll.
// =========================================================================

constexpr int64_t kVecW = 4;
constexpr int64_t kMaxPoll = 1LL << 26;

__device__ inline void publishToAll(
    int32_t* const* remote, int32_t* local,
    int64_t writer, int64_t row, int64_t m,
    int64_t ws, int32_t epoch) {
  int32_t* my = local + (writer * m + row) * kVecW;
  for (int64_t i = 0; i < kVecW; ++i) my[i] = epoch;
  __threadfence_system();
  for (int64_t p = 0; p < ws; ++p) {
    int32_t* d = remote[p] + (writer * m + row) * kVecW;
    for (int64_t i = 0; i < kVecW; ++i) d[i] = epoch;
  }
  __threadfence_system();
}

__device__ inline void publishToOne(
    int32_t* target, int64_t writer,
    int64_t row, int64_t m, int32_t epoch) {
  int32_t* d = target + (writer * m + row) * kVecW;
  for (int64_t i = 0; i < kVecW; ++i) d[i] = epoch;
  __threadfence_system();
}

__device__ inline void setLocal(
    int32_t* local, int64_t writer,
    int64_t row, int64_t m, int32_t epoch) {
  int32_t* d = local + (writer * m + row) * kVecW;
  for (int64_t i = 0; i < kVecW; ++i) d[i] = epoch;
  __threadfence_system();
}

__device__ inline void waitOne(
    int32_t* local, int64_t row, int64_t m,
    int64_t writer, int32_t epoch) {
  auto* p = reinterpret_cast<unsigned int*>(
      local + (writer * m + row) * kVecW);
  int64_t s = 0;
  while (atomicAdd(p, 0U) < (unsigned)epoch)
    if (++s > kMaxPoll) asm volatile("trap;");
}

__device__ inline void waitAll(
    int32_t* local, int64_t row, int64_t m,
    int64_t ws, int32_t epoch) {
  for (int64_t r = 0; r < ws; ++r) {
    auto* p = reinterpret_cast<unsigned int*>(
        local + (r * m + row) * kVecW);
    int64_t s = 0;
    while (atomicAdd(p, 0U) < (unsigned)epoch)
      if (++s > kMaxPoll) asm volatile("trap;");
  }
}

// =========================================================================
// Section 3: Kernel definitions
// =========================================================================

// --- 3a. naiveRemoteReadKernel ---
// Each thread computes one C[row,col].  A is read directly from the
// owner rank's shard via remote pointers -- no staging, no gather.
__global__ void naiveRemoteReadKernel(
    const __half* const* a_shards,
    const __half* b, __half* c,
    int64_t m, int64_t n, int64_t k,
    int64_t m_per_rank) {
  int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;
  int64_t owner = row / m_per_rank;
  int64_t lr = row - owner * m_per_rank;
  const __half* a = a_shards[owner];
  float acc = 0.f;
  for (int64_t kk = 0; kk < k; ++kk)
    acc += __half2float(a[lr * k + kk]) *
        __half2float(b[kk * n + col]);
  c[row * n + col] = __float2half(acc);
}

// --- 3b. threadloadGatherKernel ---
// Two-stage fused kernel with synchronized P2P gather:
//   Stage 1: cooperative thread loads copy one A row from the owner
//     rank's remote shard into a local staging buffer.
//   Stage 2: scalar matmul from staged A.  Skipped when n==0
//     (CUTLASS variants launch host-side CUTLASS instead).
// Owner signals readiness; non-owners wait.  After compute, readers
// ack completion; owner waits for all readers.
__global__ void threadloadGatherKernel(
    const __half* const* a_shards,
    __half* a_gathered,
    int32_t* const* ready_r, int32_t* ready_l,
    int32_t* const* done_r, int32_t* done_l,
    int64_t rank, int64_t ws, int32_t epoch_base,
    int64_t m, int64_t n, int64_t k,
    int64_t m_per_rank,
    const __half* b, __half* c) {
  const int32_t epoch = epoch_base + 1;
  for (int64_t row = blockIdx.x; row < m;
       row += gridDim.x) {
    int64_t owner = row / m_per_rank;
    int64_t lr = row - owner * m_per_rank;
    const __half* a = a_shards[owner];

    // --- Semaphore: owner signals readiness ---
    if (threadIdx.x == 0 && rank == owner)
      publishToAll(
          ready_r, ready_l, rank, row, m, ws, epoch);
    __syncthreads();
    if (threadIdx.x == 0 && rank != owner)
      waitOne(ready_l, row, m, owner, epoch);
    __syncthreads();

    // --- Stage 1: P2P gather via thread loads ---
    for (int64_t kk = threadIdx.x; kk < k;
         kk += blockDim.x)
      a_gathered[row * k + kk] = a[lr * k + kk];
    __syncthreads();

    // --- Stage 2: scalar matmul (skip when n==0) ---
    for (int64_t col = threadIdx.x; col < n;
         col += blockDim.x) {
      float acc = 0.f;
      for (int64_t kk = 0; kk < k; ++kk)
        acc += __half2float(a_gathered[row * k + kk]) *
            __half2float(b[kk * n + col]);
      c[row * n + col] = __float2half(acc);
    }
    __syncthreads();

    // --- Semaphore: readers ack, owner waits ---
    if (threadIdx.x == 0) {
      if (rank == owner)
        setLocal(done_l, rank, row, m, epoch);
      else
        publishToOne(done_r[owner], rank, row, m, epoch);
    }
    __syncthreads();
    if (threadIdx.x == 0 && rank == owner)
      waitAll(done_l, row, m, ws, epoch);
    __syncthreads();
  }
}

// --- 3c. multimemGatherKernel ---
// Two-stage fused kernel using Hopper multimem stores:
//   Stage 1: the owner rank writes each A row to a multicast buffer
//     via multimem.st.global.v4.f32 (hardware broadcast to all peers).
//   Stage 2: scalar matmul from multicast buffer.  Skipped when n==0.
// Requires SM90+ and multicast-capable symmetric memory.
__global__ void multimemGatherKernel(
    const __half* const* a_shards,
    __half* a_mc,
    int32_t* const* sem_r, int32_t* sem_l,
    int64_t rank, int64_t ws, int32_t epoch_base,
    int64_t m, int64_t n, int64_t k,
    int64_t m_per_rank,
    const __half* b, __half* c) {
  for (int64_t row = blockIdx.x; row < m;
       row += gridDim.x) {
    int64_t owner = row / m_per_rank;
    int64_t lr = row - owner * m_per_rank;
    const __half* a = a_shards[owner];
    __half* arow = a_mc + row * k;

    // --- Stage 1: multimem store (owner only) ---
    constexpr int64_t kVec = 8;
    int64_t nvec = k / kVec;
    if (rank == owner) {
      for (int64_t vi = threadIdx.x; vi < nvec;
           vi += blockDim.x) {
        uint4 val = reinterpret_cast<const uint4*>(
            a + lr * k)[vi];
#if __CUDA_ARCH__ >= 900
        asm volatile(
            "multimem.st.global.v4.f32 [%0],"
            " {%1, %2, %3, %4};"
            :
            : "l"((void*)(arow + vi * kVec)),
              "f"(__int_as_float((int)val.x)),
              "f"(__int_as_float((int)val.y)),
              "f"(__int_as_float((int)val.z)),
              "f"(__int_as_float((int)val.w))
            : "memory");
#else
        (void)val;
        asm volatile("trap;");
#endif
      }
      for (int64_t kk = nvec * kVec + threadIdx.x;
           kk < k; kk += blockDim.x)
        arow[kk] = a[lr * k + kk];
    }
    __syncthreads();

    // --- Semaphore barrier ---
#if __CUDA_ARCH__ >= 900
    const int32_t epoch = epoch_base + 1;
    if (threadIdx.x == 0 && rank == owner)
      publishToAll(
          sem_r, sem_l, rank, row, m, ws, epoch);
    __syncthreads();
    if (threadIdx.x == 0 && rank != owner)
      waitOne(sem_l, row, m, owner, epoch);
    __syncthreads();
#else
    (void)sem_r; (void)sem_l;
    (void)rank; (void)ws; (void)epoch_base;
    asm volatile("trap;");
#endif

    // --- Stage 2: scalar matmul (skip when n==0) ---
    for (int64_t col = threadIdx.x; col < n;
         col += blockDim.x) {
      float acc = 0.f;
      for (int64_t kk = 0; kk < k; ++kk)
        acc += __half2float(arow[kk]) *
            __half2float(b[kk * n + col]);
      c[row * n + col] = __float2half(acc);
    }
    __syncthreads();
  }
}

} // anonymous namespace

// =========================================================================
// Section 4: Public launcher functions
//
// Thin wrappers that set up grid/block dims and launch kernels once.
// Timing and iteration loops live in the .cpp file.
// =========================================================================

void launchNaiveRemoteRead(DistributedMatmulContext& ctx) {
  constexpr int64_t kB = 16;
  dim3 block(kB, kB);
  dim3 grid(
      (ctx.n + kB - 1) / kB, (ctx.m + kB - 1) / kB);
  naiveRemoteReadKernel<<<grid, block, 0, ctx.stream>>>(
      ctx.device_remote_ptrs,
      reinterpret_cast<const __half*>(
          ctx.b_full_half.data_ptr()),
      reinterpret_cast<__half*>(
          ctx.c_out_half.data_ptr()),
      ctx.m, ctx.n, ctx.k, ctx.m_per_rank);
}

void launchThreadloadGather(
    DistributedMatmulContext& ctx,
    int32_t epoch,
    bool compute) {
  dim3 block(256);
  dim3 grid(ctx.m);
  threadloadGatherKernel<<<grid, block, 0, ctx.stream>>>(
      ctx.device_remote_ptrs,
      reinterpret_cast<__half*>(
          ctx.a_gathered.data_ptr()),
      ctx.ready_sem_remote, ctx.ready_sem_local,
      ctx.done_sem_remote, ctx.done_sem_local,
      ctx.my_rank, ctx.world_size, epoch,
      ctx.m, compute ? ctx.n : 0,
      ctx.k, ctx.m_per_rank,
      reinterpret_cast<const __half*>(
          ctx.b_full_half.data_ptr()),
      reinterpret_cast<__half*>(
          ctx.c_out_half.data_ptr()));
}

void launchMultimemGather(
    DistributedMatmulContext& ctx,
    int32_t epoch,
    bool compute) {
  dim3 block(256);
  dim3 grid(ctx.m);
  multimemGatherKernel<<<grid, block, 0, ctx.stream>>>(
      ctx.device_remote_ptrs,
      ctx.multicast_ptr,
      ctx.stage_sem_remote, ctx.stage_sem_local,
      ctx.my_rank, ctx.world_size, epoch,
      ctx.m, compute ? ctx.n : 0,
      ctx.k, ctx.m_per_rank,
      reinterpret_cast<const __half*>(
          ctx.b_full_half.data_ptr()),
      reinterpret_cast<__half*>(
          ctx.c_out_half.data_ptr()));
}

at::Tensor matmulTma(
    const at::Tensor& a,
    const at::Tensor& b) {
  NVF_CHECK(a.is_cuda() && b.is_cuda());
  NVF_CHECK(a.dim() == 2 && b.dim() == 2);
  NVF_CHECK(a.size(1) == b.size(0));
  at::cuda::CUDAGuard guard{a.device()};
  auto* props =
      at::cuda::getDeviceProperties(a.get_device());
  NVF_CHECK(props->major >= 9, "Requires Hopper+.");
  int64_t m = a.size(0), n = b.size(1), k = a.size(1);
  at::Tensor out = at::empty(
      {m, n},
      at::TensorOptions()
          .dtype(a.scalar_type())
          .device(a.device()));
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(a.get_device());
#if defined(NVFUSER_ENABLE_CUTLASS)
  if (a.scalar_type() == at::ScalarType::Half)
    runGemmSm90<cutlass::half_t>(
        out, a, b, m, n, k, stream);
  else
    runGemmSm90<cutlass::bfloat16_t>(
        out, a, b, m, n, k, stream);
#else
  NVF_THROW("CUTLASS support required.");
#endif
  return out;
}

bool canRunCutlassCompute(
    const at::Tensor& a,
    const at::Tensor& b) {
  if (!hasValidTmaShape(a, b)) return false;
#if !defined(NVFUSER_ENABLE_CUTLASS) || \
    !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  return false;
#else
  auto* props =
      at::cuda::getDeviceProperties(a.get_device());
  return props->major == 9 && props->minor == 0;
#endif
}

const char* implName(DistributedMatmulImpl impl) {
  switch (impl) {
    case DistributedMatmulImpl::baselineNcclAllgatherMatmul:
      return "baselineNcclAllgatherMatmul";
    case DistributedMatmulImpl::baselineCudaAllgatherMatmul:
      return "baselineCudaAllgatherMatmul";
    case DistributedMatmulImpl::naiveRemoteRead:
      return "naiveRemoteRead";
    case DistributedMatmulImpl::threadloadGatherScalarCompute:
      return "threadloadGatherScalarCompute";
    case DistributedMatmulImpl::multimemGatherScalarCompute:
      return "multimemGatherScalarCompute";
    case DistributedMatmulImpl::threadloadGatherCutlassCompute:
      return "threadloadGatherCutlassCompute";
    case DistributedMatmulImpl::multimemGatherCutlassCompute:
      return "multimemGatherCutlassCompute";
  }
  return "unknown";
}

bool isMulticastSupported(int64_t device_id) {
  int val = 0;
  auto r = cuDeviceGetAttribute(
      &val,
      CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
      static_cast<CUdevice>(device_id));
  return r == CUDA_SUCCESS && val != 0;
}

} // namespace nvfuser
