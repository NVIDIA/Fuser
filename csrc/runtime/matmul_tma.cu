// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <runtime/matmul_tma.h>

#include <exceptions.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

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

#if defined(NVFUSER_ENABLE_CUTLASS) && defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
using namespace cute;

template <typename ElementT>
struct MatmulTmaSm90 {
  using ElementA = ElementT;
  using ElementB = ElementT;
  using ElementC = ElementT;
  using ElementD = ElementT;

  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::RowMajor;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;

  static constexpr int kAlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int kAlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int kAlignmentC =
      128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int kAlignmentD =
      128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using MmaTileShape = Shape<_128, _128, _64>;
  using ClusterShape = Shape<_1, _1, _1>;
  using PerSmTileShape_MNK = Shape<_128, _128, _64>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          PerSmTileShape_MNK,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementAccumulator,
          ElementC,
          LayoutCTag,
          kAlignmentC,
          ElementD,
          LayoutDTag,
          kAlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementA,
          LayoutATag,
          kAlignmentA,
          ElementB,
          LayoutBTag,
          kAlignmentB,
          ElementAccumulator,
          MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

template <typename ElementT>
typename MatmulTmaSm90<ElementT>::Gemm::Arguments buildArguments(
    at::Tensor& output,
    const at::Tensor& a,
    const at::Tensor& b,
    int64_t m,
    int64_t n,
    int64_t k) {
  using Config = MatmulTmaSm90<ElementT>;
  using ElementA = typename Config::ElementA;
  using ElementB = typename Config::ElementB;
  using ElementD = typename Config::ElementD;
  using StrideA = typename Config::StrideA;
  using StrideB = typename Config::StrideB;
  using StrideC = typename Config::StrideC;
  using StrideD = typename Config::StrideD;

  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {static_cast<int>(m), static_cast<int>(k), 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {static_cast<int>(k), static_cast<int>(n), 1});
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {static_cast<int>(m), static_cast<int>(n), 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {static_cast<int>(m), static_cast<int>(n), 1});

  typename Config::GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementA*>(a.data_ptr()),
      stride_a,
      static_cast<const ElementB*>(b.data_ptr()),
      stride_b};

  typename Config::GemmKernel::EpilogueArguments epilogue_args{
      {}, // epilogue.thread
      nullptr,
      stride_c,
      static_cast<ElementD*>(output.data_ptr()),
      stride_d};

  typename Config::GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1},
      mainloop_args,
      epilogue_args};

  return args;
}

template <typename ElementT>
void runMatmulSm90(
    at::Tensor& output,
    const at::Tensor& a,
    const at::Tensor& b,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  using Config = MatmulTmaSm90<ElementT>;
  typename Config::Gemm gemm;
  auto args = buildArguments<ElementT>(output, a, b, m, n, k);

  size_t workspace_size = Config::Gemm::get_workspace_size(args);
  auto workspace_options =
      at::TensorOptions().dtype(at::kByte).device(a.device());
  auto workspace =
      at::empty({static_cast<int64_t>(workspace_size)}, workspace_options);

  auto can_implement_status = gemm.can_implement(args);
  NVF_CHECK(
      can_implement_status == cutlass::Status::kSuccess,
      "TMA GEMM cannot be implemented for the given inputs.");

  auto status = gemm.initialize(args, workspace.data_ptr(), stream);
  NVF_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM.");

  status = gemm.run(
      args,
      workspace.data_ptr(),
      stream,
      /*cuda_adapter=*/nullptr,
      /*launch_with_pdl=*/true);
  NVF_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM.");
}
#else
template <typename ElementT>
void runMatmulSm90(
    at::Tensor& output,
    const at::Tensor& a,
    const at::Tensor& b,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  NVF_THROW("CUTLASS SM90 support is required for TMA matmul.");
}
#endif // NVFUSER_ENABLE_CUTLASS && CUTLASS_ARCH_MMA_SM90_SUPPORTED

void validateInputs(const at::Tensor& a, const at::Tensor& b) {
  NVF_CHECK(a.is_cuda(), "Expected CUDA tensor for operand A.");
  NVF_CHECK(b.is_cuda(), "Expected CUDA tensor for operand B.");
  NVF_CHECK(a.dim() == 2, "Operand A must be rank-2.");
  NVF_CHECK(b.dim() == 2, "Operand B must be rank-2.");
  NVF_CHECK(
      a.scalar_type() == b.scalar_type(),
      "Operands A and B must have the same dtype.");
  NVF_CHECK(
      a.scalar_type() == at::ScalarType::Half ||
          a.scalar_type() == at::ScalarType::BFloat16,
      "Only Half and BFloat16 are supported.");
  NVF_CHECK(
      a.is_contiguous() && b.is_contiguous(),
      "Operands must be contiguous row-major tensors.");
  NVF_CHECK(
      a.size(1) == b.size(0),
      "Mismatched matmul dimensions: A[K] must match B[K].");
  NVF_CHECK(
      a.get_device() == b.get_device(),
      "Operands must be on the same CUDA device.");

  constexpr int64_t kAlignment = 8;
  NVF_CHECK(
      a.size(1) % kAlignment == 0,
      "K dimension must be a multiple of 8 for TMA alignment.");
  NVF_CHECK(
      b.size(1) % kAlignment == 0,
      "N dimension must be a multiple of 8 for TMA alignment.");
}

} // namespace

at::Tensor matmulTma(const at::Tensor& a, const at::Tensor& b) {
  validateInputs(a, b);
  at::cuda::CUDAGuard device_guard{a.device()};
  auto* props = at::cuda::getDeviceProperties(a.get_device());
  NVF_CHECK(
      props->major == 9 && props->minor == 0,
      "TMA matmul requires SM90 (Hopper).");

  const int64_t m = a.size(0);
  const int64_t n = b.size(1);
  const int64_t k = a.size(1);

  auto options =
      at::TensorOptions().dtype(a.scalar_type()).device(a.device());
  at::Tensor output = at::empty({m, n}, options);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device());

#if defined(NVFUSER_ENABLE_CUTLASS)
  if (a.scalar_type() == at::ScalarType::Half) {
    runMatmulSm90<cutlass::half_t>(output, a, b, m, n, k, stream);
  } else {
    runMatmulSm90<cutlass::bfloat16_t>(output, a, b, m, n, k, stream);
  }
#else
  NVF_THROW("CUTLASS support is required for TMA matmul.");
#endif

  return output;
}

} // namespace nvfuser
