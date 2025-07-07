// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cutlass_utils.h>
#include <exceptions.h>
#include <nvf_cutlass.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace nvfuser::cutlass_kernels {

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
// Kernel Perf config
template <typename T>
struct KernelTraits;

template <>
struct KernelTraits<float> {
  using MmaTileShape = Shape<_128, _128, _256>;
  using ClusterShape = Shape<_1, _1, _1>;
  using PerSmTileShape_MNK = Shape<_128, _128, _256>;
};

template <>
struct KernelTraits<cutlass::half_t> {
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<_4, _4, _1>;
  using PerSmTileShape_MNK = Shape<_128, _256, _256>;
};

template <>
struct KernelTraits<cutlass::bfloat16_t> {
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<_4, _4, _1>;
  using PerSmTileShape_MNK = Shape<_128, _256, _256>;
};

template <typename T>
struct Fp4GemmSm100 {
  // A matrix configuration
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  // B matrix configuration
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  // C/D matrix configuration
  using ElementD = T;
  using ElementC = T;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  // Kernel functional config
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  // Kernel Perf config
  using MmaTileShape = typename KernelTraits<T>::MmaTileShape;
  using ClusterShape = typename KernelTraits<T>::ClusterShape;
  using PerSmTileShape_MNK = typename KernelTraits<T>::PerSmTileShape_MNK;

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
          AlignmentC,
          ElementD,
          LayoutDTag,
          AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementA,
          LayoutATag,
          AlignmentA,
          ElementB,
          LayoutBTag,
          AlignmentB,
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
  using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));
};

template <typename T>
typename T::Gemm::Arguments args_from_options(
    at::Tensor& output,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scales_a,
    const at::Tensor& scales_b,
    const at::Tensor& alpha,
    int64_t M,
    int64_t N,
    int64_t K) {
  using ElementA = typename T::Gemm::ElementA;
  using ElementB = typename T::Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementD = typename T::Gemm::ElementD;
  using ElementCompute = float;
  using StrideA = typename T::StrideA;
  using StrideB = typename T::StrideB;
  using StrideD = typename T::StrideD;
  using Sm1xxBlkScaledConfig =
      typename T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(m, n, k, 1));

  typename T::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<ElementA const*>(a.data_ptr()),
       stride_A,
       static_cast<ElementB const*>(b.data_ptr()),
       stride_B,
       static_cast<ElementSFA const*>(scales_a.data_ptr()),
       layout_SFA,
       static_cast<ElementSFB const*>(scales_b.data_ptr()),
       layout_SFB},
      {// Epilogue arguments
       {}, // epilogue.thread
       static_cast<ElementD const*>(output.data_ptr()),
       stride_D,
       static_cast<ElementD*>(output.data_ptr()),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());
  return arguments;
}

template <typename T>
void runGemm(
    at::Tensor& output,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scales_a,
    const at::Tensor& scales_b,
    const at::Tensor& alpha,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  typename Fp4GemmSm100<T>::Gemm gemm;

  auto arguments = args_from_options<Fp4GemmSm100<T>>(
      output, a, b, scales_a, scales_b, alpha, m, n, k);

  size_t workspace_size = Fp4GemmSm100<T>::Gemm::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  auto can_implement_status = gemm.can_implement(arguments);
  NVF_CHECK(
      can_implement_status == cutlass::Status::kSuccess,
      "Failed to implement GEMM");

  auto status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  NVF_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm.run(arguments, workspace.data_ptr(), stream);
  NVF_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}
#else
template <typename T>
void runGemm(
    at::Tensor& output,
    at::Tensor const& a,
    at::Tensor const& b,
    at::Tensor const& scales_a,
    at::Tensor const& scales_b,
    at::Tensor const& alpha,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  NVF_CHECK(
      false,
      "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
      "a CUTLASS 3.8 source directory to enable support.");
}
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#define CHECK_TYPE(x, st, m) \
  NVF_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type:", m)
#define CHECK_TH_CUDA(x, m) NVF_CHECK(x.is_cuda(), m, "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) \
  NVF_CHECK(x.is_contiguous(), m, "must be contiguous")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)

constexpr auto FLOAT4_E2M1X2 = at::ScalarType::Byte;
constexpr auto SF_DTYPE = at::ScalarType::Float8_e4m3fn;

void nvfp4_scaled_mm_assert(
    at::ScalarType out_dtype,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& scales_a,
    torch::Tensor const& scales_b,
    torch::Tensor const& alpha) {
  CHECK_INPUT(a, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(b, FLOAT4_E2M1X2, "b");

  CHECK_INPUT(scales_a, SF_DTYPE, "scale_a");
  CHECK_INPUT(scales_b, SF_DTYPE, "scale_b");

  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");

  NVF_CHECK(a.dim() == 2, "a must be a matrix");
  NVF_CHECK(b.dim() == 2, "b must be a matrix");
  NVF_CHECK(
      a.sizes()[1] == b.sizes()[1],
      "a and b shapes cannot be multiplied (",
      a.sizes()[0],
      "x",
      a.sizes()[1],
      " and ",
      b.sizes()[0],
      "x",
      b.sizes()[1],
      ")");

  auto const m = a.sizes()[0];
  auto const n = b.sizes()[0];
  auto const k = a.sizes()[1] * 2;

  constexpr int alignment = 32;
  NVF_CHECK(
      k % alignment == 0,
      "Expected k to be divisible by ",
      alignment,
      ", but got a shape: (",
      a.sizes()[0],
      "x",
      a.sizes()[1],
      "), k: ",
      k,
      ".");
  NVF_CHECK(
      n % alignment == 0,
      "Expected n to be divisible by ",
      alignment,
      ", but got b shape: (",
      b.sizes()[0],
      "x",
      b.sizes()[1],
      ").");

  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  // Since k is divisible by 32 (alignment), k / 16 is guaranteed to be an
  // integer.
  int rounded_k = round_up(k / 16, 4);
  
  NVF_CHECK(scales_a.dim() == 2, "scale_a must be a matrix");
  NVF_CHECK(scales_b.dim() == 2, "scale_b must be a matrix");
  NVF_CHECK(
      scales_a.sizes()[1] == scales_b.sizes()[1],
      "scale_a and scale_b shapes cannot be multiplied (",
      scales_a.sizes()[0],
      "x",
      scales_a.sizes()[1],
      " and ",
      scales_b.sizes()[0],
      "x",
      scales_b.sizes()[1],
      ")");
  NVF_CHECK(
      scales_a.sizes()[0] == rounded_m && scales_a.sizes()[1] == rounded_k,
      "scale_a must be padded and swizzled to a shape (",
      rounded_m,
      "x",
      rounded_k,
      "), but got a shape (",
      scales_a.sizes()[0],
      "x",
      scales_a.sizes()[1],
      ")");
  NVF_CHECK(
      scales_b.sizes()[0] == rounded_n && scales_b.sizes()[1] == rounded_k,
      "scale_b must be padded and swizzled to a shape (",
      rounded_n,
      "x",
      rounded_k,
      "), but got a shape (",
      scales_b.sizes()[0],
      "x",
      scales_b.sizes()[1],
      ")");

  NVF_CHECK(out_dtype == at::ScalarType::Half ||
 out_dtype == at::ScalarType::BFloat16 ||
 out_dtype == at::ScalarType::Float, "unsupported dtype on output: ", out_dtype)
}

bool nvfp4_scaled_mm_check(
    at::ScalarType out_dtype,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& scales_a,
    torch::Tensor const& scales_b,
    torch::Tensor const& alpha) {
  try {
    nvfp4_scaled_mm_assert(out_dtype, a, b, scales_a, scales_b, alpha);
    return true;
  } catch (...) {
    return false;
  }
}
void nvfp4_scaled_mm(
    torch::Tensor& output,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& scales_a,
    torch::Tensor const& scales_b,
    torch::Tensor const& alpha) {
  auto out_dtype = output.scalar_type();
  nvfp4_scaled_mm_assert(out_dtype, a, b, scales_a, scales_b, alpha);

  auto const m = a.sizes()[0];
  auto const n = b.sizes()[0];
  auto const k = a.sizes()[1] * 2;

  at::cuda::CUDAGuard device_guard{(int8_t)a.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device());

  if (out_dtype == at::ScalarType::Half) {
    runGemm<cutlass::half_t>(
        output, a, b, scales_a, scales_b, alpha, m, n, k, stream);
  } else if (out_dtype == at::ScalarType::BFloat16) {
    runGemm<cutlass::bfloat16_t>(
        output, a, b, scales_a, scales_b, alpha, m, n, k, stream);
  } else if (out_dtype == at::ScalarType::Float) {
    runGemm<float>(output, a, b, scales_a, scales_b, alpha, m, n, k, stream);
  } else {
    NVF_CHECK(false, "Unsupported output data type of nvfp4 mm");
  }
}

} // namespace nvfuser::cutlass_kernels
