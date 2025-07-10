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

namespace {

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
// Kernel configuration traits for different output data types
// Defines tile shapes and cluster configurations.
template <typename T>
struct KernelTraits;

// Kernel traits for FP16 output
template <>
struct KernelTraits<cutlass::half_t> {
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<_4, _4, _1>;
  using PerSmTileShape_MNK = Shape<_128, _256, _256>;
};

// Kernel traits for BF16 output
template <>
struct KernelTraits<cutlass::bfloat16_t> {
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<_4, _4, _1>;
  using PerSmTileShape_MNK = Shape<_128, _256, _256>;
};

// Main GEMM configuration for NVFP4 scaled matrix multiplication on SM100+
// Defines all the types, layouts, and configurations needed for the CUTLASS
// kernel
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

// Constructs CUTLASS GEMM arguments from PyTorch tensors and dimensions
//
// This function converts PyTorch tensor data and metadata into the format
// expected by CUTLASS GEMM kernels, including proper stride calculations
// and layout configurations for the scaled matrix multiplication.
//
// Parameters:
//   output: Output tensor for storing results
//   a: Input matrix A in NVFP4 format
//   b: Input matrix B in NVFP4 format
//   scales_a: Per-block scaling factors for matrix A
//   scales_b: Per-block scaling factors for matrix B
//   alpha: Global scaling factor
//   M, N, K: Matrix dimensions
//
// Returns: CUTLASS GEMM arguments structure ready for kernel execution
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

// Executes the FP4 scaled matrix multiplication using CUTLASS kernels
//
// This function orchestrates the GEMM operation by setting up the kernel,
// allocating workspace memory, and running the computation on the GPU.
// It handles the complete lifecycle from kernel initialization to execution.
//
// Parameters:
//   output: Output tensor to store the result
//   a, b: Input matrices in FP4 format
//   scales_a, scales_b: Per-block scaling factors
//   alpha: Global scaling factor
//   m, n, k: Matrix dimensions
//   stream: CUDA stream for asynchronous execution
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
// Fallback implementation for unsupported CUTLASS versions
// Throws an error when SM100+ CUTLASS support is not available
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
  NVF_THROW("Unsupported CUTLASS version.");
}
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Helper function to round up to the nearest multiple of y
inline int64_t roundUp(int64_t x, int64_t y) {
  return (x + y - 1) / y * y;
}

// Validates all input parameters and tensor properties for NVFP4 scaled matrix
// multiplication
//
// This function performs comprehensive validation of input tensors including:
// - CUDA device and contiguity checks
// - Data type validation for all inputs
// - Matrix dimension and shape compatibility
// - Alignment requirements for optimal performance
// - Scale matrix shape validation
//
// Parameters:
//   a, b: Input matrices to validate
//   scales_a, scales_b: Scale matrices to validate
//   alpha: Alpha scaling factor to validate
//
// Returns: Tuple of (m, n, k) dimensions for the GEMM operation
//
// Throws: NVF_CHECK exceptions for any validation failures
std::tuple<int64_t, int64_t, int64_t> validateInputs(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& alpha) {
  // Check CUDA device and contiguity for all input tensors
  for (const torch::Tensor& t : {a, b, scales_a, scales_b, alpha}) {
    NVF_CHECK(
        t.is_cuda() && t.is_contiguous(),
        "Input argument must be a CUDA tensor and contiguous.")
  }

  // Validate data types
  NVF_CHECK(
      a.scalar_type() == at::ScalarType::Float4_e2m1fn_x2,
      "Expected Float4_e2m1fn_x2 for Operand A.")
  NVF_CHECK(
      b.scalar_type() == at::ScalarType::Float4_e2m1fn_x2,
      "Expected Float4_e2m1fn_x2 for Operand B.")
  NVF_CHECK(
      scales_a.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "Expected FP8_E4M3 for Blockscale scale_a.")
  NVF_CHECK(
      scales_b.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "Expected FP8_E4M3 for Blockscale scale_b.")
  NVF_CHECK(
      alpha.scalar_type() == at::ScalarType::Float,
      "Expected FP32 for alpha scalar.")

  // Validate matrix dimensions
  NVF_CHECK(a.dim() == 2, "Operand A must be a matrix.");
  NVF_CHECK(b.dim() == 2, "Operand B must be a matrix.");
  NVF_CHECK(
      a.sizes()[1] == b.sizes()[1],
      "A and B shapes cannot be multiplied (",
      a.sizes()[0],
      ",",
      a.sizes()[1],
      " and ",
      b.sizes()[0],
      ",",
      b.sizes()[1],
      ")");

  const int64_t m = a.sizes()[0];
  const int64_t n = b.sizes()[0];
  const int64_t k = a.sizes()[1] * 2;

  // Check alignment requirements
  constexpr int64_t alignment = 32;
  NVF_CHECK(
      k % alignment == 0,
      "The K dimension",
      k,
      "is not divisible by ",
      alignment)
  NVF_CHECK(
      n % alignment == 0,
      "The N dimension",
      n,
      "is not divisible by ",
      alignment)

  // Calculate rounded dimensions for scale matrix validation
  int64_t rounded_m = roundUp(m, 128);
  int64_t rounded_n = roundUp(n, 128);
  int64_t rounded_k = roundUp(k / 16, 4);

  // Validate scale matrix properties
  NVF_CHECK(scales_a.dim() == 2, "Blockscale scale_a must be a matrix.");
  NVF_CHECK(scales_b.dim() == 2, "Blockscale scale_b must be a matrix.");
  NVF_CHECK(
      scales_a.sizes()[1] == scales_b.sizes()[1],
      "scale_a and scale_b shapes cannot be multiplied because the inner-most "
      "dimensions are not equal.")
  NVF_CHECK(
      scales_a.sizes()[0] == rounded_m && scales_a.sizes()[1] == rounded_k,
      "scale_a must be padded and swizzled to a shape (",
      rounded_m,
      ",",
      rounded_k,
      "), but got a shape (",
      scales_a.sizes()[0],
      ",",
      scales_a.sizes()[1],
      ")");
  NVF_CHECK(
      scales_b.sizes()[0] == rounded_n && scales_b.sizes()[1] == rounded_k,
      "scale_b must be padded and swizzled to a shape (",
      rounded_n,
      ",",
      rounded_k,
      "), but got a shape (",
      scales_b.sizes()[0],
      ",",
      scales_b.sizes()[1],
      ")");

  return {m, n, k};
}

} // namespace

torch::Tensor nvfp4_scaled_mm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& alpha,
    const at::ScalarType out_dtype) {
  // Validate all inputs and get matrix dimensions
  auto [m, n, k] = validateInputs(a, b, scales_a, scales_b, alpha);

  at::cuda::CUDAGuard device_guard{(int8_t)a.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device());

  auto options =
      at::TensorOptions().dtype(out_dtype).device(at::kCUDA, a.get_device());
  torch::Tensor output = at::empty({a.sizes()[0], b.sizes()[0]}, options);

  if (out_dtype == at::ScalarType::Half) {
    runGemm<cutlass::half_t>(
        output, a, b, scales_a, scales_b, alpha, m, n, k, stream);
  } else if (out_dtype == at::ScalarType::BFloat16) {
    runGemm<cutlass::bfloat16_t>(
        output, a, b, scales_a, scales_b, alpha, m, n, k, stream);
  } else {
    NVF_THROW("Unsupported output data type of nvfp4 scaled_mm.");
  }
  return output;
}

} // namespace nvfuser::cutlass_kernels
