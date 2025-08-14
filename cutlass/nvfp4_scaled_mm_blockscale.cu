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

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/packed_stride.hpp"

namespace nvfuser::cutlass_kernels {

namespace {

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
// Kernel configuration traits for different output data types
// Defines tile shapes and cluster configurations.
template <typename T>
struct KernelTraits;

// Kernel traits for NVFP4 output
template <>
struct KernelTraits<cutlass::float_e2m1_t> {
  using MmaTileShape = Shape<_128, _128, _256>;
  using ClusterShape = Shape<_1, _1, _1>;
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
  using ElementD = cutlass::float_e2m1_t;
  using ElementC = cutlass::float_e2m1_t;
  using ElementSFD = cutlass::float_e4m3_t;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  using LayoutSFDTag = LayoutDTag;

  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  // Kernel functional config
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  // Kernel Perf config
  using MmaTileShape = typename KernelTraits<T>::MmaTileShape;
  using ClusterShape = typename KernelTraits<T>::ClusterShape;

  static constexpr int InputSFVectorSize = 16;
  static constexpr int OutputSFVectorSize = InputSFVectorSize;

  // D = alpha * acc + beta * C
  // With BlockScaleFactor generation.
  using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
      OutputSFVectorSize,
      ElementD,
      ElementCompute,
      ElementSFD,
      LayoutSFDTag,
      ElementC>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          MmaTileShape,
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
          cutlass::epilogue::collective::EpilogueScheduleAuto,
          FusionOperation>::CollectiveOp;

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

  // Reference device GEMM implementation type
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

  using FusionOp = typename Gemm::EpilogueOutputOp;
  static constexpr bool IsBlockScaleSupported = FusionOp::IsBlockScaleSupported;
  using SfdOutputCfg =
      cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;
  using LayoutSFD = typename SfdOutputCfg::LayoutSF;
};

// Constructs CUTLASS GEMM arguments from PyTorch tensors and dimensions
//
// This function converts PyTorch tensor data and metadata into the format
// expected by CUTLASS GEMM kernels, including proper stride calculations
// and layout configurations for the scaled matrix multiplication.
//
// Parameters:
//   output: Output tensor for storing matmul results
//   output_blockscale: Output tensor to store nvfp4 blockscale factor
//   a: Input matrix A in NVFP4 format
//   b: Input matrix B in NVFP4 format
//   scales_a: Per-block scaling factors for matrix A
//   scales_b: Per-block scaling factors for matrix B
//   alpha: Global scaling factor for operands A and B
//   global_normconst: Global scaling factor for output blockscaling
//   M, N, K: Matrix dimensions
//
// Returns: CUTLASS GEMM arguments structure ready for kernel execution
template <typename T>
typename T::Gemm::Arguments args_from_options(
    at::Tensor& output,
    at::Tensor& output_blockscale,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scales_a,
    const at::Tensor& scales_b,
    const at::Tensor& alpha,
    const at::Tensor& global_normconst,
    int64_t M,
    int64_t N,
    int64_t K) {
  using ElementA = typename T::Gemm::ElementA;
  using ElementB = typename T::Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementD = typename T::Gemm::ElementD;
  using ElementSFD = cutlass::float_e4m3_t;
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

  if constexpr (T::IsBlockScaleSupported) {
    arguments.epilogue.thread.block_scale_factor_ptr =
        static_cast<ElementSFD*>(output_blockscale.data_ptr());
    // Matrix-wide normalization constant
    arguments.epilogue.thread.norm_constant_ptr =
        static_cast<ElementCompute const*>(global_normconst.data_ptr());
  }
  return arguments;
}

// Executes the FP4 scaled matrix multiplication using CUTLASS kernels
//
// This function orchestrates the GEMM operation by setting up the kernel,
// allocating workspace memory, and running the computation on the GPU.
// It handles the complete lifecycle from kernel initialization to execution.
//
// Parameters:
//   output: Output tensor to store the matmul result
//   output_blockscale: Output tensor to store nvfp4 blockscale factor
//   a, b: Input matrices in FP4 format
//   scales_a, scales_b: Per-block scaling factors
//   alpha: Global scaling factor
//   global_normconst: Global scaling factor for output blockscaling
//   m, n, k: Matrix dimensions
//   stream: CUDA stream for asynchronous execution
template <typename T>
void runGemm(
    at::Tensor& output,
    at::Tensor& output_blockscale,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scales_a,
    const at::Tensor& scales_b,
    const at::Tensor& alpha,
    const at::Tensor& global_normconst,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  typename Fp4GemmSm100<T>::Gemm gemm;

  auto arguments = args_from_options<Fp4GemmSm100<T>>(
      output,
      output_blockscale,
      a,
      b,
      scales_a,
      scales_b,
      alpha,
      global_normconst,
      m,
      n,
      k);

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
    at::Tensor& output_blockscale,
    at::Tensor const& a,
    at::Tensor const& b,
    at::Tensor const& scales_a,
    at::Tensor const& scales_b,
    at::Tensor const& alpha,
    at::Tensor const& global_normconst,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  NVF_THROW("Unsupported CUTLASS version.");
}
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

} // namespace

std::pair<torch::Tensor, torch::Tensor> nvfp4_scaled_mm_blockscale(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& alpha,
    const torch::Tensor& global_normconst,
    bool skip_checks) {
  // Validate all inputs and get matrix dimensions
  auto [m, n, k] =
      validateInputsNvfp4ScaledMm(a, b, scales_a, scales_b, alpha, skip_checks);

  at::cuda::CUDAGuard device_guard{(int8_t)a.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device());

  auto output_options = at::TensorOptions()
                            .dtype(at::ScalarType::Float4_e2m1fn_x2)
                            .device(at::kCUDA, a.get_device());
  // Only half the number of elements in inner-most dimension for packed format.
  torch::Tensor output = at::empty({m, roundUp(n / 2)}, output_options);

  auto blockscale_options = at::TensorOptions()
                                .dtype(at::ScalarType::Float8_e4m3fn)
                                .device(at::kCUDA, a.get_device());
  torch::Tensor output_blockscale =
      at::empty({m, roundUp(n / 16, 4)}, blockscale_options);

  runGemm<cutlass::float_e2m1_t>(
      output,
      output_blockscale,
      a,
      b,
      scales_a,
      scales_b,
      alpha,
      global_normconst,
      m,
      n,
      k,
      stream);
  return std::make_pair(output, output_blockscale);
}

} // namespace nvfuser::cutlass_kernels
