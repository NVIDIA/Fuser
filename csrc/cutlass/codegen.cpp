// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <exceptions.h>
#include <fusion.h>
#include <runtime/cutlass_compiled_kernel.h>
#include <type.h>

#include <format>
#include <string>

namespace nvfuser {

namespace cutlass_codegen {

namespace {

std::string mapDataTypeToCutlass(DataType dtype) {
  if (dtype == DataType::Float) {
    return "cutlass::half_t";
  } else if (dtype == DataType::Half) {
    return "cutlass::half_t";
  } else if (dtype == DataType::BFloat16) {
    return "cutlass::bfloat16_t";
  } else {
    return "float";
  }
}

std::string mapLayoutToCutlass(const TensorView* tv) {
  // Map nvFuser layout to CUTLASS layout
  return "cutlass::layout::RowMajor";
}

std::string dtypeToCutlass(const DataType& dtype) {
  NVF_ERROR(std::holds_alternative<PrimDataType>(dtype.type));
  switch (std::get<PrimDataType>(dtype.type)) {
    case (DataType::Half):
      return "cutlass::half_t";
    case (DataType::BFloat16):
      return "cutlass::bfloat16_t";
    default:
      return "UNKNOWN_DTYPE";
  }
}

std::string generateNvfp4ScaledMmKernel(Fusion* fusion) {
  NVF_ERROR(fusion != nullptr);
  NVF_ERROR_EQ(
      fusion->outputs().size(),
      1,
      "Cutlass executor currently only supports a single output");
  auto* main_output = fusion->outputs().front()->as<TensorView>();
  const std::string output_dtype = dtypeToCutlass(main_output->dtype());

  NVF_ERROR_GE(fusion->inputs().size(), 4);
  auto* a = fusion->inputs()[0]->as<TensorView>();
  auto* b = fusion->inputs()[1]->as<TensorView>();
  auto* a_scale = fusion->inputs()[2]->as<TensorView>();
  auto* b_scale = fusion->inputs()[3]->as<TensorView>();

  std::string code =
      R"(
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#define NVF_THROW(msg) throw std::runtime_error(msg);
#define NVF_ERROR(cond, msg)                                      \
  if (!(cond)) {                                                  \
    NVF_THROW("Condition " #cond " failed: " + std::string(msg)); \
  }

#define NVFUSER_CUDA_RT_SAFE_CALL(x)               \
  do {                                             \
    cudaError_t _result = x;                       \
    NVF_ERROR(                                     \
        _result == cudaSuccess,                    \
        std::string("CUDA error: ") +              \
        std::string(cudaGetErrorName(_result)) +   \
        std::string(" failed with error ") +       \
        std::string(cudaGetErrorString(_result))); \
  } while (0)

namespace nvfuser::cutlass_kernels {

namespace {
using namespace cute;

// Kernel configuration traits for different output data types
// Defines tile shapes and cluster configurations.
struct KernelTraits {
)";
  code += std::format(
      R"(
  using MmaTileShape = Shape<_{}, _{}, _{}>;
  using ClusterShape = Shape<_{}, _{}, _{}>;
  using PerSmTileShape_MNK = Shape<_{}, _{}, _{}>;
)",
      // TODO: Accept heuristic parameters and use them here
        256,
        256,
        256,
        2,
        1,
        1,
        128,
        256,
        256
        /*params.mma_tile.m,
          params.mma_tile.n,
          params.mma_tile.k,
          params.cluster_shape.m,
          params.cluster_shape.n,
          params.cluster_shape.k,
          params.per_sm_tile.m,
          params.per_sm_tile.n,
          params.per_sm_tile.k
          */);

  code += R"(
};

// Main GEMM configuration for NVFP4 scaled matrix multiplication on SM100+
// Defines all the types, layouts, and configurations needed for the CUTLASS
// kernel
struct Fp4GemmSm100 {
  // A matrix configuration
)";
  NVF_ERROR_EQ(
      a->dtype(),
      DataType::Float4_e2m1fn,
      "Only float_e2m1_t is supported so far");
  code += "  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;\n";
  if (a->getLogicalDomain().back() == a->getMaybeAllocationDomain().back()) {
    code += "  using LayoutATag = cutlass::layout::RowMajor;\n";
  } else {
    code += "  using LayoutATag = cutlass::layout::ColumnMajor;\n";
  }
  // TODO: check alignment of A and save in cutlass_params.supported_vec_sizes
  // as is done for Ampere
  code += R"(
  static constexpr int AlignmentA = 32;

  // B matrix configuration
)";
  NVF_ERROR_EQ(
      b->dtype(),
      DataType::Float4_e2m1fn,
      "Only float_e2m1_t is supported so far");
  code += "  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;\n";
  if (b->getLogicalDomain().back() == b->getMaybeAllocationDomain().back()) {
    code += "  using LayoutBTag = cutlass::layout::RowMajor;\n";
  } else {
    code += "  using LayoutBTag = cutlass::layout::ColumnMajor;\n";
  }
  // TODO: check alignment of B and save in cutlass_params.supported_vec_sizes
  // as is done for Ampere
  code += R"(
  static constexpr int AlignmentB = 32;

  // C/D matrix configuration
)";
  code += "  using ElementD = " + output_dtype + ";\n";
  code += "  using ElementC = " + output_dtype + ";\n";

  NVF_ERROR(
      !main_output->hasAllocation(),
      "Cutlass executor doesn't yet support transposed output");
  // TODO: support transposed output by changing the below lines as needed
  code += "  using LayoutDTag = cutlass::layout::RowMajor;\n";
  code += "  using LayoutCTag = cutlass::layout::RowMajor;\n";

  code += R"(
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  // Kernel functional config
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  // Kernel Perf config
  using MmaTileShape = typename KernelTraits::MmaTileShape;
  using ClusterShape = typename KernelTraits::ClusterShape;
  using PerSmTileShape_MNK = typename KernelTraits::PerSmTileShape_MNK;

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
)";
  NVF_ERROR_EQ(a_scale->dtype(), DataType::Float8_e4m3fn);
  code += "  using ElementSFA = cutlass::float_ue4m3_t;\n";
  NVF_ERROR_EQ(b_scale->dtype(), DataType::Float8_e4m3fn);
  code += "  using ElementSFB = cutlass::float_ue4m3_t;\n";

  code += R"(
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
  typename Fp4GemmSm100::Gemm gemm;

  auto arguments = args_from_options<Fp4GemmSm100>(
      output, a, b, scales_a, scales_b, alpha, m, n, k);

  size_t workspace_size = Fp4GemmSm100::Gemm::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  auto can_implement_status = gemm.can_implement(arguments);
  NVF_ERROR(
      can_implement_status == cutlass::Status::kSuccess,
      "Failed to implement GEMM");

  auto status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  NVF_ERROR(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm.run(arguments, workspace.data_ptr(), stream);
  NVF_ERROR(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

} // namespace

// Main kernel function that will be called from nvFuser
extern "C" void run_kernel(
    const std::vector<void*>& args,
    cudaStream_t stream) {

  // TODO: determine this ordering from fusion inputs and pattern matching
  NVF_ERROR(args.size() == 6, "Expected 6 arguments");
  const at::Tensor& a = *reinterpret_cast<at::Tensor*>(args[0]);
  const at::Tensor& b = *reinterpret_cast<at::Tensor*>(args[1]);
  const at::Tensor& scales_a = *reinterpret_cast<at::Tensor*>(args[2]);
  const at::Tensor& scales_b = *reinterpret_cast<at::Tensor*>(args[3]);
  const at::Tensor& alpha = *reinterpret_cast<at::Tensor*>(args[4]);
  at::Tensor& output = *reinterpret_cast<at::Tensor*>(args[5]);

  // Determine output data type
  auto out_dtype = output.scalar_type();

  int64_t m = a.size(0);
  int64_t n = b.size(1);
  int64_t k = a.size(1);

  runGemm(output, a, b, scales_a, scales_b, alpha, m, n, k, stream);
}

} // namespace nvfuser::cutlass_kernels
)";

  return code;
}

} // namespace

std::string generateCode(Fusion* fusion) {
  // TODO: match patterns and dispatch to different generators here
  return generateNvfp4ScaledMmKernel(fusion);
}

} // namespace cutlass_codegen

} // namespace nvfuser
