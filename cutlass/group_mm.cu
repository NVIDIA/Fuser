// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cassert>

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cutlass/arch/arch.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass_utils.h>
#include <exceptions.h>
#include <nvf_cutlass.h>

namespace nvfuser::cutlass_kernels {

namespace {

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
// Kernel configuration traits for different output data types
// Defines tile shapes and cluster configurations.
template <typename T, bool is_single_sm>
struct KernelTraits;

// Kernel traits for FP16 output
template <>
struct KernelTraits<cutlass::half_t, true> {
  using MmaTileShape = Shape<_128, _256, Int<128 / sizeof(cutlass::half_t)>>;
  using ClusterShape = Shape<_1, _1, _1>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
};

// Kernel traits for BFloat16 output
template <>
struct KernelTraits<cutlass::bfloat16_t, true> {
  using MmaTileShape =
      Shape<_128, _256, Int<128 / sizeof(cutlass::bfloat16_t)>>;
  using ClusterShape = Shape<_1, _1, _1>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
};

// CUDA kernel to compute memory offsets and layout information for grouped GEMM
// operations
//
// This kernel calculates the starting pointers and layout configurations for
// each expert in a grouped matrix multiplication.
//
// Parameters:
//   a_offsets: Output array of pointers to matrix A data for each expert
//   b_offsets: Output array of pointers to matrix B data for each expert
//   out_offsets: Output array of pointers to output matrix C data for each
//   expert
//   a_base_as_int: Base pointer to matrix A data
//   b_base_as_int: Base pointer to matrix B data
//   out_base_as_int: Base pointer to output matrix C data
//   expert_offsets: Offset indices for expert selection
//   problem_sizes_as_shapes: Matrix dimensions (M, N, K) for each expert
//   K: Common K dimension across all experts
//   N: Common N dimension across all experts
template <typename ElementAB, typename ElementC>
__global__ void get_group_gemm_starts(
    ElementAB** a_offsets,
    ElementAB** b_offsets,
    ElementC** out_offsets,
    ElementAB* a_base_as_int,
    ElementAB* b_base_as_int,
    ElementC* out_base_as_int,
    const int32_t* expert_offsets,
    const int32_t* problem_sizes_as_shapes,
    const int K,
    const int N) {
  int64_t expert_id = threadIdx.x;
  if (expert_id >= gridDim.x * blockDim.x) {
    return;
  }
  // Upcast from int32_t to int64_t to avoid overflow during offset calculations
  int64_t expert_offset = static_cast<int64_t>(expert_offsets[expert_id]);
  int64_t n = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 1]);
  int64_t k = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 2]);
  assert((n == N && k == K) && "Unexpected problem sizes");

  // Shape of A as uint8/byte = [M, K]
  a_offsets[expert_id] = a_base_as_int + expert_offset * k;

  // Shape of B as uint8/byte = [E, N, K]
  b_offsets[expert_id] = b_base_as_int + expert_id * n * k;

  // Shape of C = [M, N]
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
}

// Launches the CUDA kernel to compute memory offsets and layout information for
// grouped GEMM
//
// This function launches the get_group_gemm_starts kernel with appropriate
// template parameters based on the output data type. It handles the setup and
// execution of the offset computation kernel for grouped matrix multiplication
// operations.
//
// Parameters:
//   a_starts: Output tensor for matrix A pointers
//   b_starts: Output tensor for matrix B pointers
//   out_starts: Output tensor for output matrix C pointers
//   a_tensors: Input matrix A data
//   b_tensors: Input matrix B data
//   out_tensors: Output matrix C data
//   expert_offsets: Expert offset indices
//   problem_sizes: Matrix dimensions for each expert
//   M:  Aggregated M dimension across all groups
//   N: Common N dimension across all groups
//   K: Common K dimension across all groups
//   stream: CUDA stream for kernel execution
void run_get_group_gemm_starts(
    const at::Tensor& a_starts,
    const at::Tensor& b_starts,
    const at::Tensor& out_starts,
    const at::Tensor& a_tensors,
    const at::Tensor& b_tensors,
    const at::Tensor& out_tensors,
    const at::Tensor& expert_offsets,
    const at::Tensor& problem_sizes,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
  int num_experts = (int)expert_offsets.size(0);

  NVF_CHECK_EQ(out_tensors.size(1), N);
  NVF_CHECK_EQ(K, b_tensors.size(2));

  if (out_tensors.dtype() == at::kBFloat16) {
    get_group_gemm_starts<cutlass::bfloat16_t, cutlass::bfloat16_t>
        <<<1, num_experts, 0, stream>>>(
            static_cast<cutlass::bfloat16_t**>(a_starts.data_ptr()),
            static_cast<cutlass::bfloat16_t**>(b_starts.data_ptr()),
            static_cast<cutlass::bfloat16_t**>(out_starts.data_ptr()),
            static_cast<cutlass::bfloat16_t*>(a_tensors.data_ptr()),
            static_cast<cutlass::bfloat16_t*>(b_tensors.data_ptr()),
            static_cast<cutlass::bfloat16_t*>(out_tensors.data_ptr()),
            static_cast<int32_t*>(expert_offsets.data_ptr()),
            static_cast<int32_t*>(problem_sizes.data_ptr()),
            K,
            N);
  } else if (out_tensors.dtype() == at::kHalf) {
    get_group_gemm_starts<cutlass::half_t, cutlass::half_t>
        <<<1, num_experts, 0, stream>>>(
            static_cast<cutlass::half_t**>(a_starts.data_ptr()),
            static_cast<cutlass::half_t**>(b_starts.data_ptr()),
            static_cast<cutlass::half_t**>(out_starts.data_ptr()),
            static_cast<cutlass::half_t*>(a_tensors.data_ptr()),
            static_cast<cutlass::half_t*>(b_tensors.data_ptr()),
            static_cast<cutlass::half_t*>(out_tensors.data_ptr()),
            static_cast<int32_t*>(expert_offsets.data_ptr()),
            static_cast<int32_t*>(problem_sizes.data_ptr()),
            K,
            N);
  } else {
    NVF_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}

// Executes grouped matrix multiplication with CUTLASS kernels
//
// Parameters:
//   output: Output tensor for the grouped matrix multiplication results
//   a: Input matrix A in BF16 or FP16 format
//   b: Input matrix B in BF16 or FP16 format
//   ab_strides: Stride information for matrices A and B
//   c_strides: Stride information for output matrix C
//   problem_sizes: Matrix dimensions for each group
//   expert_offsets: Expert offset indices
//   M: Aggregated M dimension across all groups
//   N: Common N dimension across all groups
//   K: Common K dimension across all groups
//   stream: CUDA stream for kernel execution
template <typename DType>
void run_group_mm(
    at::Tensor& output,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& ab_strides,
    const at::Tensor& c_strides,
    const at::Tensor& problem_sizes,
    const at::Tensor& expert_offsets,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
  using ProblemShape =
      cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
  using ElementType = DType;
  using ElementA = DType;
  using ElementB = DType;

  using ElementC = DType;
  using ElementD = ElementC;
  using ElementAccumulator = float;

  // Layout definitions
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = LayoutC;

  // Alignment constraints
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // Architecture definitions
  using ArchTag = cutlass::arch::Sm100;
  using EpilogueOperatorClass =
      cutlass::arch::OpClassTensorOp; // Epilogue Operator class tag
  using MainloopOperatorClass =
      cutlass::arch::OpClassTensorOp; // Mainloop Operator class tag
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto; // Stage count maximized based
                                                 // on the tile size
  using SingleSmKernelTraits = KernelTraits<DType, /*is_single_sm=*/true>;
  using MmaTileShape = typename SingleSmKernelTraits::MmaTileShape;
  using ClusterShape = typename SingleSmKernelTraits::ClusterShape;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          EpilogueOperatorClass,
          MmaTileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementAccumulator,
          ElementC,
          LayoutC*,
          AlignmentC,
          ElementD,
          LayoutC*,
          AlignmentD,
          typename SingleSmKernelTraits::EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          MainloopOperatorClass,
          ElementA,
          LayoutA*,
          AlignmentA,
          ElementB,
          LayoutB*,
          AlignmentB,
          ElementAccumulator,
          MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          typename SingleSmKernelTraits::KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm1SM = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using Gemm = Gemm1SM;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;

  // Create cutlass arguments
  int num_experts = static_cast<int>(expert_offsets.size(0));
  auto options_int = at::TensorOptions().dtype(at::kLong).device(a.device());
  at::Tensor a_ptrs = at::empty(num_experts, options_int);
  at::Tensor b_ptrs = at::empty(num_experts, options_int);
  at::Tensor out_ptrs = at::empty(num_experts, options_int);
  run_get_group_gemm_starts(
      a_ptrs,
      b_ptrs,
      out_ptrs,
      a,
      b,
      output,
      expert_offsets,
      problem_sizes,
      M,
      N,
      K,
      stream);

  // Create an instance of the GEMM
  Gemm gemm_op;

  // Initialize problem_sizes_as_shapes correctly
  UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());

  // Set the Scheduler info
  cutlass::KernelHardwareInfo hw_info;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::
      PersistentTileSchedulerSm100GroupParams<
          typename ProblemShape::UnderlyingProblemShape>::RasterOrderOptions;
  typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = RasterOrderOptions::AlongM;
  hw_info.device_id = a.get_device();
  static std::unordered_map<int, int> cached_sm_counts;
  if (cached_sm_counts.find(hw_info.device_id) == cached_sm_counts.end()) {
    cached_sm_counts[hw_info.device_id] =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);
  }
  hw_info.sm_count = min(cached_sm_counts[hw_info.device_id], INT_MAX);

  // Mainloop Arguments
  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementType**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(ab_strides.data_ptr()),
      static_cast<const ElementType**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(ab_strides.data_ptr())};

  // Epilogue Arguments
  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, // epilogue.thread
      nullptr,
      static_cast<StrideC*>(c_strides.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(c_strides.data_ptr())};
  // Add fusion_args here

  // Gemm Arguments
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info,
      scheduler};

  size_t workspace_size = Gemm::get_workspace_size(args);
  auto const workspace_options =
      at::TensorOptions().dtype(at::kByte).device(a.device());
  auto workspace = at::empty(workspace_size, workspace_options);

  auto can_implement_status = gemm_op.can_implement(args);
  NVF_CHECK(
      can_implement_status == cutlass::Status::kSuccess,
      "Failed to implement GEMM");

  // Run the GEMM
  auto status = gemm_op.initialize(args, workspace.data_ptr());
  NVF_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm_op.run(args, workspace.data_ptr(), stream);
  NVF_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}
#else
// This is a fallback implementation that throws an error when the required
// CUTLASS version with SM100+ support is not available. It maintains the same
// function signature as the main implementation for compatibility.
//
// Parameters: Same as the main implementation
// Returns: Never returns (throws exception)
template <typename DType>
void run_group_mm(
    at::Tensor& output,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& ab_strides,
    const at::Tensor& c_strides,
    const at::Tensor& problem_sizes,
    const at::Tensor& expert_offsets,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
  NVF_THROW("Unsupported CUTLASS version.");
}
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Validates input parameters for grouped matrix multiplication
//
// This function performs comprehensive validation of all input tensors and
// parameters for the grouped matrix multiplication operation. It checks data
// types, device placement, contiguity, and shape requirements to ensure the
// operation can be performed correctly.
//
// Parameters:
//   a: Input matrix A to validate
//   b: Input matrix B to validate
//   problem_sizes: Problem dimensions to validate
//   expert_offsets: Expert offset indices to validate
//
// Throws: NVF_CHECK exceptions for any validation failures
void validateInputsGroupMm(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& ab_strides,
    const at::Tensor& c_strides,
    const at::Tensor& problem_sizes,
    const at::Tensor& expert_offsets) {
  // Check data types
  NVF_CHECK(
      a.scalar_type() == at::ScalarType::BFloat16 ||
          a.scalar_type() == at::ScalarType::Half,
      "Expected BFloat16 or Half for Operand A.")
  NVF_CHECK(
      b.scalar_type() == at::ScalarType::BFloat16 ||
          b.scalar_type() == at::ScalarType::Half,
      "Expected BFloat16 or Half for Operand B.")

#ifndef NDEBUG
  if (c10::cuda::currentStreamCaptureStatusMayInitCtx() ==
      c10::cuda::CaptureStatus::None) {
    const int64_t m = a.size(0);
    const int64_t g = expert_offsets.size(0);
    // This validation requires an expensive synchronization and therfore is
    // only enabled in debug mode. See #5470.
    at::Tensor expert_offsets_cpu = expert_offsets.cpu();
    int64_t prev_offset = 0;
    for (int64_t i = 0; i < g; i++) {
      const auto expert_offset = expert_offsets_cpu[i].item<int64_t>();
      NVF_CHECK_LE(expert_offset, m);
      NVF_CHECK_LE(prev_offset, expert_offset);
      prev_offset = expert_offset;
    }
  }
#endif

  NVF_CHECK_EQ(ab_strides.dtype(), at::kLong);
  NVF_CHECK_EQ(c_strides.dtype(), at::kLong);

  // Check CUDA device
  NVF_CHECK(a.is_cuda(), "Expected CUDA tensor for Operand A.")
  NVF_CHECK(b.is_cuda(), "Expected CUDA tensor for Operand B.")

  // Check contiguity
  NVF_CHECK(a.is_contiguous(), "Expected contiguous tensor for Operand A.")
  NVF_CHECK(b.is_contiguous(), "Expected contiguous tensor for Operand B.")

  // Check shapes
  NVF_CHECK_EQ(problem_sizes.dim(), 2);
  NVF_CHECK_EQ(problem_sizes.size(1), 3);
  NVF_CHECK_EQ(problem_sizes.size(0), expert_offsets.size(0));
  NVF_CHECK_EQ(problem_sizes.dtype(), at::kInt);
}

} // namespace

// Main entry point for grouped matrix multiplication.
//
// It handles input validation, output tensor allocation, and dispatches to the
// appropriate implementation based on the output data type and CUTLASS support
// availability.
//
// Parameters:
//   a: Input matrix A in BF16 or FP16 format (M x K)
//   b: Input matrix B in BF16 or FP16 format (G x N x K)
//   ab_strides: Stride information for matrices A and B across groups
//   c_strides: Stride information for output matrix C across groups
//   problem_sizes: Matrix dimensions (M, N, K) for each group
//   expert_offsets: Offset indices for expert selection in grouped format
//   out_dtype: Output data type (Half or BFloat16)
//
// Returns: Grouped matrix C = A @ B for all groups in the specified
// output dtype
at::Tensor grouped_mm(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& ab_strides,
    const at::Tensor& c_strides,
    const at::Tensor& problem_sizes,
    const at::Tensor& expert_offsets) {
  // Calculate output shape and allocate output tensor
  auto options = at::TensorOptions()
                     .dtype(a.scalar_type())
                     .device(at::kCUDA, a.get_device());
  at::Tensor output = at::empty({a.size(0), b.size(1)}, options);

  validateInputsGroupMm(
      a, b, ab_strides, c_strides, problem_sizes, expert_offsets);

  int M = static_cast<int>(a.size(0));
  int N = static_cast<int>(b.size(1));
  int K = static_cast<int>(b.size(2));

  at::cuda::CUDAGuard device_guard{(int8_t)a.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device());

  if (a.scalar_type() == at::ScalarType::Half) {
    run_group_mm<cutlass::half_t>(
        output,
        a,
        b,
        ab_strides,
        c_strides,
        problem_sizes,
        expert_offsets,
        M,
        N,
        K,
        stream);
  } else if (a.scalar_type() == at::ScalarType::BFloat16) {
    run_group_mm<cutlass::bfloat16_t>(
        output,
        a,
        b,
        ab_strides,
        c_strides,
        problem_sizes,
        expert_offsets,
        M,
        N,
        K,
        stream);
  } else {
    NVF_THROW("Unsupported output data type of grouped_mm.");
  }
  return output;
}

} // namespace nvfuser::cutlass_kernels
