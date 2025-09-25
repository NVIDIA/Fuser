#include <cutlass_utils.h>
#include <nvf_cutlass.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cutlass/arch/arch.h>
#include <torch/torch.h>

#include <cassert>

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"

#include <exceptions.h>

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
  using MmaTileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
};

// Kernel traits for BFloat16 output
template <>
struct KernelTraits<cutlass::bfloat16_t> {
  using MmaTileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
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
//   a_scales_offsets: Output array of pointers to A scaling factors for
//   each expert
//   b_scales_offsets: Output array of pointers to B scaling factors
//   for each expert
//   alpha_offsets: Output array of pointers to alpha scaling
//   factors for each expert
//   layout_sfa_base_as_int: Base pointer for A scale layout configurations
//   layout_sfb_base_as_int: Base pointer for B scale layout configurations
//   a_base_as_int: Base pointer to matrix A data
//   b_base_as_int: Base pointer to matrix B data
//   out_base_as_int: Base pointer to output matrix C data
//   a_scales_base_as_int: Base pointer to A scaling factors
//   b_scales_base_as_int: Base pointer to B scaling factors
//   alphas_base_as_int: Base pointer to alpha scaling factors
//   expert_offsets: Offset indices for expert selection
//   sf_offsets: Scale factor offsets for each expert
//   problem_sizes_as_shapes: Matrix dimensions (M, N, K) for each expert
//   K: Common K dimension across all experts
//   N: Common N dimension across all experts
template <
    typename ElementAB,
    typename ElementC,
    typename ElementSF,
    typename ElementAccumulator,
    typename LayoutSFA,
    typename LayoutSFB,
    typename ScaleConfig>
__global__ void get_group_gemm_starts(
    ElementAB** a_offsets,
    ElementAB** b_offsets,
    ElementC** out_offsets,
    ElementSF** a_scales_offsets,
    ElementSF** b_scales_offsets,
    ElementAccumulator** alpha_offsets,
    LayoutSFA* layout_sfa_base_as_int,
    LayoutSFB* layout_sfb_base_as_int,
    ElementAB* a_base_as_int,
    ElementAB* b_base_as_int,
    ElementC* out_base_as_int,
    ElementSF* a_scales_base_as_int,
    ElementSF* b_scales_base_as_int,
    ElementAccumulator* alphas_base_as_int,
    const int32_t* expert_offsets,
    const int32_t* sf_offsets,
    const int32_t* problem_sizes_as_shapes,
    const int K,
    const int N) {
  int64_t expert_id = threadIdx.x;
  if (expert_id >= gridDim.x * blockDim.x) {
    return;
  }
  // Upcast from int32_t to int64_t to avoid overflow
  // during offset calculations
  int64_t expert_offset = static_cast<int64_t>(expert_offsets[expert_id]);
  int64_t sf_offset = static_cast<int64_t>(sf_offsets[expert_id]);

  // The block size for nvfp4.
  constexpr int64_t nvfp4_block_size = 16;

  int64_t m = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3]);
  int64_t n = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 1]);
  int64_t k = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 2]);
  assert(
      (m >= 0 && n == N && k == K && k % 2 == 0) && "Unexpected problem sizes");

  int64_t half_k = static_cast<int64_t>(k / 2);
  int64_t group_k = static_cast<int64_t>(k / nvfp4_block_size);

  // Shape of A as uint8/byte = [M, K // 2]
  a_offsets[expert_id] = a_base_as_int + expert_offset * half_k;

  // Shape of B as uint8/byte = [E, N, K // 2]
  b_offsets[expert_id] = b_base_as_int + expert_id * n * half_k;

  // Shape of C = [M, N]
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;

  // Shape of a_scale = [sum(sf_sizes), K // nvfp4_block_size]
  a_scales_offsets[expert_id] = a_scales_base_as_int + sf_offset * group_k;
  assert(
      (reinterpret_cast<uintptr_t>(a_scales_offsets[expert_id]) % 128) == 0 &&
      "TMA requires 128-byte alignment");

  // Shape of B scale = [E, N, K // nvfp4_block_size]
  b_scales_offsets[expert_id] = b_scales_base_as_int + expert_id * n * group_k;
  assert(
      (reinterpret_cast<uintptr_t>(b_scales_offsets[expert_id]) % 128) == 0 &&
      "TMA requires 128-byte alignment");

  // Shape of alpha = [E]
  alpha_offsets[expert_id] = alphas_base_as_int + expert_id;

  LayoutSFA* layout_sfa_ptr = layout_sfa_base_as_int + expert_id;
  LayoutSFB* layout_sfb_ptr = layout_sfb_base_as_int + expert_id;

  *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
  *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
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
//   a_scales_starts: Output tensor for A scaling factor pointers
//   b_scales_starts: Output tensor for B scaling factor pointers
//   alpha_starts: Output tensor for alpha scaling factor pointers
//   layout_sfa: Output tensor for A scale layout configurations
//   layout_sfb: Output tensor for B scale layout configurations
//   a_tensors: Input matrix A data
//   b_tensors: Input matrix B data
//   out_tensors: Output matrix C data
//   a_scales: A scaling factors
//   b_scales: B scaling factors
//   alphas: Alpha scaling factors
//   expert_offsets: Expert offset indices
//   sf_offsets: Scale factor offsets
//   problem_sizes: Matrix dimensions for each expert
//   M: Aggregated M dimension across all groups
//   N: Common N dimension across all groups
//   K: Common K dimension across all groups
//   stream: CUDA stream for kernel execution
template <typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
void run_get_group_gemm_starts(
    const torch::Tensor& a_starts,
    const torch::Tensor& b_starts,
    const torch::Tensor& out_starts,
    const torch::Tensor& a_scales_starts,
    const torch::Tensor& b_scales_starts,
    const torch::Tensor& alpha_starts,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& a_tensors,
    const torch::Tensor& b_tensors,
    const torch::Tensor& out_tensors,
    const torch::Tensor& a_scales,
    const torch::Tensor& b_scales,
    const torch::Tensor& alphas,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& sf_offsets,
    const torch::Tensor& problem_sizes,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
  int num_experts = (int)expert_offsets.size(0);

  NVF_CHECK(
      out_tensors.size(1) == N,
      "Output tensor shape doesn't match expected shape");
  NVF_CHECK(
      K / 2 == b_tensors.size(2),
      "b_tensors(dim = 2) and a_tensors(dim = 1) trailing"
      " dimension must match");

  if (out_tensors.dtype() == torch::kBFloat16) {
    get_group_gemm_starts<
        cutlass::float_e2m1_t,
        cutlass::bfloat16_t,
        cutlass::float_ue4m3_t,
        float,
        LayoutSFA,
        LayoutSFB,
        ScaleConfig><<<1, num_experts, 0, stream>>>(
        static_cast<cutlass::float_e2m1_t**>(a_starts.data_ptr()),
        static_cast<cutlass::float_e2m1_t**>(b_starts.data_ptr()),
        static_cast<cutlass::bfloat16_t**>(out_starts.data_ptr()),
        static_cast<cutlass::float_ue4m3_t**>(a_scales_starts.data_ptr()),
        static_cast<cutlass::float_ue4m3_t**>(b_scales_starts.data_ptr()),
        static_cast<float**>(alpha_starts.data_ptr()),
        reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
        reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr()),
        static_cast<cutlass::float_e2m1_t*>(a_tensors.data_ptr()),
        static_cast<cutlass::float_e2m1_t*>(b_tensors.data_ptr()),
        static_cast<cutlass::bfloat16_t*>(out_tensors.data_ptr()),
        static_cast<cutlass::float_ue4m3_t*>(a_scales.data_ptr()),
        static_cast<cutlass::float_ue4m3_t*>(b_scales.data_ptr()),
        static_cast<float*>(alphas.data_ptr()),
        static_cast<int32_t*>(expert_offsets.data_ptr()),
        static_cast<int32_t*>(sf_offsets.data_ptr()),
        static_cast<int32_t*>(problem_sizes.data_ptr()),
        K,
        N);
  } else if (out_tensors.dtype() == torch::kFloat16) {
    get_group_gemm_starts<
        cutlass::float_e2m1_t,
        cutlass::half_t,
        cutlass::float_ue4m3_t,
        float,
        LayoutSFA,
        LayoutSFB,
        ScaleConfig><<<1, num_experts, 0, stream>>>(
        static_cast<cutlass::float_e2m1_t**>(a_starts.data_ptr()),
        static_cast<cutlass::float_e2m1_t**>(b_starts.data_ptr()),
        static_cast<cutlass::half_t**>(out_starts.data_ptr()),
        static_cast<cutlass::float_ue4m3_t**>(a_scales_starts.data_ptr()),
        static_cast<cutlass::float_ue4m3_t**>(b_scales_starts.data_ptr()),
        static_cast<float**>(alpha_starts.data_ptr()),
        reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
        reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr()),
        static_cast<cutlass::float_e2m1_t*>(a_tensors.data_ptr()),
        static_cast<cutlass::float_e2m1_t*>(b_tensors.data_ptr()),
        static_cast<cutlass::half_t*>(out_tensors.data_ptr()),
        static_cast<cutlass::float_ue4m3_t*>(a_scales.data_ptr()),
        static_cast<cutlass::float_ue4m3_t*>(b_scales.data_ptr()),
        static_cast<float*>(alphas.data_ptr()),
        static_cast<int32_t*>(expert_offsets.data_ptr()),
        static_cast<int32_t*>(sf_offsets.data_ptr()),
        static_cast<int32_t*>(problem_sizes.data_ptr()),
        K,
        N);
  } else {
    NVF_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}

// Executes grouped scaled matrix multiplication using NVFP4 format with CUTLASS
// kernels
//
// This function implements the core grouped matrix multiplication using CUTLASS
// kernels optimized for SM100+ architecture. It handles the complex setup
// required for grouped operations including pointer arrays, layout
// configurations, and memory management for NVFP4 format with per-block scaling
// factors.
//
// Parameters:
//   output: Output tensor for the grouped matrix multiplication results
//   a: Input matrix A in Float4_e2m1fn_x2 format
//   b: Input matrix B in Float4_e2m1fn_x2 format
//   a_blockscale: Per-block scaling factors for matrix A
//   b_blockscales: Per-block scaling factors for matrix B
//   alphas: Global scaling factors for each group
//   ab_strides: Stride information for matrices A and B
//   c_strides: Stride information for output matrix C
//   problem_sizes: Matrix dimensions for each group
//   expert_offsets: Expert offset indices
//   sf_offsets: Scale factor offsets
//   M: Aggregated M dimension across all groups
//   N: Common N dimension across all groups
//   K: Common K dimension across all groups
//   stream: CUDA stream for kernel execution
template <typename OutType>
void run_nvfp4_scaled_group_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_blockscale,
    const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas,
    const torch::Tensor& ab_strides,
    const torch::Tensor& c_strides,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& sf_offsets,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
  using ProblemShape =
      cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
  using ElementType = cutlass::float_e2m1_t;
  using ElementSFType = cutlass::float_ue4m3_t;
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

  using ElementC = OutType;
  using ElementD = ElementC;
  using ElementAccumulator = float;
  // Layout definitions
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = LayoutC;

  // Alignment constraints
  static constexpr int AlignmentA = 32;
  static constexpr int AlignmentB = 32;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // Architecture definitions
  using ArchTag = cutlass::arch::Sm100;
  using EpilogueOperatorClass =
      cutlass::arch::OpClassTensorOp; // Epilogue Operator class tag
  using MainloopOperatorClass =
      cutlass::arch::OpClassBlockScaledTensorOp; // Mainloop Operator class tag
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto; // Stage count maximized based
                                                 // on the tile size
  using MmaTileShape = typename KernelTraits<OutType>::MmaTileShape;
  using ClusterShape = typename KernelTraits<OutType>::ClusterShape;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          EpilogueOperatorClass,
          MmaTileShape,
          ClusterShape,
          Shape<_128, _64>,
          ElementAccumulator,
          ElementAccumulator,
          ElementC,
          LayoutC*,
          AlignmentC,
          ElementD,
          LayoutC*,
          AlignmentD,
          cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;

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
          cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100>::
          CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm1SM = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using Gemm = Gemm1SM;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  using LayoutSFA =
      typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB =
      typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using ScaleConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  int num_experts = static_cast<int>(expert_offsets.size(0));
  auto options_int =
      torch::TensorOptions().dtype(torch::kInt64).device(a.device());

  torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor alpha_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor layout_sfa = torch::empty({num_experts, 5}, options_int);
  torch::Tensor layout_sfb = torch::empty({num_experts, 5}, options_int);

  run_get_group_gemm_starts<LayoutSFA, LayoutSFB, ScaleConfig>(
      a_ptrs,
      b_ptrs,
      out_ptrs,
      a_scales_ptrs,
      b_scales_ptrs,
      alpha_ptrs,
      layout_sfa,
      layout_sfb,
      a,
      b,
      output,
      a_blockscale,
      b_blockscales,
      alphas,
      expert_offsets,
      sf_offsets,
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
      static_cast<StrideB*>(ab_strides.data_ptr()),
      static_cast<const ElementSFType**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
      static_cast<const ElementSFType**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr())};

  // Epilogue Arguments
  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, // epilogue.thread
      nullptr,
      static_cast<StrideC*>(c_strides.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(c_strides.data_ptr())};
  auto& fusion_args = epilogue_args.thread;
  fusion_args.alpha_ptr_array =
      reinterpret_cast<float**>(alpha_ptrs.data_ptr());
  fusion_args.dAlpha = {_0{}, _0{}, 1};

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
      torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

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
template <typename OutType>
void run_nvfp4_scaled_group_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_blockscale,
    const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas,
    const torch::Tensor& ab_strides,
    const torch::Tensor& c_strides,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& sf_offsets,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
  NVF_THROW("Unsupported CUTLASS version.");
}
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Validates input parameters for grouped NVFP4 scaled matrix multiplication
//
// This function performs comprehensive validation of all input tensors and
// parameters for the grouped matrix multiplication operation. It checks data
// types, device placement, contiguity, and shape requirements to ensure the
// operation can be performed correctly.
//
// Parameters:
//   a: Input matrix A to validate
//   b: Input matrix B to validate
//   a_blockscale: A scaling factors to validate
//   b_blockscales: B scaling factors to validate
//   alphas: Alpha scaling factors to validate
//   problem_sizes: Problem dimensions to validate
//   expert_offsets: Expert offset indices to validate
//
// Throws: NVF_CHECK exceptions for any validation failures
void validateInputsNvfp4ScaledGroupMm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_blockscale,
    const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& sf_offsets) {
  // Check data types
  NVF_CHECK(
      a.scalar_type() == at::ScalarType::Float4_e2m1fn_x2,
      "Expected Float4_e2m1fn_x2 for Operand A.")
  NVF_CHECK(
      b.scalar_type() == at::ScalarType::Float4_e2m1fn_x2,
      "Expected Float4_e2m1fn_x2 for Operand B.")
  NVF_CHECK(
      a_blockscale.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "Expected FP8_E4M3 for Blockscale scale_a.")
  NVF_CHECK(
      b_blockscales.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "Expected FP8_E4M3 for Blockscale scale_b.")
  NVF_CHECK(
      alphas.scalar_type() == at::ScalarType::Float,
      "Expected FP32 for alpha scalar.")

  // Check CUDA device
  NVF_CHECK(a.is_cuda(), "Expected CUDA tensor for Operand A.")
  NVF_CHECK(b.is_cuda(), "Expected CUDA tensor for Operand B.")
  NVF_CHECK(
      a_blockscale.is_cuda(), "Expected CUDA tensor for Blockscale scale_a.")
  NVF_CHECK(
      b_blockscales.is_cuda(), "Expected CUDA tensor for Blockscale scale_b.")
  NVF_CHECK(alphas.is_cuda(), "Expected CUDA tensor for alpha scalar.")

  // Check contiguity
  NVF_CHECK(a.is_contiguous(), "Expected contiguous tensor for Operand A.")
  NVF_CHECK(b.is_contiguous(), "Expected contiguous tensor for Operand B.")
  NVF_CHECK(
      a_blockscale.is_contiguous(),
      "Expected contiguous tensor for Blockscale scale_a.")
  NVF_CHECK(
      b_blockscales.is_contiguous(),
      "Expected contiguous tensor for Blockscale scale_b.")
  NVF_CHECK(
      alphas.is_contiguous(), "Expected contiguous tensor for alpha scalar.")

  // Check shapes
  NVF_CHECK(
      a_blockscale.dim() == 2,
      "Expected a_blockscale to be of shape: "
      "[padded_m, k // nvfp4_block_size], observed rank: ",
      a_blockscale.dim())
  NVF_CHECK(
      b_blockscales.dim() == 3,
      "Expected b_blockscale to be of shape: "
      " [num_experts, n, k // nvfp4_block_size], observed rank: ",
      b_blockscales.dim())
  NVF_CHECK(problem_sizes.dim() == 2, "problem_sizes must be  a 2D tensor");
  NVF_CHECK(
      problem_sizes.size(1) == 3,
      "problem_sizes must have the shape (num_experts, 3)");
  NVF_CHECK(
      problem_sizes.size(0) == expert_offsets.size(0),
      "Number of experts in problem_sizes must match expert_offsets");
  NVF_CHECK(
      problem_sizes.size(0) == expert_offsets.size(0),
      "Number of experts in problem_sizes must match expert_offsets");
  NVF_CHECK(
      problem_sizes.dtype() == torch::kInt32, "problem_sizes must be int32.");
}

} // namespace

// Main entry point for grouped scaled matrix multiplication using NVFP4 format
//
// This function serves as the main entry point for grouped matrix
// multiplication operations using NVFP4 format. It handles input validation,
// output tensor allocation, and dispatches to the appropriate implementation
// based on the output data type and CUTLASS support availability.
//
// Parameters:
//   a: Input matrix A in Float4_e2m1fn_x2 format (M x K/2)
//   b: Input matrix B in Float4_e2m1fn_x2 format (G x N, K/2)
//   a_blockscale: Per-block scaling factors for matrix A in FP8_E4M3 format
//   b_blockscales: Per-block scaling factors for matrix B in FP8_E4M3 format
//   alphas: Global scaling factors for each group in FP32 format
//   ab_strides: Stride information for matrices A and B across groups
//   c_strides: Stride information for output matrix C across groups
//   problem_sizes: Matrix dimensions (M, N, K) for each group
//   expert_offsets: Offset indices for expert selection in grouped format
//   sf_offsets: Scale factor offsets for each group
//   out_dtype: Output data type (Half or BFloat16)
//
// Returns: Grouped matrix C = alpha * (A @ B) for all groups in the specified
// output dtype
torch::Tensor nvfp4_scaled_grouped_mm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_blockscale,
    const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas,
    const torch::Tensor& ab_strides,
    const torch::Tensor& c_strides,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& sf_offsets,
    const at::ScalarType out_dtype) {
  // Calculate output shape and allocate output tensor
  auto options =
      at::TensorOptions().dtype(out_dtype).device(at::kCUDA, a.get_device());
  torch::Tensor output = at::empty({a.size(0), b.size(1)}, options);

  validateInputsNvfp4ScaledGroupMm(
      a,
      b,
      a_blockscale,
      b_blockscales,
      alphas,
      problem_sizes,
      expert_offsets,
      sf_offsets);

  int M = static_cast<int>(a.size(0));
  int N = static_cast<int>(b.size(1));
  int K = static_cast<int>(2 * b.size(2));

  at::cuda::CUDAGuard device_guard{(int8_t)a.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device());

  if (out_dtype == at::ScalarType::Half) {
    run_nvfp4_scaled_group_mm<cutlass::half_t>(
        output,
        a,
        b,
        a_blockscale,
        b_blockscales,
        alphas,
        ab_strides,
        c_strides,
        problem_sizes,
        expert_offsets,
        sf_offsets,
        M,
        N,
        K,
        stream);
  } else {
    if (out_dtype == at::ScalarType::BFloat16) {
      run_nvfp4_scaled_group_mm<cutlass::bfloat16_t>(
          output,
          a,
          b,
          a_blockscale,
          b_blockscales,
          alphas,
          ab_strides,
          c_strides,
          problem_sizes,
          expert_offsets,
          sf_offsets,
          M,
          N,
          K,
          stream);
    } else {
      NVF_THROW("Unsupported output data type of nvfp4 scaled_grouped_mm.");
    }
  }
  return output;
}

} // namespace nvfuser::cutlass_kernels
