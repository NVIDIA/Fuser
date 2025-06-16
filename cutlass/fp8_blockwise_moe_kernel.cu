#include <cutlass_utils.h>
#include <nvf_cutlass.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/arch/arch.h>
#include <torch/torch.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/tensor_view_io.h"

#include <exceptions.h>

namespace nvfuser::cutlass_kernels {

using namespace cute;

template <
    typename ElementAB,
    typename ElementC,
    typename ElementAccumulator,
    typename LayoutSFA,
    typename LayoutSFB,
    typename ScaleConfig>
__global__ void get_group_gemm_starts(
    int32_t* expert_offsets,
    ElementAB** a_offsets,
    ElementAB** b_offsets,
    ElementC** out_offsets,
    ElementAccumulator** a_scales_offsets,
    ElementAccumulator** b_scales_offsets,
    ElementAB* a_base_as_int,
    ElementAB* b_base_as_int,
    ElementC* out_base_as_int,
    ElementAccumulator* a_scales_base_as_int,
    ElementAccumulator* b_scales_base_as_int,
    LayoutSFA* layout_sfa_base_as_int,
    LayoutSFB* layout_sfb_base_as_int,
    int* problem_sizes,
    int* problem_sizes_transpose,
    bool transpose = false) {
  int expert_id = threadIdx.x;

  if (expert_id >= gridDim.x * blockDim.x) {
    return;
  }

  int m = problem_sizes[expert_id * 3];
  int n = problem_sizes[expert_id * 3 + 1];
  int k = problem_sizes[expert_id * 3 + 2];
  if (transpose) {
    problem_sizes_transpose[expert_id * 3] = n;
    problem_sizes_transpose[expert_id * 3 + 1] = m;
    problem_sizes_transpose[expert_id * 3 + 2] = k;
  }

  int32_t expert_offset = expert_offsets[expert_id];
  int a_stride = 0;
  int b_stride = 0;
  int a_scale_stride = 0;
  int b_scale_stride = 0;
  if (!transpose) {
    a_stride = expert_offset * k;
    b_stride = expert_id * k * n;
    a_scale_stride = expert_offset * k / 128;
    b_scale_stride = expert_id * k * n / 128 / 128;
  } else {
    a_stride = expert_id * k * n;
    b_stride = expert_offset * k;
    a_scale_stride = expert_id * k * n / 128 / 128;
    b_scale_stride = expert_offset * k / 128;
  }
  a_offsets[expert_id] = a_base_as_int + a_stride;
  b_offsets[expert_id] = b_base_as_int + b_stride;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  a_scales_offsets[expert_id] = a_scales_base_as_int + a_scale_stride;
  b_scales_offsets[expert_id] = b_scales_base_as_int + b_scale_stride;

  LayoutSFA* layout_sfa_ptr = layout_sfa_base_as_int + expert_id;
  LayoutSFB* layout_sfb_ptr = layout_sfb_base_as_int + expert_id;

  if (!transpose) {
    *layout_sfa_ptr =
        ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    *layout_sfb_ptr =
        ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  } else {
    *layout_sfa_ptr =
        ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(n, m, k, 1));
    *layout_sfb_ptr =
        ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(n, m, k, 1));
  }
}

#define __CALL_GET_STARTS_KERNEL(                                  \
    TENSOR_C_TYPE, C_TYPE, LayoutSFA, LayoutSFB, ScaleConfig)      \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                 \
    get_group_gemm_starts<                                         \
        cutlass::float_e4m3_t,                                     \
        C_TYPE,                                                    \
        float,                                                     \
        LayoutSFA,                                                 \
        LayoutSFB,                                                 \
        ScaleConfig><<<1, num_experts, 0, stream>>>(               \
        static_cast<int32_t*>(expert_offsets.data_ptr()),          \
        static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr()),   \
        static_cast<cutlass::float_e4m3_t**>(b_ptrs.data_ptr()),   \
        static_cast<C_TYPE**>(out_ptrs.data_ptr()),                \
        static_cast<float**>(a_scales_ptrs.data_ptr()),            \
        static_cast<float**>(b_scales_ptrs.data_ptr()),            \
        static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr()), \
        static_cast<cutlass::float_e4m3_t*>(b_tensors.data_ptr()), \
        static_cast<C_TYPE*>(out_tensors.data_ptr()),              \
        static_cast<float*>(a_scales.data_ptr()),                  \
        static_cast<float*>(b_scales.data_ptr()),                  \
        reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),       \
        reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr()),       \
        static_cast<int*>(problem_sizes.data_ptr()),               \
        static_cast<int*>(problem_sizes_transpose.data_ptr()),     \
        transpose);                                                \
  }

template <typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
void run_get_group_gemm_starts(
    torch::Tensor const& expert_offsets,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor& out_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& layout_sfa,
    torch::Tensor const& layout_sfb,
    torch::Tensor const& problem_sizes,
    torch::Tensor& problem_sizes_transpose,
    bool transpose = false) {
  NVF_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn);
  NVF_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn);
  NVF_CHECK(a_scales.dtype() == torch::kFloat32);
  NVF_CHECK(b_scales.dtype() == torch::kFloat32);
  NVF_CHECK(out_tensors.size(1) % 128 == 0 or out_tensors.size(0) % 128 == 0);
  NVF_CHECK(a_tensors.size(1) % 128 == 0 or a_tensors.size(0) % 128 == 0);

  int num_experts = (int)expert_offsets.size(0);
  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());

  if (false) {
  }
  __CALL_GET_STARTS_KERNEL(
      torch::kBFloat16, cutlass::bfloat16_t, LayoutSFA, LayoutSFB, ScaleConfig)
  __CALL_GET_STARTS_KERNEL(
      torch::kFloat16, half, LayoutSFA, LayoutSFB, ScaleConfig)
  else {
    NVF_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
template <typename OutType, typename ScheduleConfig, typename LayoutD>
void launch_sm100_fp8_blockwise_scaled_group_mm(
    torch::Tensor& out_ptrs,
    const torch::Tensor& a_ptrs,
    const torch::Tensor& b_ptrs,
    const torch::Tensor& a_scales_ptrs,
    const torch::Tensor& b_scales_ptrs,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = OutType;
  using ElementD = ElementC;
  using ElementAccumulator = float;
  // Layout definitions
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = LayoutD;

  // Alignment constraints
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  // Architecture definitions
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          typename ScheduleConfig::MmaTileShape,
          typename ScheduleConfig::ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementAccumulator,
          void,
          LayoutC*,
          AlignmentC,
          ElementD,
          LayoutC*,
          AlignmentC,
          typename ScheduleConfig::EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementA,
          cute::tuple<LayoutA*, typename ScheduleConfig::LayoutSFA*>,
          AlignmentA,
          ElementB,
          cute::tuple<LayoutB*, typename ScheduleConfig::LayoutSFB*>,
          AlignmentB,
          ElementAccumulator,
          typename ScheduleConfig::MmaTileShape,
          typename ScheduleConfig::ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          typename ScheduleConfig::KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  int num_experts = (int)expert_offsets.size(0);
  // Create an instance of the GEMM
  Gemm gemm_op;

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementA**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(stride_a.data_ptr()),
      static_cast<const ElementB**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(stride_b.data_ptr()),
      static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<typename ScheduleConfig::LayoutSFA*>(
          layout_sfa.data_ptr()),
      static_cast<const ElementAccumulator**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<typename ScheduleConfig::LayoutSFB*>(
          layout_sfb.data_ptr())};

  cutlass::KernelHardwareInfo hw_info;

  hw_info.device_id = 0;
  // For SM100 Blackwell, the number of SM is 148.
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  hw_info.sm_count = dev_prop->multiProcessorCount;
  // Currently, we are only able to do broadcast on either all or none a_scales
  // and on either all or none b_scales
  typename GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      static_cast<StrideC*>(stride_c.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(stride_c.data_ptr())};

  UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  at::cuda::CUDAGuard device_guard{(int8_t)a_ptrs.get_device()};
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(a_ptrs.get_device());

  auto can_implement_status = gemm_op.can_implement(args);
  NVF_CHECK(
      can_implement_status == cutlass::Status::kSuccess,
      "Failed to implement GEMM");

  // Run the GEMM
  auto status = gemm_op.initialize(args, workspace.data_ptr(), stream);

  NVF_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm_op.run(stream);
  NVF_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

template <typename OutType>
void sm100_fp8_blockwise_group_mm_dispatch_shape(
    torch::Tensor& output,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  // Check the first matrix size to decide on the configuration
  // Assuming all matrices in the group have similar size characteristics
  // bool use_small_config = a[0].size(0) <= 128;
  struct MmaConfig1 {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_128, _32, _128>;
    using ClusterShape =
        Shape<_1, _1, _1>; // Layout type for SFB matrix operand
    using KernelSchedule =
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
        128,
        1,
        128,
        cute::UMMA::Major::K,
        cute::UMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };
  struct MmaConfig2 {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_128, _128, _128>;
    using ClusterShape =
        Shape<_1, _1, _1>; // Layout type for SFB matrix operand
    using KernelSchedule =
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
        1,
        128,
        128,
        cute::UMMA::Major::K,
        cute::UMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };
  struct MMAConfig3 {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_64, _128, _128>;
    using ClusterShape =
        Shape<_1, _1, _1>; // Layout type for SFB matrix operand
    using KernelSchedule =
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
        1,
        128,
        128,
        cute::UMMA::Major::K,
        cute::UMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };
  int num_experts = (int)expert_offsets.size(0);
  torch::TensorOptions options_int =
      torch::TensorOptions().dtype(torch::kInt64).device(a.device());
  torch::Tensor problem_sizes_transpose =
      torch::empty(num_experts * 3, options_int);
  torch::Tensor output_t = output.t();
  torch::Tensor a_t = a.t();
  torch::Tensor b_t = b.transpose(1, 2);
  torch::Tensor scales_a_t = scales_a.t();
  torch::Tensor scales_b_t = scales_b.transpose(1, 2);

  if (a.size(0) <= 512 && a.size(1) >= 2048) {
    run_get_group_gemm_starts<
        MmaConfig1::LayoutSFA,
        MmaConfig1::LayoutSFB,
        MmaConfig1::ScaleConfig>(
        expert_offsets,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        b_t,
        a_t,
        output_t,
        scales_b_t,
        scales_a_t,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        problem_sizes_transpose,
        true);
    launch_sm100_fp8_blockwise_scaled_group_mm<
        OutType,
        MmaConfig1,
        cutlass::layout::ColumnMajor>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes_transpose,
        expert_offsets,
        workspace);
    output = output_t.t();
  } else if (a.size(0) > 512 && a.size(1) >= 2048) {
    run_get_group_gemm_starts<
        MmaConfig2::LayoutSFA,
        MmaConfig2::LayoutSFB,
        MmaConfig2::ScaleConfig>(
        expert_offsets,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        a,
        b,
        output,
        scales_a,
        scales_b,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        problem_sizes_transpose);
    launch_sm100_fp8_blockwise_scaled_group_mm<
        OutType,
        MmaConfig2,
        cutlass::layout::RowMajor>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets,
        workspace);
  } else {
    run_get_group_gemm_starts<
        MMAConfig3::LayoutSFA,
        MMAConfig3::LayoutSFB,
        MMAConfig3::ScaleConfig>(
        expert_offsets,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        a,
        b,
        output,
        scales_a,
        scales_b,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        problem_sizes_transpose);
    launch_sm100_fp8_blockwise_scaled_group_mm<
        OutType,
        MMAConfig3,
        cutlass::layout::RowMajor>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets,
        workspace);
  }
}

/**
 * @brief Performs blockwise grouped matrix multiplication on FP8 quantized
 * inputs, with per-block scaling.
 *
 * This function dispatches to hardware-specific implementations (e.g., SM100
 * FP8) to compute: C_i = scale_a[i] * A_i * scale_b[i] * B_i for each expert
 * group `i`, using input `problem_sizes` and `expert_offsets` to describe the
 * individual matrix dimensions and their offsets.
 *
 * Input tensors A and B must be quantized to 8-bit formats and dequantized
 * before multiplication. The output tensor is written with bfloat16 or half
 * precision.
 *
 * @param output         Output tensor (must be of type bfloat16 or half).
 * @param a              Input tensor A (must be kFloat8_e4m3fn).
 * @param b              Input tensor B (must be kFloat8_e4m3fn).
 * @param scales_a       Scaling factors for tensor A, float32 per expert group.
 * @param scales_b       Scaling factors for tensor B, float32 per expert group.
 * @param stride_a       Stride information for tensor A (int32).
 * @param stride_b       Stride information for tensor B (int32).
 * @param stride_c       Stride information for output tensor C (int32).
 * @param layout_sfa     Layout descriptor for A (int32), e.g.,
 * row-major/column-major.
 * @param layout_sfb     Layout descriptor for B (int32).
 * @param problem_sizes  2D int32 tensor of shape (num_experts, 3), specifying
 * (M, N, K) for each grouped matrix multiplication problem.
 * @param expert_offsets 1D int32 tensor of size (num_experts), used to index
 * into the grouped input tensors for dispatch.
 *  @note Performance Optimization:
 *       If the batch size (a.size(0)) is smaller than 512, the implementation
 *       will internally transpose input matrices to align with the optimal
 * memory access pattern for better GPU efficiency. This transformation is done
 * within the kernel.
 */
void fp8_blockwise_scaled_grouped_mm(
    torch::Tensor& output,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  NVF_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  NVF_CHECK(
      problem_sizes.size(1) == 3,
      "problem_sizes must have shape (num_experts, 3)");
  NVF_CHECK(
      problem_sizes.size(0) == expert_offsets.size(0),
      "Number of experts in problem_sizes must match expert_offsets");
  NVF_CHECK(
      problem_sizes.dtype() == torch::kInt32, "problem_sizes must be int32");
  NVF_CHECK(
      a.scalar_type() == torch::kFloat8_e4m3fn, "a must be kFloat8_e4m3fn");
  NVF_CHECK(
      b.scalar_type() == torch::kFloat8_e4m3fn, "b must be kFloat8_e4m3fn");
  NVF_CHECK(
      output.scalar_type() == torch::kBFloat16 ||
          output.scalar_type() == torch::kHalf,
      "output must be bfloat16 or half");
  NVF_CHECK(
      scales_a.scalar_type() == torch::kFloat32, "scales_a must be float32");
  NVF_CHECK(
      scales_b.scalar_type() == torch::kFloat32, "scales_b must be float32");
  NVF_CHECK(stride_a.scalar_type() == torch::kInt64, "stride_a must be int64");
  NVF_CHECK(stride_b.scalar_type() == torch::kInt64, "stride_b must be int64");
  NVF_CHECK(stride_c.scalar_type() == torch::kInt64, "stride_c must be int64");
  NVF_CHECK(
      layout_sfa.scalar_type() == torch::kInt32, "layout_sfa must be int32");
  NVF_CHECK(
      layout_sfb.scalar_type() == torch::kInt32, "layout_sfb must be int32");
  NVF_CHECK(
      expert_offsets.scalar_type() == torch::kInt32,
      "expert_offsets must be int32");

  NVF_CHECK(output.dim() == 2, "output must be 2D tensor");
  NVF_CHECK(a.dim() == 2, "a must be 2D tensor");
  NVF_CHECK(b.dim() == 3, "b must be 3D tensor");
  NVF_CHECK(scales_a.dim() == 2, "scales_a must be 2D tensor");
  NVF_CHECK(scales_b.dim() == 3, "scales_b must be 3D tensor");
  NVF_CHECK(stride_a.dim() == 1, "stride_a must be 1D tensor");
  NVF_CHECK(stride_b.dim() == 1, "stride_b must be 1D tensor");
  NVF_CHECK(stride_c.dim() == 1, "stride_c must be 1D tensor");
  NVF_CHECK(layout_sfa.dim() == 2, "layout_sfa must be 1D tensor");
  NVF_CHECK(layout_sfb.dim() == 2, "layout_sfb must be 1D tensor");
  NVF_CHECK(a_ptrs.dim() == 1, "a_ptrs must be 1D tensor");
  NVF_CHECK(b_ptrs.dim() == 1, "b_ptrs must be 1D tensor");
  NVF_CHECK(out_ptrs.dim() == 1, "out_ptrs must be 1D tensor");
  NVF_CHECK(a_scales_ptrs.dim() == 1, "a_scales_ptrs must be 1D tensor");
  NVF_CHECK(b_scales_ptrs.dim() == 1, "b_scales_ptrs must be 1D tensor");
  NVF_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  NVF_CHECK(
      problem_sizes.size(1) == 3,
      "problem_sizes must have shape (num_experts, 3)");
  NVF_CHECK(
      problem_sizes.size(0) == expert_offsets.size(0),
      "Number of experts in problem_sizes must match expert_offsets");
  NVF_CHECK(
      problem_sizes.dtype() == torch::kInt32, "problem_sizes must be int32");
  NVF_CHECK(expert_offsets.dim() == 1, "expert_offsets must be 1D tensor");
  NVF_CHECK(workspace.dim() == 1, "workspace must be 1D tensor");

  bool can_implement = false;
  auto sm_version = getSMVersion();

#if defined(CUTLASS_ARCH_MMA_SM100A_SUPPORTED) || \
    defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#if defined CUDA_VERSION && CUDA_VERSION >= 12080
  if (sm_version == 100) {
    if (output.scalar_type() == torch::kBFloat16) {
      sm100_fp8_blockwise_group_mm_dispatch_shape<cutlass::bfloat16_t>(
          output,
          a_ptrs,
          b_ptrs,
          out_ptrs,
          a_scales_ptrs,
          b_scales_ptrs,
          a,
          b,
          scales_a,
          scales_b,
          stride_a,
          stride_b,
          stride_c,
          layout_sfa,
          layout_sfb,
          problem_sizes,
          expert_offsets,
          workspace);
    } else {
      sm100_fp8_blockwise_group_mm_dispatch_shape<cutlass::half_t>(
          output,
          a_ptrs,
          b_ptrs,
          out_ptrs,
          a_scales_ptrs,
          b_scales_ptrs,
          a,
          b,
          scales_a,
          scales_b,
          stride_a,
          stride_b,
          stride_c,
          layout_sfa,
          layout_sfb,
          problem_sizes,
          expert_offsets,
          workspace);
    }
    can_implement = true;
  }
#endif
#endif

  NVF_CHECK(
      can_implement,
      "fp8_blockwise_scaled_grouped_mm is not implemented for current compute "
      "capability: ",
      sm_version);
}

} // namespace nvfuser::cutlass_kernels
