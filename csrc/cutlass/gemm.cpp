// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cutlass/codegen.h>
#include <cutlass/evt.h>
#include <cutlass/gemm.h>
#include <dispatch.h>
#include <exceptions.h>
#include <fusion.h>
#include <ir/interface_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/utils.h>
#include <scheduler/cutlass.h>
#include <type.h>

#include <algorithm>
#include <format>
#include <string>

namespace nvfuser {

namespace cutlass_codegen {

namespace {

// Map nvFuser layout to CUTLASS layout. This is RowMajor if the inner logical
// dimension is the same as the inner allocation dimension. Otherwise it is
// ColumnMajor
std::string mapLayoutToCutlass(const TensorView* tv) {
  NVF_ERROR(!tv->getLogicalDomain().empty());
  return tv->getMaybeAllocationDomain().back() == tv->getLogicalDomain().back()
      ? "cutlass::layout::RowMajor"
      : "cutlass::layout::ColumnMajor";
}

template <typename T>
T* findOp(Fusion* fusion) {
  auto exprs = fusion->exprs();
  const auto smmas = ir_utils::filterByType<T>(exprs.begin(), exprs.end());
  if (smmas.size() != 1) {
    return nullptr;
  }
  return *smmas.begin();
}

} // namespace

MatmulPattern findPattern(Fusion* fusion) {
  if (auto smma = findOp<ScaledMmaOp>(fusion)) {
    NVF_ERROR(
        smma->outScale() == nullptr,
        "Output block scale factor not supported for EVT translation");
    NVF_ERROR(
        smma->outGamma() == nullptr,
        "Output global scale factor not supported for EVT translation");
    return {
        .mma = smma,
        .a = smma->matrix1(),
        .b = smma->matrix2(),
        .a_scale = smma->scale1(),
        .b_scale = smma->scale2(),
        .alpha = smma->alpha(),
        .beta = smma->beta(),
        .bias = smma->bias(),
        .is_grouped = false};
  } else if (auto gmma = findOp<CutlassNvfp4GroupedMmaOp>(fusion)) {
    return {
        .mma = gmma,
        .a = gmma->matrix1(),
        .b = gmma->matrix2(),
        .a_scale = gmma->scale1(),
        .b_scale = gmma->scale2(),
        .alpha = gmma->alpha(),
        .problem_sizes = gmma->problemSizes(),
        .expert_offsets = gmma->expertOffsets(),
        .scale_factor_offsets = gmma->scalingFactorOffsets(),
        .is_grouped = true};
  } else {
    NVF_THROW("Could not find a supported matmul pattern in Fusion");
  }
}

int64_t fusionInputPosition(Fusion* fusion, Val* v) {
  NVF_ERROR(v->isFusionInput());
  return std::distance(
      fusion->inputs().begin(),
      std::find(fusion->inputs().begin(), fusion->inputs().end(), v));
}

int64_t fusionOutputPosition(Fusion* fusion, Val* v) {
  NVF_ERROR(v->isFusionOutput());
  return std::distance(
      fusion->outputs().begin(),
      std::find(fusion->outputs().begin(), fusion->outputs().end(), v));
}

CutlassGeneratedCode generateNvfp4ScaledMmKernel(
    Fusion* fusion,
    const CutlassParams& params) {
  NVF_ERROR(fusion != nullptr);

  // We always need a workspace
  int64_t num_temp_tensors = 1;

  TensorView* main_output = fusion->outputs().front()->as<TensorView>();
  const mma_utils::DataWrapperOpt<EVTModel> model_opt = extractEVTModel(fusion);
  const bool has_evt = model_opt.isValid();
  if (has_evt) {
    main_output = model_opt.getData().getRootTensorView();
  } else {
    NVF_ERROR_EQ(
        fusion->outputs().size(),
        1,
        "Fusions without EVT must have a single output");
  }
  NVF_ERROR(main_output != nullptr);

  const std::string output_dtype = dtypeToCutlass(main_output->dtype());

  const MatmulPattern pattern = findPattern(fusion);

  NVF_ERROR(pattern.mma != nullptr);
  NVF_ERROR(pattern.a->isFusionInput());
  NVF_ERROR(pattern.b->isFusionInput());
  NVF_ERROR(pattern.a_scale->isFusionInput());
  NVF_ERROR(pattern.b_scale->isFusionInput());

  // Validate that the inputs and scale factors are all contiguous
  for (TensorView* tv :
       {pattern.a,
        pattern.b,
        pattern.a_scale,
        pattern.b_scale,
        pattern.alpha,
        pattern.beta,
        pattern.bias,
        pattern.problem_sizes,
        pattern.expert_offsets,
        pattern.scale_factor_offsets}) {
    if (tv == nullptr) {
      continue;
    }
    for (const auto& c : tv->getContiguity()) {
      if (c.has_value()) {
        NVF_ERROR(
            c.value() == true,
            "We require input TensorView ",
            tv->toString(),
            " to be fully contiguous for the Cutlass executor.");
      }
    }
  }

  std::string code =
      R"(
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_conversion.h"
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

// This is a surrogate for a CUDA at::Tensor
struct TensorArg {
  void* data_ptr;
  int64_t dim;
  int64_t* sizes;
  int64_t* strides=nullptr;
};

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
      params.mma_tile.m,
      params.mma_tile.n,
      params.mma_tile.k,
      params.cluster_shape.m,
      params.cluster_shape.n,
      params.cluster_shape.k,
      params.per_sm_tile.m,
      params.per_sm_tile.n,
      params.per_sm_tile.k);

  code += R"(
};

// Main GEMM configuration for NVFP4 scaled matrix multiplication on SM100+
// Defines all the types, layouts, and configurations needed for the CUTLASS
// kernel
struct Fp4GemmSm100 {
  // A matrix configuration
)";
  NVF_ERROR_EQ(
      pattern.a->dtype(),
      DataType::Float4_e2m1fn,
      "Only float_e2m1_t is supported so far");
  code += "  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;\n";
  code += "  using LayoutATag = " + mapLayoutToCutlass(pattern.a) + ";\n";
  // TODO: check alignment of A and save in cutlass_params.supported_vec_sizes
  // as is done for Ampere
  code += R"(
  static constexpr int AlignmentA = 32;

  // B matrix configuration
)";
  NVF_ERROR_EQ(
      pattern.b->dtype(),
      DataType::Float4_e2m1fn,
      "Only float_e2m1_t is supported so far");
  code += "  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;\n";
  code += "  using LayoutBTag = " + mapLayoutToCutlass(pattern.b) + ";\n";
  // TODO: check alignment of B and save in cutlass_params.supported_vec_sizes
  // as is done for Ampere
  code += R"(
  static constexpr int AlignmentB = 32;

  // C/D matrix configuration
)";
  code += "  using ElementD = " + output_dtype + ";\n";
  // This should be void unless there is a bias
  code += "  using ElementC = void;\n";

  NVF_ERROR(
      !main_output->hasAllocation(),
      "Cutlass executor doesn't yet support transposed output");
  code += "  using LayoutDTag = " + mapLayoutToCutlass(main_output) + ";\n";
  // TODO: C is
  code += "  using LayoutCTag = cutlass::layout::RowMajor;\n";

  code += R"(
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  // Avoid division by zero in case ElementC is void
  static constexpr int AlignmentC = 128 / std::max(cutlass::sizeof_bits<ElementC>::value, 1);

  // Kernel functional config
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  // Kernel Perf config
  using MmaTileShape = typename KernelTraits::MmaTileShape;
  using ClusterShape = typename KernelTraits::ClusterShape;
  using PerSmTileShape_MNK = typename KernelTraits::PerSmTileShape_MNK;

  // For OpClassBlockScaledTensorOp, Is2SmMma is true when MmaTileM == 256
  static constexpr bool Is2SmMma = (cute::size<0>(MmaTileShape{}) == 256);
  using TmemWarpShape_MN =
      decltype(cutlass::epilogue::collective::detail::
                   sm100_tmem_warps<Is2SmMma, MmaTileShape>());
  // Compute the actual epilogue tile that will be auto-selected by the builder
  using EpilogueTileShape =
      decltype(cutlass::epilogue::collective::detail::
                   sm100_dense_compute_tile_shape_or_override<
                       OperatorClass,
                       PerSmTileShape_MNK,
                       cutlass::epilogue::collective::EpilogueTileAuto,
                       TmemWarpShape_MN,
                       ElementC,
                       LayoutCTag,
                       ElementD,
                       LayoutDTag,
                       /*IsPerColScaleSupported=*/false>());

)";
  if (has_evt) {
    const EVTModel& evt_model = model_opt.getData();
    code += "  using EVTOp =\n" +
        evt_model.defString(/*node=*/nullptr, /*indent=*/4) + ";\n";
    code += R"(
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
          cutlass::epilogue::collective::EpilogueScheduleAuto,
          EVTOp>::CollectiveOp;
)";
  } else {
    code += R"(
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
)";
  }
  code += R"(

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
typename T::Gemm::Arguments args_from_inputs(
  const std::vector<TensorArg>& inputs) {
  using ElementA = typename T::Gemm::ElementA;
  using ElementB = typename T::Gemm::ElementB;
)";
  // TODO: Handle other scale factor dtypes
  NVF_ERROR_EQ(pattern.a_scale->dtype(), DataType::Float8_e4m3fn);
  code += "  using ElementSFA = cutlass::float_ue4m3_t;\n";
  NVF_ERROR_EQ(pattern.b_scale->dtype(), DataType::Float8_e4m3fn);
  code += "  using ElementSFB = cutlass::float_ue4m3_t;\n";

  code += R"(
  using ElementD = typename T::Gemm::ElementD;
  using ElementCompute = float;
  using StrideA = typename T::StrideA;
  using StrideB = typename T::StrideB;
  using StrideC = typename T::StrideC;
  using StrideD = typename T::StrideD;
  using Sm1xxBlkScaledConfig =
      typename T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
)";

  code += "  const TensorArg& a = inputs.at(" +
      std::to_string(fusionInputPosition(fusion, pattern.a)) + ");\n";
  code += "  const TensorArg& b = inputs.at(" +
      std::to_string(fusionInputPosition(fusion, pattern.b)) + ");\n";
  code += "  const TensorArg& scales_a = inputs.at(" +
      std::to_string(fusionInputPosition(fusion, pattern.a_scale)) + ");\n";
  code += "  const TensorArg& scales_b = inputs.at(" +
      std::to_string(fusionInputPosition(fusion, pattern.b_scale)) + ");\n";
  if (pattern.alpha != nullptr) {
    code += "  const TensorArg& alpha = inputs.at(" +
        std::to_string(fusionInputPosition(fusion, pattern.alpha)) + ");\n";
  }
  if (pattern.beta != nullptr) {
    code += "  const TensorArg& beta = inputs.at(" +
        std::to_string(fusionInputPosition(fusion, pattern.beta)) + ");\n";
  }
  if (pattern.bias != nullptr) {
    code += "  const TensorArg& bias = inputs.at(" +
        std::to_string(fusionInputPosition(fusion, pattern.bias)) + ");\n";
  }
  code +=
      "  const TensorArg& output = inputs.at(" +
      std::to_string(
          fusion->inputs().size() + fusionOutputPosition(fusion, main_output)) +
      ");\n";
  code += "  NVF_ERROR(a.dim == " +
      std::to_string(pattern.a->getLogicalDomain().size()) +
      ", \"Wrong dimension for argument a\");\n";
  code += "  NVF_ERROR(b.dim == " +
      std::to_string(pattern.b->getLogicalDomain().size()) +
      ", \"Wrong dimension for argument b\");\n";
  code += R"(

  int m = static_cast<int>(a.sizes[0]);
  int n = static_cast<int>(b.sizes[1]);
  int k = static_cast<int>(a.sizes[1]) * 2;
  NVF_ERROR(b.sizes[0] == a.sizes[1], "Mismatched K dims");

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(m, n, k, 1));

  typename T::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<ElementA const*>(a.data_ptr),
       stride_A,
       static_cast<ElementB const*>(b.data_ptr),
       stride_B,
       static_cast<ElementSFA const*>(scales_a.data_ptr),
       layout_SFA,
       static_cast<ElementSFB const*>(scales_b.data_ptr),
       layout_SFB},
      {// Epilogue arguments
)";
  if (has_evt) {
    code += model_opt.getData().argString(/*node=*/nullptr, /*indent=*/4);
  } else {
    code += "       {}";
  }
  code += ",  // epilogue.thread\n";
  if (pattern.bias != nullptr) {
    code += "       bias.data_ptr,";
  } else {
    code += "       /*bias=*/nullptr,";
  }
  code += R"(
       stride_C,
       static_cast<ElementD*>(output.data_ptr),
       stride_D}};
)";
  if (!has_evt && (pattern.alpha != nullptr || pattern.beta != nullptr)) {
    // Passing alpha and beta by name is for the default epilogue only
    code += "  auto& fusion_args = arguments.epilogue.thread;\n";
    if (pattern.alpha != nullptr) {
      code +=
          "  fusion_args.alpha_ptr = static_cast<ElementCompute "
          "const*>(alpha.data_ptr);\n";
    }
    if (pattern.beta != nullptr) {
      code +=
          "  fusion_args.beta_ptr = static_cast<ElementCompute "
          "const*>(beta.data_ptr);\n";
    }
  }
  code += R"(
  return arguments;
}

} // namespace

// Calling code should pass a pointer to a vector of TensorArgs
extern "C" void temp_tensor_size(
    int64_t* out_tensor_sizes,
    const std::vector<TensorArg>& inputs) {
  auto arguments = args_from_inputs<Fp4GemmSm100>(inputs);
  out_tensor_sizes[0] = Fp4GemmSm100::Gemm::get_workspace_size(arguments);
}

extern "C" void init_temp_tensors(uint8_t** temp_tensors) {
  // TODO: do stuff here other than workspace initialization
  // The cutlass workspace _is_ a temporary tensor, but since it needs
  // arguments to be built in order to initialize it, I currently left it in
  // run_kernel to avoid needing to call args_from_inputs twice.
}

// Executes the FP4 scaled matrix multiplication using CUTLASS kernels
//
// This function orchestrates the GEMM operation by setting up the kernel,
// allocating workspace memory, and running the computation on the GPU.
// It handles the complete lifecycle from kernel initialization to execution.
extern "C" void run_kernel(
    const std::vector<TensorArg>& inputs,
    uint8_t** temp_tensor_ptrs,
    cudaStream_t stream) {
  typename Fp4GemmSm100::Gemm gemm;

  auto arguments = args_from_inputs<Fp4GemmSm100>(inputs);

  auto can_implement_status = gemm.can_implement(arguments);
  NVF_ERROR(
      can_implement_status == cutlass::Status::kSuccess,
      "Failed to implement GEMM");

  uint8_t* workspace_ptr = temp_tensor_ptrs[0];
  auto status = gemm.initialize(arguments, workspace_ptr, stream);
  NVF_ERROR(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm.run(arguments, workspace_ptr, stream);
  NVF_ERROR(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}
)";

  return {code, num_temp_tensors};
}

} // namespace cutlass_codegen

} // namespace nvfuser
