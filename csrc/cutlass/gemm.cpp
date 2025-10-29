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

} // namespace

ScaledMmaOp* findScaledMmaOp(Fusion* fusion) {
  auto exprs = fusion->exprs();
  const auto smmas =
      ir_utils::filterByType<ScaledMmaOp>(exprs.begin(), exprs.end());
  if (smmas.size() != 1) {
    return nullptr;
  }
  return *smmas.begin();
}

std::string generateNvfp4ScaledMmKernel(
    Fusion* fusion,
    const CutlassParams& params) {
  NVF_ERROR(fusion != nullptr);
  NVF_ERROR_EQ(
      fusion->outputs().size(),
      1,
      "Cutlass executor currently only supports a single output");
  auto* main_output = fusion->outputs().front()->as<TensorView>();
  const std::string output_dtype = dtypeToCutlass(main_output->dtype());

  ScaledMmaOp* smma = findScaledMmaOp(fusion);
  NVF_ERROR(smma != nullptr);

  TensorView* a = smma->matrix1();
  TensorView* b = smma->matrix2();
  TensorView* a_scale = smma->scale1();
  TensorView* b_scale = smma->scale2();
  TensorView* bias = smma->bias();

  NVF_ERROR(a->isFusionInput());
  NVF_ERROR(b->isFusionInput());
  NVF_ERROR(a_scale->isFusionInput());
  NVF_ERROR(b_scale->isFusionInput());

  // Validate that the inputs and scale factors are all contiguous
  for (TensorView* tv : {a, b, a_scale, b_scale}) {
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
  using EpilogueTileShape_MNK = Shape<_{}, _{}, _{}>;
)",
      params.mma_tile.m,
      params.mma_tile.n,
      params.mma_tile.k,
      params.cluster_shape.m,
      params.cluster_shape.n,
      params.cluster_shape.k,
      params.per_sm_tile.m,
      params.per_sm_tile.n,
      params.per_sm_tile.k,
      params.epilogue_tile.m,
      params.epilogue_tile.n,
      params.epilogue_tile.k);

  // This replicates the CUTLASS logic for determining whether the mma uses 1
  // or 2 SMs on blackwell. Note that for non-blockscaled mmas,
  // params.mma_tile.m can also be 128 and use 2SM but we currently only
  // support block-scaled inputs.
  const bool is_2sm =
      params.cluster_shape.m % 2 == 0 && params.mma_tile.m == 256;

  if (is_2sm) {
    code +=
        "  using EpilogueScheduleType = "
        "cutlass::epilogue::TmaWarpSpecialized2Sm;\n";
  } else {
    code +=
        "  using EpilogueScheduleType = "
        "cutlass::epilogue::TmaWarpSpecialized1Sm;\n";
  }
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
  code += "  using LayoutATag = " + mapLayoutToCutlass(a) + ";\n";
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
  code += "  using LayoutBTag = " + mapLayoutToCutlass(b) + ";\n";
  // TODO: check alignment of B and save in cutlass_params.supported_vec_sizes
  // as is done for Ampere
  code += R"(
  static constexpr int AlignmentB = 32;

  // C/D matrix configuration
)";
  code += "  using ElementD = " + output_dtype + ";\n";
  if (bias != nullptr) {
    code += "  using ElementC = " + dtypeToCutlass(bias->dtype()) + ";\n";
  } else {
    code += "  using ElementC = " + output_dtype + ";\n";
  }

  NVF_ERROR(
      !main_output->hasAllocation(),
      "Cutlass executor doesn't yet support transposed output");
  code += "  using LayoutDTag = " + mapLayoutToCutlass(main_output) + ";\n";
  // TODO: C is
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
  using EpilogueTileShape_MNK = typename KernelTraits::EpilogueTileShape_MNK;
  using EpilogueScheduleType = typename KernelTraits::EpilogueScheduleType;
  using EpilogueTileShape = Shape<_64, _64>;

  using EpilogueDescriptor = cutlass::epilogue::collective::detail::Sm100EpilogueDescriptor<
    OperatorClass,
    MmaTileShape,
    EpilogueTileShape,
    ElementAccumulator,
    ElementC,
    ElementD,
    EpilogueScheduleType,
    cutlass::gemm::TagToStrideC_t<LayoutCTag>,
    cutlass::gemm::TagToStrideC_t<LayoutDTag>,
    // NOTE: The following flags are only affect the default epilogue and are
    // ignored when passing a custom epilogue
    false, // IsPerColScaleSupported
    false  // IsBlockScaleSupported
  >;
)";
  const mma_utils::DataWrapperOpt<EVTModel> model_opt = extractEVTModel(fusion);
  const bool has_evt = model_opt.isValid();
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
  NVF_ERROR_EQ(a_scale->dtype(), DataType::Float8_e4m3fn);
  code += "  using ElementSFA = cutlass::float_ue4m3_t;\n";
  NVF_ERROR_EQ(b_scale->dtype(), DataType::Float8_e4m3fn);
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
      std::to_string(fusionInputPosition(fusion, a)) + ");\n";
  code += "  const TensorArg& b = inputs.at(" +
      std::to_string(fusionInputPosition(fusion, b)) + ");\n";
  code += "  const TensorArg& scales_a = inputs.at(" +
      std::to_string(fusionInputPosition(fusion, a_scale)) + ");\n";
  code += "  const TensorArg& scales_b = inputs.at(" +
      std::to_string(fusionInputPosition(fusion, b_scale)) + ");\n";
  if (smma->alpha() != nullptr) {
    code += "  const TensorArg& alpha = inputs.at(" +
        std::to_string(fusionInputPosition(fusion, smma->alpha())) + ");\n";
  }
  if (smma->beta() != nullptr) {
    code += "  const TensorArg& beta = inputs.at(" +
        std::to_string(fusionInputPosition(fusion, smma->beta())) + ");\n";
  }
  if (smma->bias() != nullptr) {
    code += "  const TensorArg& bias = inputs.at(" +
        std::to_string(fusionInputPosition(fusion, smma->bias())) + ");\n";
  }
  NVF_ERROR_EQ(fusion->outputs().size(), 1);
  code +=
      "  const TensorArg& output = inputs.at(" +
      std::to_string(
          fusion->inputs().size() + fusionOutputPosition(fusion, main_output)) +
      ");\n";
  code +=
      "  NVF_ERROR(a.dim == " + std::to_string(a->getLogicalDomain().size()) +
      ", \"Wrong dimension for argument a\");\n";
  code +=
      "  NVF_ERROR(b.dim == " + std::to_string(b->getLogicalDomain().size()) +
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
  if (bias != nullptr) {
    code += "       static_cast<" + dtypeToCutlass(bias->dtype()) +
        "*>(bias.data_ptr),";
  } else {
    code += "       /*bias=*/nullptr,";
  }
  code += R"(
       stride_C,
       static_cast<ElementD*>(output.data_ptr),
       stride_D}};
)";
  if (!has_evt && (smma->alpha() != nullptr || smma->beta() != nullptr)) {
    // Passing alpha and beta by name is for the default epilogue only
    code += "  auto& fusion_args = arguments.epilogue.thread;\n";
    if (smma->alpha() != nullptr) {
      code +=
          "  fusion_args.alpha_ptr = static_cast<ElementCompute "
          "const*>(alpha.data_ptr);\n";
    }
    if (smma->beta() != nullptr) {
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
extern "C" size_t workspace_size(void* input_ptr) {
  const std::vector<TensorArg>& inputs =
      *reinterpret_cast<const std::vector<TensorArg>*>(input_ptr);
  auto arguments = args_from_inputs<Fp4GemmSm100>(inputs);
  return Fp4GemmSm100::Gemm::get_workspace_size(arguments);
}

// Executes the FP4 scaled matrix multiplication using CUTLASS kernels
//
// This function orchestrates the GEMM operation by setting up the kernel,
// allocating workspace memory, and running the computation on the GPU.
// It handles the complete lifecycle from kernel initialization to execution.
extern "C" void run_kernel(
    const std::vector<TensorArg>& inputs,
    uint8_t* workspace_ptr,
    cudaStream_t stream) {
  typename Fp4GemmSm100::Gemm gemm;

  auto arguments = args_from_inputs<Fp4GemmSm100>(inputs);

  auto can_implement_status = gemm.can_implement(arguments);
  NVF_ERROR(
      can_implement_status == cutlass::Status::kSuccess,
      "Failed to implement GEMM");

  auto status = gemm.initialize(arguments, workspace_ptr, stream);
  NVF_ERROR(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm.run(arguments, workspace_ptr, stream);
  NVF_ERROR(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}
)";

  return code;
}

} // namespace cutlass_codegen

} // namespace nvfuser
