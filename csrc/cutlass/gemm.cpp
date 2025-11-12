// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cutlass/block_scaling.h>
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
  const std::vector<IterDomain*> nored_logical =
      TensorDomain::noReductions(tv->getLogicalDomain());
  NVF_CUTLASS_REJECT_IF(
      nored_logical.size() != 2,
      tv->toString(),
      " has dimension ",
      nored_logical.size(),
      " but only dimension 2 tensors are supported");
  const std::vector<IterDomain*> nored_alloc =
      TensorDomain::noReductions(tv->getMaybeAllocationDomain());
  return nored_alloc.back() == nored_logical.back()
      ? "cutlass::layout::RowMajor"
      : "cutlass::layout::ColumnMajor";
}

class CutlassCodeGenerator {
 public:
  static std::string generate(Fusion* fusion, const CutlassParams& params) {
    CutlassCodeGenerator gen(fusion, params);
    gen.run();
    return gen.code_;
  }

  static std::string getRejectReason(Fusion* fusion) {
    try {
      const CutlassParams params;
      CutlassCodeGenerator gen(fusion, params);
      gen.gatherInfo();
    } catch (const UnsupportedFusion& e) {
      return e.what();
    }
    return "";
  }

 private:
  CutlassCodeGenerator(Fusion* fusion, const CutlassParams& params)
      : fusion_(fusion), params_(params) {
    NVF_ERROR(fusion_ != nullptr);
  }

  void findPattern() {
    pattern_ = findCutlassMatmulPattern(fusion_);

    // These must always be set
    NVF_ERROR(pattern_.mma != nullptr);
    NVF_ERROR(pattern_.a != nullptr);
    NVF_ERROR(pattern_.b != nullptr);

    NVF_CUTLASS_REJECT_IF(
        pattern_.a->dtype() != DataType::Float4_e2m1fn,
        "Only NVFP4 inputs are supported but tensor ",
        pattern_.a->toString(),
        " has A role and dtype is ",
        pattern_.a->dtype());
    NVF_CUTLASS_REJECT_IF(
        pattern_.b->dtype() != DataType::Float4_e2m1fn,
        "Only NVFP4 inputs are supported but tensor ",
        pattern_.b->toString(),
        " has B role and dtype is ",
        pattern_.b->dtype());

    NVF_CUTLASS_REJECT_IF(
        !pattern_.a->isFusionInput(), "A must be a fusion input");
    NVF_CUTLASS_REJECT_IF(
        !pattern_.b->isFusionInput(), "B must be a fusion input");

    NVF_CUTLASS_REJECT_IF(
        pattern_.a_scale == nullptr, "Could not find A scale factors");
    NVF_CUTLASS_REJECT_IF(
        pattern_.b_scale == nullptr, "Could not find B scale factors");

    NVF_CUTLASS_REJECT_IF(
        !pattern_.a_scale->isFusionInput(),
        "Scale factors for A must be a fusion input");
    NVF_CUTLASS_REJECT_IF(
        !pattern_.b_scale->isFusionInput(),
        "Scale factors for B must be a fusion input");

    // Validate that the inputs and scale factors are all contiguous
    for (TensorView* tv :
         {pattern_.a, pattern_.b, pattern_.a_scale, pattern_.b_scale}) {
      if (tv == nullptr) {
        continue;
      }
      for (const auto& c : tv->getContiguity()) {
        if (c.has_value()) {
          NVF_CUTLASS_REJECT_IF(
              c.value() != true,
              "We require input TensorView ",
              tv->toString(),
              " to be fully contiguous for the Cutlass executor.");
        }
      }
    }

    NVF_CUTLASS_REJECT_IF(
        !pattern_.mma->isA<ScaledMmaOp>(),
        "Only ScaledMmaOp is supported so far");
    NVF_CUTLASS_REJECT_IF(
        pattern_.is_grouped, "Grouped patterns are not yet supported");
  }

  // Gathers necessary info from fusion_ but does not start generating code. If
  // this method succeeds then we are able to schedule this fusion, so this can
  // be used in a canScheduleCompile check
  void gatherInfo() {
    findPattern();

    evt_model_ = std::make_unique<EVTModel>(extractEVTModel(fusion_));

    block_scaled_outputs_ = findBlockScaledOutputs(fusion_);
    NVF_CUTLASS_REJECT_IF(
        block_scaled_outputs_.size() > 1,
        "At most one block scaled output is currently supported");
    if (block_scaled_outputs_.empty()) {
      main_output_ = fusion_->outputs().front()->as<TensorView>();
    } else {
      main_output_ = block_scaled_outputs_.front().quantized_output;
    }
  }

  void generateCode() {
    // Fill the preamble first
    code_ += R"(#include "cutlass/cutlass.h"
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
)";
    genParams();

    genGemmConfigClass();

    genArgumentsFunction();

    code_ += R"(
} // namespace
)";
    genRunKernel();
  }

  //! Here we put all the CutlassParams into the KernelTraits struct in the
  //! generated code.
  void genParams() {
    code_ += R"(
// Kernel configuration traits for different output data types
// Defines tile shapes and cluster configurations.
struct KernelTraits {
)";
    code_ += std::format(
        R"(
  using MmaTileShape = Shape<_{}, _{}, _{}>;
  using ClusterShape = Shape<_{}, _{}, _{}>;
  using PerSmTileShape_MNK = Shape<_{}, _{}, _{}>;
)",
        params_.mma_tile.m,
        params_.mma_tile.n,
        params_.mma_tile.k,
        params_.cluster_shape.m,
        params_.cluster_shape.n,
        params_.cluster_shape.k,
        params_.per_sm_tile.m,
        params_.per_sm_tile.n,
        params_.per_sm_tile.k);

    code_ += R"(
};
)";
  }

  void genGemmConfigClass() {
    code_ += R"(
// Main GEMM configuration for NVFP4 scaled matrix multiplication on SM100+
// Defines all the types, layouts, and configurations needed for the CUTLASS
// kernel
struct Fp4GemmSm100 {
)";
    genMatrixDescription(pattern_.a, "A", /*is_nvfp4=*/true);
    genMatrixDescription(pattern_.b, "B", /*is_nvfp4=*/true);

    // TODO: support bias here as C
    genMatrixDescription(nullptr, "C", /*is_nvfp4=*/false);

    genMatrixDescription(main_output_, "D", /*is_nvfp4=*/false);

    // Sets up basic
    genBasicConfig();

    genEpilogueConfig();

    genFinalGemmConfig();

    code_ += R"(
};
)";
  }

  //! Define
  //!   - ElementA
  //!   - LayoutATag
  //!   - AlignmentA
  //! where "A" == tv_name. If is_nvfp4 is true, then we use
  //! cutlass::nv_float4_t<> to pack the nvfuser DataType, as is required for
  //! operands to cutlass block scaled GEMM.
  void genMatrixDescription(
      TensorView* tv,
      const std::string& tv_name,
      bool is_nvfp4) {
    // TODO: alignment of each gmem tensor should be recorded in CutlassParams
    // and used here instead.

    // set dtype to void for null tensors. This is used to represent missing
    // bias
    std::string dtype = "void";
    if (tv != nullptr) {
      dtype = is_nvfp4 ? "cutlass::nv_float4_t<cutlass::float_e2m1_t>"
                       : dtypeToCutlass(tv->dtype());
    }
    std::string layout =
        tv == nullptr ? "cutlass::layout::RowMajor" : mapLayoutToCutlass(tv);
    int64_t alignment_bits =
        tv == nullptr ? 128 : 128 / dataTypeSizeBit(tv->dtype());
    code_ += std::format(
        R"(
  using Element{0} = {1};
  using Layout{0}Tag = {2};
  static constexpr int Alignment{0} = {3};

    )",
        tv_name,
        dtype,
        layout,
        alignment_bits);
  }

  void genBasicConfig() {
    code_ += R"(
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
  }

  void genEpilogueConfig() {
    NVF_ERROR(evt_model_.get() != nullptr);
    code_ += "  using EVTOp =\n" +
        evt_model_->defString(/*node=*/nullptr, /*indent=*/4) + ";\n";
    code_ += R"(
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
  }

  void genFinalGemmConfig() {
    code_ += R"(
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
)";
  }

  void genArgumentsFunction() {
    // TODO: Handle other scale factor dtypes
    code_ += R"(
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
typename Fp4GemmSm100::Gemm::Arguments args_from_inputs(
  const std::vector<TensorArg>& inputs) {
  using T = Fp4GemmSm100;

  using ElementA = typename T::Gemm::ElementA;
  using ElementB = typename T::Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
)";
    if (pattern_.bias != nullptr) {
      code_ += "  using ElementC = " + dtypeToCutlass(pattern_.bias->dtype()) +
          ";\n";
    }
    code_ += R"(
  using ElementD = typename T::Gemm::ElementD;
  using ElementCompute = float;
  using StrideA = typename T::StrideA;
  using StrideB = typename T::StrideB;
  using StrideC = typename T::StrideC;
  using StrideD = typename T::StrideD;
  using Sm1xxBlkScaledConfig =
      typename T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
)";
    const auto maybe_define =
        [&](std::string tv_name, TensorView* tv, bool is_output) {
          if (tv == nullptr) {
            return;
          }
          int64_t pos = is_output
              ? fusionOutputPosition(fusion_, tv) + fusion_->inputs().size()
              : fusionInputPosition(fusion_, tv);
          code_ += "  const TensorArg& " + tv_name + " = inputs.at(" +
              std::to_string(pos) + ");\n";
        };
    maybe_define("a", pattern_.a, /*is_output=*/false);
    maybe_define("b", pattern_.b, /*is_output=*/false);
    maybe_define("a_scale", pattern_.a_scale, /*is_output=*/false);
    maybe_define("b_scale", pattern_.b_scale, /*is_output=*/false);
    maybe_define("alpha", pattern_.alpha, /*is_output=*/false);
    maybe_define("beta", pattern_.beta, /*is_output=*/false);
    maybe_define("bias", pattern_.bias, /*is_output=*/false);
    maybe_define("output", main_output_, /*is_output=*/true);

    // TODO: handle finding mnk for differently sized dimensions of a and b
    code_ += R"(
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
       static_cast<ElementSFA const*>(a_scale.data_ptr),
       layout_SFA,
       static_cast<ElementSFB const*>(b_scale.data_ptr),
       layout_SFB},
      {// Epilogue arguments
)";
    code_ += evt_model_->argString(/*node=*/nullptr, /*indent=*/4);
    code_ += ",  // epilogue.thread\n";
    if (pattern_.bias != nullptr) {
      code_ += "       bias.data_ptr,";
    } else {
      code_ += "       /*bias=*/nullptr,";
    }
    code_ += R"(
       stride_C,
       static_cast<ElementD*>(output.data_ptr),
       stride_D}};
  return arguments;
}
)";
  }

  void genRunKernel() {
    code_ += R"(

// Calling code should pass a pointer to a vector of TensorArgs
extern "C" size_t workspace_size(void* input_ptr) {
  const std::vector<TensorArg>& inputs =
      *reinterpret_cast<const std::vector<TensorArg>*>(input_ptr);
  auto arguments = args_from_inputs(inputs);
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

  auto arguments = args_from_inputs(inputs);

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
  }

  void run() {
    gatherInfo();

    generateCode();
  }

 private:
  Fusion* fusion_;
  const CutlassParams& params_;

  CutlassMatmulPattern pattern_;
  TensorView* main_output_ = nullptr;

  std::unique_ptr<EVTModel> evt_model_ = nullptr;

  std::vector<BlockScaledOutputPattern> block_scaled_outputs_;

  std::string code_;
};

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

int64_t fusionInputPosition(Fusion* fusion, Val* v) {
  NVF_CUTLASS_REJECT_IF(
      !v->isFusionInput(), "Expected ", v->toString(), " to be a fusion input");
  return std::distance(
      fusion->inputs().begin(),
      std::find(fusion->inputs().begin(), fusion->inputs().end(), v));
}

int64_t fusionOutputPosition(Fusion* fusion, Val* v) {
  NVF_CUTLASS_REJECT_IF(
      !v->isFusionOutput(),
      "Expected ",
      v->toString(),
      " to be a fusion output");
  return std::distance(
      fusion->outputs().begin(),
      std::find(fusion->outputs().begin(), fusion->outputs().end(), v));
}

CutlassMatmulPattern findCutlassMatmulPattern(Fusion* fusion) {
  if (auto smma = findOp<ScaledMmaOp>(fusion)) {
    NVF_CUTLASS_REJECT_IF(
        smma->outScale() != nullptr,
        "Output block scale factor not supported for EVT translation");
    NVF_CUTLASS_REJECT_IF(
        smma->outGamma() != nullptr,
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
    NVF_CUTLASS_REJECT("Could not find a supported matmul pattern in Fusion");
  }
}

std::string generateNvfp4ScaledMmKernel(
    Fusion* fusion,
    const CutlassParams& params) {
  return CutlassCodeGenerator::generate(fusion, params);
}

std::string getGemmRejectReason(Fusion* fusion) {
  return CutlassCodeGenerator::getRejectReason(fusion);
}

} // namespace cutlass_codegen

} // namespace nvfuser
