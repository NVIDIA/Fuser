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
#include <ranges>
#include <string>

namespace nvfuser {

namespace cutlass_codegen {

namespace {

// Map nvFuser layout to CUTLASS layout. This is RowMajor if the inner logical
// dimension is the same as the inner allocation dimension. Otherwise it is
// ColumnMajor
std::string mapLayoutToCutlass(const TensorView* tv) {
  auto nored_logical = tv->getLogicalDomain() | TensorDomain::kNoReductions;
  const size_t ndims = std::ranges::distance(nored_logical);

  NVF_CUTLASS_REJECT_IF(
      ndims < 2,
      tv->toString(),
      " has dimension ",
      ndims,
      " but only dimension 2 or higher tensors are supported");

  auto nored_alloc =
      tv->getMaybeAllocationDomain() | TensorDomain::kNoReductions;
  return *std::ranges::rbegin(nored_logical) ==
          *std::ranges::rbegin(nored_alloc)
      ? "cutlass::layout::RowMajor"
      : "cutlass::layout::ColumnMajor";
}

class CutlassCodeGenerator {
 public:
  static CutlassGeneratedCode generate(
      Fusion* fusion,
      const CutlassParams& params) {
    CutlassCodeGenerator gen(fusion, params);
    gen.run();
    return {gen.code_, gen.num_temp_tensors_};
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

    fusion_->printMath();
    std::cout << pattern_.toString() << std::endl;

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

    NVF_CUTLASS_REJECT_IF(
        pattern_.a_scale->dtype() != DataType::Float8_e4m3fn,
        "Expected A scale factors to be fp8");
    NVF_CUTLASS_REJECT_IF(
        pattern_.b_scale->dtype() != DataType::Float8_e4m3fn,
        "Expected B scale factors to be fp8");

    // Validate that the inputs and scale factors are all contiguous
    for (TensorView* tv :
         {pattern_.a, pattern_.b, pattern_.a_scale, pattern_.b_scale}) {
      if (tv == nullptr) {
        continue;
      }
      const std::vector<std::optional<bool>>& contiguity = tv->getContiguity();
      const bool is_tv_contiguous =
          std::all_of(contiguity.begin(), contiguity.end(), [](auto c) {
            return c.value_or(true);
          });
      NVF_CUTLASS_REJECT_IF(
          !is_tv_contiguous,
          "We require input TensorView ",
          tv->toString(),
          " to be fully contiguous for the Cutlass executor.");
    }
  }

  // Gathers necessary info from fusion_ but does not start generating code. If
  // this method succeeds then we are able to schedule this fusion, so this can
  // be used in a canScheduleCompile check
  void gatherInfo() {
    findPattern();

    block_scaled_outputs_ = findBlockScaledOutputs(fusion_);
    NVF_CUTLASS_REJECT_IF(
        block_scaled_outputs_.size() > 1,
        "At most one block scaled output is currently supported");
    if (block_scaled_outputs_.empty()) {
      main_output_ = fusion_->outputs().front()->as<TensorView>();
    } else {
      main_output_ = block_scaled_outputs_.front().quantized_output;
    }

    // There is always a workspace tensor, even though it might be empty
    num_temp_tensors_ = 1;

    // Build a map from tensors to pointer arrays
    if (pattern_.is_grouped) {
      // There are always going to be pointer arrays for A, B, A_sf, B_sf. There
      // is also one for each output and one for each _epilogue_ input.
      auto register_temp_tensor = [&](TensorView* tv) {
        temp_tensor_map_.emplace(tv, num_temp_tensors_++);
      };
      for (Val* inp : fusion_->inputs()) {
        if (auto* tv = dynamic_cast<TensorView*>(inp); tv &&
            inp != pattern_.problem_sizes && inp != pattern_.expert_offsets &&
            inp != pattern_.scale_factor_offsets && tv->nDims() > 0) {
          register_temp_tensor(tv);
        }
      }
      for (Val* outp : fusion_->outputs()) {
        if (auto* tv = dynamic_cast<TensorView*>(outp)) {
          register_temp_tensor(tv);
        }
      }
    }

    evt_model_ =
        std::make_unique<EVTModel>(extractEVTModel(fusion_, temp_tensor_map_));
  }

  void generateCode() {
    genPreamble();

    code_ += R"(
namespace {
using namespace cute;
)";
    genParams();

    genGemmConfigClass();

    genInputMapping();

    genArgumentsFunction();

    code_ += "} // namespace\n";

    genRunKernel();
  }

  void genPreamble() {
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
)";
  }

  //! Here we put all the CutlassParams into the Params struct in the
  //! generated code.
  void genParams() {
    code_ += R"(
// Kernel configuration traits for different output data types
// Defines tile shapes and cluster configurations.
struct Params {
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

    genMatrixDescription(pattern_.bias, "C", /*is_nvfp4=*/false);

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
  using MmaTileShape = typename Params::MmaTileShape;
  using ClusterShape = typename Params::ClusterShape;
  using PerSmTileShape_MNK = typename Params::PerSmTileShape_MNK;

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

  //! Generates a function mapping from vector<TensorArg> to a struct describing
  //! the problem independent of input position
  void genInputMapping() {
    code_ += R"(
struct Inputs {
  int m;
  int n;
  int k;
)";
    // Generate typed pointer fields for each tensor
    auto add_field =
        [&](std::string tv_name, TensorView* tv, bool force_unsigned = false) {
          if (tv == nullptr) {
            return;
          }
          std::string dtype = dtypeToCutlass(tv->dtype(), force_unsigned);
          // Determine if this is const or mutable based on whether it's an
          // output
          bool is_output = tv->isFusionOutput();
          code_ += "  " + dtype;
          if (!is_output) {
            code_ += " const";
          }
          code_ += "* " + tv_name + ";\n";
        };

    add_field("a", pattern_.a);
    add_field("b", pattern_.b);
    // Scale factors need to be unsigned for cutlass
    add_field("a_scale", pattern_.a_scale, /*force_unsigned=*/true);
    add_field("b_scale", pattern_.b_scale, /*force_unsigned=*/true);
    add_field("alpha", pattern_.alpha);
    add_field("beta", pattern_.beta);
    add_field("bias", pattern_.bias);
    add_field("main_output", main_output_);

    // Add block scaled output fields if they exist
    if (!block_scaled_outputs_.empty()) {
      NVF_ERROR_EQ(
          block_scaled_outputs_.size(),
          1,
          "Currently at most one block scaled output is supported");
      add_field(
          "main_output_block_scale_factor",
          block_scaled_outputs_[0].block_scale_factors);
      add_field(
          "main_output_global_scale_factor",
          block_scaled_outputs_[0].global_scale_factor);
    }

    code_ += R"(};

// Map vectors of inputs to an Inputs struct
Inputs standardize_args(const std::vector<TensorArg>& inputs) {
  Inputs result;
)";
    auto maybe_add_mapping =
        [&](std::string tv_name, TensorView* tv, bool is_output) {
          if (tv == nullptr) {
            return;
          }
          int64_t pos = is_output
              ? fusionOutputPosition(fusion_, tv) + fusion_->inputs().size()
              : fusionInputPosition(fusion_, tv);
          std::string dtype = dtypeToCutlass(tv->dtype());
          code_ += "  result." + tv_name + " = static_cast<" + dtype;
          if (!is_output) {
            code_ += " const";
          }
          code_ += "*>(inputs.at(" + std::to_string(pos) + ").data_ptr);\n";
        };
    maybe_add_mapping("a", pattern_.a, /*is_output=*/false);
    maybe_add_mapping("b", pattern_.b, /*is_output=*/false);
    // Scale factors need special handling to match CUTLASS types
    if (pattern_.a_scale != nullptr) {
      int64_t pos = fusionInputPosition(fusion_, pattern_.a_scale);
      code_ +=
          "  result.a_scale = static_cast<cutlass::float_ue4m3_t "
          "const*>(inputs.at(" +
          std::to_string(pos) + ").data_ptr);\n";
    }
    if (pattern_.b_scale != nullptr) {
      int64_t pos = fusionInputPosition(fusion_, pattern_.b_scale);
      code_ +=
          "  result.b_scale = static_cast<cutlass::float_ue4m3_t "
          "const*>(inputs.at(" +
          std::to_string(pos) + ").data_ptr);\n";
    }
    maybe_add_mapping("alpha", pattern_.alpha, /*is_output=*/false);
    maybe_add_mapping("beta", pattern_.beta, /*is_output=*/false);
    maybe_add_mapping("bias", pattern_.bias, /*is_output=*/false);
    maybe_add_mapping("main_output", main_output_, /*is_output=*/true);

    // Add block scaled output mappings if they exist
    if (!block_scaled_outputs_.empty()) {
      maybe_add_mapping(
          "main_output_block_scale_factor",
          block_scaled_outputs_[0].block_scale_factors,
          /*is_output=*/true);
      // The global scale factor is actually a fusion input
      maybe_add_mapping(
          "main_output_global_scale_factor",
          block_scaled_outputs_[0].global_scale_factor,
          /*is_output=*/false);
    }

    code_ += R"(
  // Extract m, n, k from tensor dimensions
  const TensorArg& a_arg = inputs.at()";
    code_ += std::to_string(fusionInputPosition(fusion_, pattern_.a));
    code_ += R"();
  const TensorArg& b_arg = inputs.at()";
    code_ += std::to_string(fusionInputPosition(fusion_, pattern_.b));
    code_ += R"();
  result.m = static_cast<int>(a_arg.sizes[0]);
  result.n = static_cast<int>(b_arg.sizes[1]);
  result.k = static_cast<int>(a_arg.sizes[1]) * 2;
)";
    // A has size [M, K] for grouped or ungrouped GEMM
    // B has size [E, N, K] for grouped GEMM
    // B has size [K, N] for ungrouped GEMM
    // In either case, N is b_arg.sizes[1]
    if (pattern_.is_grouped) {
      code_ +=
          "  NVF_ERROR(b_arg.sizes[2] == a_arg.sizes[1], \"Mismatched K dims\");\n";
    } else {
      code_ +=
          "  NVF_ERROR(b_arg.sizes[0] == a_arg.sizes[1], \"Mismatched K dims\");\n";
    }
    code_ += R"(
  return result;
}
)";
  }

  void genArgumentsFunction() {
    // TODO: Handle other scale factor dtypes
    code_ += R"(
// Constructs CUTLASS GEMM arguments from standardized inputs
//
// This function converts the Inputs struct into the format
// expected by CUTLASS GEMM kernels, including proper stride calculations
// and layout configurations for the scaled matrix multiplication.
//
// Returns CUTLASS GEMM arguments structure ready for kernel execution
typename Fp4GemmSm100::Gemm::Arguments cutlass_args_from_inputs(
  const Inputs& args) {
  using T = Fp4GemmSm100;

  using ElementA = typename T::Gemm::ElementA;
  using ElementB = typename T::Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementC = typename T::Gemm::ElementC;
  using ElementD = typename T::Gemm::ElementD;
  using ElementCompute = float;
  using StrideA = typename T::StrideA;
  using StrideB = typename T::StrideB;
  using StrideC = typename T::StrideC;
  using StrideD = typename T::StrideD;
  using Sm1xxBlkScaledConfig =
      typename T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {inputs.m, inputs.k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {inputs.n, inputs.k, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {inputs.m, inputs.n, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {inputs.m, inputs.n, 1});

  // TODO: these should be pointer arrays for grouped GEMM
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(inputs.m, inputs.n, inputs.k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(inputs.m, inputs.n, inputs.k, 1));

  typename T::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {inputs.m, inputs.n, inputs.k, 1},
      {// Mainloop arguments
       inputs.a,
       stride_A,
       inputs.b,
       stride_B,
       inputs.a_scale,
       layout_SFA,
       inputs.b_scale,
       layout_SFB},
      {// Epilogue arguments
)";
    code_ += evt_model_->argString(/*node=*/nullptr, /*indent=*/4);
    code_ += ",  // epilogue.thread\n";
    if (pattern_.bias != nullptr) {
      code_ += "       inputs.bias,";
    } else {
      code_ += "       /*bias=*/nullptr,";
    }
    code_ += R"(
       stride_C,
       inputs.main_output,
       stride_D}};
  return arguments;
}
)";
  }

  void genRunKernel() {
    genTempTensorSizes();

    genInitTempTensors();

    code_ += R"(
// Executes the FP4 scaled matrix multiplication using CUTLASS kernels
//
// This function orchestrates the GEMM operation by setting up the kernel,
// allocating workspace memory, and running the computation on the GPU.
// It handles the complete lifecycle from kernel initialization to execution.
extern "C" void run_kernel(
    const std::vector<TensorArg>& tensor_args,
    uint8_t** temp_tensor_ptrs,
    cudaStream_t stream) {
  typename Fp4GemmSm100::Gemm gemm;

  Inputs inputs = standardize_args(tensor_args);
  auto cutlass_args = cutlass_args_from_inputs(inputs);

  auto can_implement_status = gemm.can_implement(cutlass_args);
  NVF_ERROR(
      can_implement_status == cutlass::Status::kSuccess,
      "Failed to implement GEMM");

  init_temp_tensors(inputs, cutlass_args, stream);

  auto status = gemm.initialize(cutlass_args, workspace_ptr, stream);
  NVF_ERROR(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  uint8_t* workspace_ptr = temp_tensor_ptrs[0];

  status = gemm.run(cutlass_args, workspace_ptr, stream);
  NVF_ERROR(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}
)";
  }

  void genTempTensorSizes() {
    code_ += R"(
// Calling code should pass a pointer to a vector of TensorArgs
extern "C" void temp_tensor_sizes(
    int64_t* out_tensor_sizes,
    const std::vector<TensorArg>& inputs,
    uint8_t** temp_tensor_ptrs) {
  auto arguments = args_from_inputs(inputs, temp_tensor_ptrs);
  out_tensor_sizes[0] = Fp4GemmSm100::Gemm::get_workspace_size(arguments);
)";

    if (pattern_.is_grouped) {
      // TODO: For grouped gemm, we need one temp tensor for each grouped input
      // and output. These are the pointer arrays and they are all the same
      // size: [num_experts].
      NVF_ERROR(
          pattern_.expert_offsets != nullptr,
          "expert_offsets must be provided for grouped GEMM");
      code_ += "  const TensorArg& expert_offsets = inputs.at(" +
          std::to_string(
                   fusionInputPosition(fusion_, pattern_.expert_offsets)) +
          ");\n";
      code_ += R"(
  int64_t num_experts = expert_offsets.sizes[0];
  // All pointer arrays are the same size since they represent an array of
  // base pointers, so they are independent of the dimension of the tensor or
  // the inner dimensions of each group.
  int64_t ptr_array_bytes = num_experts * sizeof(int64_t);
)";
      for (auto i : arange(1, num_temp_tensors_)) {
        code_ += "  out_tensor_sizes[" + std::to_string(i) +
            "] = ptr_array_bytes;\n";
      };
    }

    code_ += R"(
}
)";
  }

  void genGetPointerArrays() {
    code_ += R"(
// CUDA kernel to compute memory offsets and layout information for grouped GEMM
// operations
//
// This kernel calculates the starting pointers and layout configurations for
// each expert in a grouped matrix multiplication.
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
template <typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
void run_get_group_gemm_starts(uint8_t** temp_tensor_ptrs,
    const Fp4GemmSm100::Gemm::Arguments& arguments,
    const std::vector<TensorArg>& inputs,
    cudaStream_t stream) {
  const int num_experts = (int)expert_offsets.size(0);



  NVF_CHECK(
      out_tensors.size(1) == N,
      "Output tensor shape doesn't match expected shape");
  NVF_CHECK(
      K / 2 == b_tensors.size(2),
      "b_tensors(dim = 2) and a_tensors(dim = 1) trailing"
      " dimension must match");

  // TODO: handle large number of experts by splitting into multiple CTAs

  get_group_gemm_starts<<<1, num_experts, 0, stream>>>(
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
}
)";
  }

  void genInitTempTensors() {
    if (pattern_.is_grouped) {
      genGetPointerArrays();
    }

    code_ += R"(
void init_temp_tensors(uint8_t** temp_tensor_ptrs,
    const Fp4GemmSm100::Gemm::Arguments& arguments,
    const std::vector<TensorArg>& inputs,
    cudaStream_t stream) {
  typename Fp4GemmSm100::Gemm gemm;

  // TODO: do stuff here other than workspace initialization
  // The cutlass workspace _is_ a temporary tensor, but since it needs
  // arguments to be built in order to initialize it, I currently left it in
  // run_kernel to avoid needing to call args_from_inputs twice.

  uint8_t* workspace_ptr = temp_tensor_ptrs[0];
  auto status = gemm.initialize(arguments, workspace_ptr, stream);
  NVF_ERROR(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");
)";
    if (pattern_.is_grouped) {
      code_ +=
          "  run_get_group_gemm_starts(temp_tensor_ptrs, inputs, stream);\n";
    }
    code_ += "}\n";
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

  // We require one temp tensor for the CUTLASS workspace. For grouped gemm, we
  // also need a temp tensor for each pointer array
  int64_t num_temp_tensors_ = -1;

  // Map from TensorView to position of temp tensors. Currently this is only
  // used to map each input and output to a temporary pointer array in grouped
  // GEMM
  std::unordered_map<TensorView*, int64_t> temp_tensor_map_;

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

std::string CutlassMatmulPattern::toString() const {
  std::stringstream ss;
  ss << "CutlassMatmulPattern{\n";
#define PRINTATTR(attr)                                                      \
  ss << "  " #attr " = " << (attr == nullptr ? "nullptr" : attr->toString()) \
     << "\n";
  PRINTATTR(mma);
  PRINTATTR(a);
  PRINTATTR(b);
  PRINTATTR(a_scale);
  PRINTATTR(b_scale);
  PRINTATTR(alpha);
  PRINTATTR(beta);
  PRINTATTR(bias);
  PRINTATTR(problem_sizes);
  PRINTATTR(expert_offsets);
  PRINTATTR(scale_factor_offsets);
#undef PRINTATTR
  ss << "  is_grouped = " << std::boolalpha << is_grouped << "\n";
  ss << "}";
  return ss.str();
}

CutlassMatmulPattern findCutlassMatmulPattern(Fusion* fusion) {
  if (auto* smma = findOp<ScaledMmaOp>(fusion)) {
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
  } else if (auto* gmma = findOp<CutlassNvfp4GroupedMmaOp>(fusion)) {
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

CutlassGeneratedCode generateNvfp4ScaledMmKernel(
    Fusion* fusion,
    const CutlassParams& params) {
  return CutlassCodeGenerator::generate(fusion, params);
}

std::string getGemmRejectReason(Fusion* fusion) {
  return CutlassCodeGenerator::getRejectReason(fusion);
}

} // namespace cutlass_codegen

} // namespace nvfuser
