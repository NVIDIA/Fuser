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

  // We track tensors so that we can refer to them by name in the generated
  // kernel, e.g. inputs.a or inputs.bias instead of args[0] and args[4]. This
  // means that for each gmem TensorView including Fusion inputs, outputs, and
  // temporary buffers allocated by the CutlassCompiledKernel, we associate a
  // descriptor that holds the name of the field in the generated Inputs struct
  // as well as where to find the data pointer in the vector of TensorArgs
  // provided by CutlassCompiledKernel.
  struct TensorDescriptor {
    // This is the name of the attribute in the generated Inputs struct
    std::string name;

    // String representing the cutlass dtype of a single element
    std::string dtype_str;

    // TensorView this temp tensor corresponds to, if it is a temporary pointer
    // array. Note that this can be nullptr, for example for cutlass_workspace.
    TensorView* tv = nullptr;

    // Position in the TensorArg array
    int tensor_args_pos = -1;

    // If there is a pointer array associated with this
    TensorDescriptor* pointer_array_desc = nullptr;
  };

  // We track every global tensor. If tv is non-null and we detect that this is
  // a grouped GEMM, then we will also register a temporary pointer array and
  // associate it with tv
  TensorDescriptor* registerGlobalBuffer(
      std::string name,
      TensorView* tv = nullptr,
      bool needs_ptr_array = false,
      std::string dtype_str = "",
      bool ptr_array_dtype_is_same = false) {
    auto new_temp_tensor_pos = [&]() {
      return fusion_->inputs().size() + fusion_->outputs().size() +
          num_temp_tensors_++;
    };

    if (dtype_str.empty()) {
      dtype_str = (tv == nullptr) ? "uint8_t" : dtypeToCutlass(tv->dtype());
    }

    NVF_ERROR(
        !tensor_names_.contains(name), "Tried to register ", name, " twice");
    tensor_names_.insert(name);

    int tensor_arg_pos = -1;
    if (tv == nullptr) {
      tensor_arg_pos = new_temp_tensor_pos();
    } else if (tv->isFusionInput()) {
      tensor_arg_pos = fusionInputPosition(fusion_, tv);
    } else if (tv->isFusionOutput()) {
      tensor_arg_pos = fusionOutputPosition(fusion_, tv);
    } else {
      NVF_ERROR(
          "We should never call registerGlobalBuffer on intermediate tensors");
    }

    std::string ptr_array_dtype_str =
        ptr_array_dtype_is_same ? dtype_str : dtype_str + "*";
    TensorDescriptor* pointer_array_desc = needs_ptr_array
        ? registerGlobalBuffer(
              name + "_ptrs",
              /*tv=*/nullptr,
              /*needs_ptr_array=*/false,
              ptr_array_dtype_str)
        : nullptr;

    tensor_name_map_.emplace(
        tv, needs_ptr_array ? pointer_array_desc->name : name);

    return tensor_descriptors_
        .emplace_back(std::make_unique<TensorDescriptor>(
            name, dtype_str, tv, tensor_arg_pos, pointer_array_desc))
        .get();
  }

  // Gathers necessary info from fusion_ but does not start generating code. If
  // this method succeeds then we are able to schedule this fusion, so this can
  // be used in a canScheduleCompile check
  void gatherInfo() {
    findPattern();

    registerGlobalBuffer("cutlass_workspace");

#define MAYBE_REGISTER(field)                                          \
  if (pattern_.field != nullptr) {                                     \
    registerGlobalBuffer(#field, pattern_.field, pattern_.is_grouped); \
  }
#define MAYBE_REGISTER_NO_PTR_ARRAY(field)               \
  if (pattern_.field != nullptr) {                       \
    registerGlobalBuffer(#field, pattern_.field, false); \
  }

    MAYBE_REGISTER(a)
    MAYBE_REGISTER(b)
    MAYBE_REGISTER(alpha)
    MAYBE_REGISTER(beta)
    MAYBE_REGISTER(bias)
    // These do not need pointer arrays since they describe the grouping of
    // grouped GEMM
    MAYBE_REGISTER_NO_PTR_ARRAY(problem_sizes)
    MAYBE_REGISTER_NO_PTR_ARRAY(expert_offsets)
    MAYBE_REGISTER_NO_PTR_ARRAY(scale_factor_offsets)
#undef MAYBE_REGISTER

    // We handle a_scale and b_scale separately so we can specify that their
    // dtypes are unsigned
    registerGlobalBuffer(
        "a_scale",
        pattern_.a_scale,
        /*needs_ptr_array=*/pattern_.is_grouped,
        dtypeToCutlass(pattern_.a_scale->dtype(), /*force_unsigned=*/true));
    registerGlobalBuffer(
        "b_scale",
        pattern_.b_scale,
        /*needs_ptr_array=*/pattern_.is_grouped,
        dtypeToCutlass(pattern_.b_scale->dtype(), /*force_unsigned=*/true));

    block_scaled_outputs_ = findBlockScaledOutputs(fusion_);
    NVF_CUTLASS_REJECT_IF(
        block_scaled_outputs_.size() > 1,
        "At most one block scaled output is currently supported");
    if (block_scaled_outputs_.empty()) {
      main_output_ = fusion_->outputs().front()->as<TensorView>();
    } else {
      main_output_ = block_scaled_outputs_.front().quantized_output;
    }
    registerGlobalBuffer(
        "main_output", main_output_, /*needs_ptr_array=*/pattern_.is_grouped);

    auto register_multiple = [&](const std::vector<TensorView*>& tvs,
                                 const std::string& base_name,
                                 bool ptr_array_dtype_is_same = false) {
      size_t cur_tv = 0;
      for (TensorView* tv : tvs) {
        std::string number = tvs.size() > 1 ? std::to_string(cur_tv++) : "";
        registerGlobalBuffer(
            base_name + number,
            tv,
            /*needs_ptr_array=*/pattern_.is_grouped,
            "",
            ptr_array_dtype_is_same);
      }
    };

    std::vector<TensorView*> quantized_outputs;
    std::vector<TensorView*> block_scale_factors;
    std::vector<TensorView*> global_scale_factors;
    for (const BlockScaledOutputPattern& bs_output : block_scaled_outputs_) {
      if (bs_output.quantized_output != main_output_) {
        quantized_outputs.push_back(bs_output.quantized_output);
      }
      block_scale_factors.push_back(bs_output.block_scale_factors);
      global_scale_factors.push_back(bs_output.global_scale_factor);
    }
    register_multiple(quantized_outputs, "quantized_output");
    register_multiple(block_scale_factors, "block_scale_factors");
    register_multiple(
        global_scale_factors,
        "global_scale_factor",
        /*ptr_array_dtype_is_same=*/true);

    // Register other epilogue inputs
    std::vector<TensorView*> epilogue_inputs;
    for (Val* inp : fusion_->inputs()) {
      if (auto* tv = dynamic_cast<TensorView*>(inp)) {
        if (tensor_name_map_.find(tv) == tensor_name_map_.end()) {
          epilogue_inputs.push_back(tv);
        }
      }
    }
    register_multiple(epilogue_inputs, "epilogue_input");

    // Register other outputs that are not the main output and are not block
    // scaled.
    std::vector<TensorView*> unquantized_outputs;
    for (Val* outp : fusion_->outputs()) {
      if (auto* tv = dynamic_cast<TensorView*>(outp)) {
        if (tensor_name_map_.find(tv) == tensor_name_map_.end()) {
          unquantized_outputs.push_back(tv);
        }
      }
    }
    register_multiple(unquantized_outputs, "unquantized_output");

    evt_model_ =
        std::make_unique<EVTModel>(extractEVTModel(fusion_, tensor_name_map_));
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
    if (pattern_.is_grouped) {
      code_ += "  int num_experts;\n";
    }

    // Find all tensors with pointer array mappings
    for (const std::unique_ptr<TensorDescriptor>& td_ptr :
         tensor_descriptors_) {
      code_ += "  " + td_ptr->dtype_str + "* " + td_ptr->name + ";\n";
    }
    code_ += R"(};

// Map vectors of inputs to an Inputs struct
Inputs standardize_args(const std::vector<TensorArg>& tensor_args) {
  Inputs result;
)";
    for (const std::unique_ptr<TensorDescriptor>& td_ptr :
         tensor_descriptors_) {
      code_ += "  result." + td_ptr->name + " = reinterpret_cast<" +
          td_ptr->dtype_str + "*>(tensor_args.at(" +
          std::to_string(td_ptr->tensor_args_pos) + ").data_ptr);\n";
    }
    code_ += R"(
  // Extract m, n, k from tensor dimensions
  const TensorArg& a_arg = tensor_args.at()";
    code_ += std::to_string(fusionInputPosition(fusion_, pattern_.a));
    code_ += R"();
  const TensorArg& b_arg = tensor_args.at()";
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
          "  NVF_ERROR(b_arg.sizes[2] == a_arg.sizes[1], \"Mismatched K "
          "dims\");\n";
    } else {
      code_ +=
          "  NVF_ERROR(b_arg.sizes[0] == a_arg.sizes[1], \"Mismatched K "
          "dims\");\n";
    }
    if (pattern_.is_grouped) {
      code_ += R"(
  const TensorArg& expert_offsets_arg = tensor_args.at()";
      code_ +=
          std::to_string(fusionInputPosition(fusion_, pattern_.expert_offsets));
      code_ += R"();
  result.num_experts = expert_offsets_arg.sizes[0];
)";
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
    const Inputs& inputs) {
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
    cudaStream_t stream) {
  typename Fp4GemmSm100::Gemm gemm;

  Inputs inputs = standardize_args(tensor_args);
  auto cutlass_args = cutlass_args_from_inputs(inputs);

  auto can_implement_status = gemm.can_implement(cutlass_args);
  NVF_ERROR(
      can_implement_status == cutlass::Status::kSuccess,
      "Failed to implement GEMM");

  init_temp_tensors(inputs, cutlass_args, stream);

  status = gemm.run(cutlass_args, inputs.cutlass_workspace, stream);
  NVF_ERROR(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}
)";
  }

  void genTempTensorSizes() {
    code_ += R"(
// Calling code should pass a pointer to a vector of TensorArgs
extern "C" void temp_tensor_sizes(
    int64_t* out_tensor_sizes,
    const std::vector<TensorArg>& tensor_args) {
  Inputs inputs = standardize_args(tensor_args);
  auto cutlass_args = cutlass_args_from_inputs(inputs);

  out_tensor_sizes[0] = Fp4GemmSm100::Gemm::get_workspace_size(cutlass_args);
)";

    if (pattern_.is_grouped) {
      // TODO: For grouped gemm, we need one temp tensor for each grouped input
      // and output. These are the pointer arrays and they are all the same
      // size: [num_experts].
      code_ += R"(
  // All pointer arrays are the same size since they represent an array of
  // base pointers, so they are independent of the dimension of the tensor or
  // the inner dimensions of each group.
  int64_t ptr_array_bytes = inputs.num_experts * sizeof(void*);
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
    NVF_ERROR(pattern_.is_grouped);

    code_ += R"(
// CUDA kernel to compute memory offsets and layout information for grouped GEMM
// operations
//
// This kernel calculates the starting pointers and layout configurations for
// each expert in a grouped matrix multiplication.
__global__ void get_group_gemm_starts(Inputs inputs) {
  int64_t expert_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert_id >= inputs.num_experts) {
    return;
  }
  // Upcast from int32_t to int64_t to avoid overflow
  // during offset calculations
  int64_t expert_offset = static_cast<int64_t>(inputs.expert_offsets[expert_id]);
  int64_t sf_offset = static_cast<int64_t>(sf_offsets[expert_id]);

  // The block size for nvfp4.
  constexpr int64_t nvfp4_block_size = 16;

  int64_t m = static_cast<int64_t>(inputs.problem_sizes[expert_id * 3]);
  int64_t n = static_cast<int64_t>(inputs.problem_sizes[expert_id * 3 + 1]);
  int64_t k = static_cast<int64_t>(inputs.problem_sizes[expert_id * 3 + 2]);
  assert(
      (m >= 0 && n == inputs.n && k == inputs.k && k % 2 == 0) && "Unexpected problem sizes");

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

inline int ceilDiv(int a, int b) {
  return (a + b - 1) / b;
}

// Launches the CUDA kernel to compute memory offsets and layout information for
// grouped GEMM
//
// This function launches the get_group_gemm_starts kernel with appropriate
// template parameters based on the output data type. It handles the setup and
// execution of the offset computation kernel for grouped matrix multiplication
// operations.
void run_get_group_gemm_starts(const Inputs& inputs, cudaStream_t stream) {
  int threads_per_block = 256;
  int num_blocks = ceilDiv(inputs.num_experts, threads_per_block);

  get_group_gemm_starts<<<num_blocks, threads_per_block, 0, stream>>>(inputs);
}
)";
  }

  void genInitTempTensors() {
    if (pattern_.is_grouped) {
      genGetPointerArrays();
    }

    code_ += R"(
void init_temp_tensors(const Inputs& inputs,
    const Fp4GemmSm100::Gemm::Arguments& cutlass_args,
    cudaStream_t stream) {
  typename Fp4GemmSm100::Gemm gemm;

  auto status = gemm.initialize(cutlass_args, inputs.cutlass_workspace, stream);
  NVF_ERROR(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");
)";
    if (pattern_.is_grouped) {
      code_ += "  run_get_group_gemm_starts(inputs, stream);\n";
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

  int64_t num_temp_tensors_ = 0;

  std::vector<std::unique_ptr<TensorDescriptor>> tensor_descriptors_;

  std::unordered_set<std::string> tensor_names_;

  // Map from TensorView to position of input, output, and temp tensors.
  std::unordered_map<TensorView*, std::string> tensor_name_map_;

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
