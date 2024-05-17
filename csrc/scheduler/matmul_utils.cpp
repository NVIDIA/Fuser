// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/matmul_heuristic.h>
#include <scheduler/matmul_heuristic_plugin.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/mma_utils.h>
#include <scheduler/registry.h>

// NOTE: included to avoid compilation error caused by missing destructor in
// 'SchedulerRuntimeInfo'
#include <debug.h>
#include <executor_utils.h>
#include <id_model/id_model.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/utils.h>
#include <options.h>
#include <val_graph.h>
#include <algorithm>
#include <deque>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <type_traits>
#include <utility>
#include <variant>
#include "ATen/cuda/CUDAContext.h"
#include "mma_type.h"
#include "mma_utils.h"
#include "type.h"
#include "utils.h"

namespace nvfuser {
namespace {

//! Access to the structure should be done with labels defined in MatmulDomain.
using ProblemShape = std::array<int64_t, 4>;

//! A helper for deciding the type of MMA op for given fusion and problem shape.
inline std::optional<MmaMacro> getMmaOp(
    const int dev_version,
    const ProblemShape& problem) {
  using MacroType = MmaMacro;

  // NOTE: A temp condition
  const ProblemShape::value_type n_extend = problem[(size_t)MatmulDomain::N];
  const bool use_small_n = ((n_extend % 8) == 0) && ((n_extend % 16) != 0);

  switch (dev_version) {
    case 75:
      return (use_small_n) ? MacroType::Turing_16_8_16
                           : MacroType::Turing_16_16_16;
    case 80:
    case 86:
    case 89:
    case 90: // NOTE: temp use ampere matmul for hopper
      return (use_small_n) ? MacroType::Ampere_16_8_16
                           : MacroType::Ampere_16_16_16;
    default:
      return std::nullopt;
  }
}

//! A wrapper for core heuristics initialization.
//! We should have already set params->mma_macro before calling this function.
inline bool initCoreHeuristics(
    std::shared_ptr<MatmulParams> params,
    const ProblemShape& problem_shape,
    const mma_utils::RolesMap& roles_map) {
  const GemmTile instruction_tile = getMmaOpShape(params->mma_macro);
  GemmTile warp_tile = {-1, -1, -1};
  GemmTile cta_tile = {-1, -1, -1};

  using DimType = decltype(GemmTile::m);

  // warp tile shape
  {
    // Initial target:
    // - 1 MMA ops per thread in a warp (32 threads), warp tile should be
    //   then 32x bigger than instruction tile,
    // - start with [4, 4, 2] shape, later it should depend on problem
    //   shape and have bigger impact on CTA tile shape

    const DimType m_ratio = 4;
    const DimType n_ratio = 4;
    const DimType k_ratio = 2;

    warp_tile = {
        instruction_tile.m * m_ratio,
        instruction_tile.n * n_ratio,
        instruction_tile.k * k_ratio};
  }

  // cta tile shape
  {
    // Initial target:
    // - 4 warp tiles per CTA
    // - CTA k-dim should be same as warp tile k-dim

    DimType m_ratio = 2;
    DimType n_ratio = 2;

    const auto mn_ratio = (double)problem_shape[(size_t)MatmulDomain::M] /
        (double)problem_shape[(size_t)MatmulDomain::N];
    if (mn_ratio < 0.5) {
      m_ratio = 1;
      n_ratio = 4;
    } else if (mn_ratio > 2) {
      m_ratio = 4;
      n_ratio = 1;
    }

    cta_tile = {warp_tile.m * m_ratio, warp_tile.n * n_ratio, warp_tile.k};
  }

  params->tile_sizes = {cta_tile, warp_tile, instruction_tile};

  // stages and async mem copy
  {
    // NOTE: compilation errors when async is enabled on Turing devices
    if (isAmpere(params->mma_macro)) {
      constexpr int stages = 3;

      params->double_buffer_options.double_buffer_smem_write = true;
      params->double_buffer_options.double_buffer_smem_read = true;
      params->double_buffer_options.smem_double_buffer_stage = stages;
    }
  }

  const auto& roleMinDtypeSize = [&roles_map](MatmulRole role) -> int64_t {
    const auto op_it = roles_map.find(role);
    NVF_ERROR(op_it != roles_map.end());
    int64_t min_size_bytes = 128LL;
    for (const TensorView* operand : op_it->second) {
      min_size_bytes = std::min(min_size_bytes, dataTypeSize(operand->dtype()));
    }
    return min_size_bytes;
  };
  params->async_gmem_load_operands = isCpAsyncOperandLoadSupported(
      params.get(),
      roleMinDtypeSize(MatmulRole::INPUT_A),
      roleMinDtypeSize(MatmulRole::INPUT_B));

  if (!params->async_gmem_load_operands) {
    // Circular buffering requires async load. If we cannot use async load due
    // to unsupported vectorization width, then we can only double buffer at
    // most.
    params->double_buffer_options.smem_double_buffer_stage =
        std::min(2, params->double_buffer_options.smem_double_buffer_stage);
  }
  return true;
}

//! A helper for getting problem shape from fusion and runtime info.
//!
//! For a given domain, try to find the size by evaluating the extent of an
//! IterDomain in each group of that domain type. For example, if there are
//! multiple Batch dimensions, we find all ValGroups that are mapped as
//! MatmulDomain::Batch, we evaluate the extent of each, then we multiply those
//! dimensions together to get the overall batch size.
ProblemShape getProblemShape(
    const std::unordered_map<ValGroup, MatmulDomain>& group_to_domain,
    SchedulerRuntimeInfo& runtime_info) {
  ProblemShape shape{1, 1, 1, 1};
  for (const auto& [g, dom] : group_to_domain) {
    NVF_ERROR(!g->empty());
    IterDomain* id = g->front()->as<IterDomain>();
    const PolymorphicValue extent =
        runtime_info.expressionEvaluator().evaluate(id->extent());
    NVF_ERROR(
        extent.hasValue(), "Could not evaluate extent of ", id->toString());
    shape[(size_t)dom] *= extent.as<int64_t>();
  }
  return shape;
}

std::string isMatmulFusionDefinitionSupported(
    Fusion* fusion,
    const mma_utils::MatmulPattern& pattern,
    const mma_utils::RolesMap& roles_map,
    const std::unordered_map<ValGroup, MatmulDomain>& id_roles) {
  const auto& fusion_inputs = fusion->inputs();
  const auto& fusion_outputs = fusion->outputs();
  std::vector<TensorView*> mma_inputs = {pattern.A, pattern.B};
  const auto mma_output = pattern.output;

  const auto fusion_inputs_tvs =
      ir_utils::filterByType<TensorView>(fusion_inputs).vector();
  const auto fusion_outputs_tvs =
      ir_utils::filterByType<TensorView>(fusion_outputs).vector();

  constexpr size_t minimal_number_of_inputs = 2;

  // Quick checks - MmaOp
  {
    // Check if MmaOp represents gemm (requires M/N/K == 1, B == 0)
    //  or bgemm (requires M/N/K/B == 1)
    std::array<int64_t, 4> num_axes{};
    for (const auto& [g, dom] : id_roles) {
      num_axes[(size_t)dom]++;
    }
    constexpr size_t expected_axes_numbers = 1;
    if (num_axes[(size_t)MatmulDomain::M] != expected_axes_numbers ||
        num_axes[(size_t)MatmulDomain::N] != expected_axes_numbers ||
        num_axes[(size_t)MatmulDomain::K] != expected_axes_numbers ||
        num_axes[(size_t)MatmulDomain::Batch] > expected_axes_numbers) {
      return "MmaOp has unsupported number of one of M/N/K/Batch axes";
    }

    if (!mma_output->hasReduction()) {
      return "MMA output TV has no reduction domain";
    }
  }

  // Quick checks - Fusion
  {
    // Fusion should contain at least two inputs (for now)
    if (minimal_number_of_inputs > fusion_inputs.size()) {
      return "Fusion inputs contain at least one non-TensorView object";
    }
  }

  // Fusion topology check
  {
    auto entry = roles_map.find(MatmulRole::INPUT_A);
    std::set<TensorView*> tvs_with_roles;

    if (entry != roles_map.end()) {
      if (MATMUL_CORE_ROLES_EXPECTED_COUNT == entry->second.size()) {
        tvs_with_roles.insert(entry->second.begin(), entry->second.end());
      } else {
        return "There is more than a single fusion input that can be MMA first input";
      }
    } else {
      return "No candidate in fusion inputs for MMA first input";
    }

    entry = roles_map.find(MatmulRole::INPUT_B);
    if (entry != roles_map.end()) {
      if (MATMUL_CORE_ROLES_EXPECTED_COUNT == entry->second.size()) {
        tvs_with_roles.insert(entry->second.begin(), entry->second.end());
      } else {
        return "There is more than a single fusion input that can be MMA second input";
      }
    } else {
      return "No candidate in fusion inputs for MMA second input";
    }

    entry = roles_map.find(MatmulRole::OUTPUT_D);
    if (entry != roles_map.end()) {
      tvs_with_roles.insert(entry->second.begin(), entry->second.end());
    } else {
      return "No candidate in fusion outputs MMA output";
    }

    // Non-core input roles are optional, no requirements for definitions
    entry = roles_map.find(MatmulRole::INPUT_C);
    if (entry != roles_map.end()) {
      tvs_with_roles.insert(entry->second.begin(), entry->second.end());
    }

    // Non-core output roles are optional, no requirements for definitions
    entry = roles_map.find(MatmulRole::OUTPUT_AUX);
    if (entry != roles_map.end()) {
      tvs_with_roles.insert(entry->second.begin(), entry->second.end());
    }

    const auto in_out_tvs_count =
        fusion_inputs_tvs.size() + fusion_outputs_tvs.size();
    if (in_out_tvs_count != tvs_with_roles.size()) {
      return "Detected input/output TVs without assigned roles";
    }
  }

  // Check that no non-trivial allocation domains are set on inputs or outputs.
  // TODO: Lift this requirement once we have proper allocation domain support
  for (Val* inp : fusion->inputs()) {
    if (auto tv = dynamic_cast<TensorView*>(inp);
        tv && !ir_utils::hasTrivialAllocationDomain(tv)) {
      return "detected input TV with non-trivial allocation domain";
    }
  }
  for (Val* outp : fusion->outputs()) {
    if (auto tv = dynamic_cast<TensorView*>(outp);
        tv && !ir_utils::hasTrivialAllocationDomain(tv)) {
      return "detected output TV with non-trivial allocation domain";
    }
  }

  return "";
}

// Assume that tens has a contiguous dimension, and that we will load rows of
// that dimension, without merging with another dimension first. Then determine
// the maximum vectorization that can be used.
//
// If the argument has no contiguous dimensions, then a vectorization width of 1
// is returned.
//
// The sizes and strides given should be in the same order as one another and
// should match the no-reductions allocation domain of tv.
//
// These rows can start at any multiple of the non-contiguous strides, so we
// seek the largest power of 2 that divides all those other dimensions (capped
// to 16) as well as the data pointer.
int64_t maxUnpredicatedRowVectorization(
    TensorView* tv,
    const int64_t data_ptr_int,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides) {
  // Check data pointer alignment
  int64_t vec_size = scheduler_utils::maxVectorizationWidth(data_ptr_int);
  vec_size = std::min(vec_size, 16l);
  vec_size /= dataTypeSize(tv->dtype());
  vec_size = std::max(vec_size, 1l);
  if (vec_size == 1l) {
    return vec_size;
  }

  // Check that inner dimension is contiguous
  NVF_ERROR(sizes.size() == strides.size());
  NVF_ERROR((int64_t)sizes.size() == tv->nDims());
  size_t inner_dim_pos = 0;
  for (size_t i = tv->getMaybeAllocationDomain().size() - 1; i >= 0; --i) {
    IterDomain* id = tv->getMaybeAllocationDomain()[i];
    if (id->isReduction() || id->isBroadcast()) {
      continue;
    }
    inner_dim_pos = i;
    std::optional<bool> c = tv->getContiguity().at(i);
    NVF_ERROR(c.has_value());
    if (!c.value()) {
      // If TensorView is marked discontiguous in inner dimension, we cannot
      // vectorize regardless of input.
      return 1l;
    } else {
      NVF_CHECK(
          strides[i] == 1,
          "TensorView ",
          tv->toString(),
          " has marked contiguous inner dimension ",
          id->toString(),
          " but provided tensor has stride ",
          strides[i],
          " in that dimension.");
    }
    break; // only check innermost realized dimension
  }

  // Since this is unpredicated vectorization, the size of the innermost
  // dimension must be a multiple of the vectorization factor.
  vec_size = std::min(
      vec_size, scheduler_utils::maxVectorizationWidth(sizes[inner_dim_pos]));

  // Account for misaligned rows due to outer strides
  for (size_t i : c10::irange(inner_dim_pos)) {
    if (sizes[i] == 1) {
      // outer size-1 dimensions don't affect vectorizability
      continue;
    }
    vec_size =
        std::min(vec_size, scheduler_utils::maxVectorizationWidth(strides[i]));
  }

  return vec_size;
}

MatmulParams::SupportedVectorization getSupportedVectorization(
    const mma_utils::RolesMap& roles_map,
    SchedulerRuntimeInfo& runtime_info) {
  auto getMinVectorization = [&roles_map,
                              &runtime_info](MatmulRole role) -> int64_t {
    int64_t vec_size = 16; // max vectorization size
    const auto it = roles_map.find(role);
    if (it == roles_map.end()) {
      return 16;
    }
    for (TensorView* tv : it->second) {
      // TODO: handle the case when tv is not a Fusion input by filling default
      // contiguous strides based on tv->getMaybeAllocationDomain() and data
      // pointer aligned to 16 bytes.
      int64_t v = maxUnpredicatedRowVectorization(
          tv,
          (int64_t)runtime_info.ptrOf(tv),
          runtime_info.getInputAllocationSizes(tv),
          runtime_info.getInputAllocationStrides(tv));
      if (v < vec_size) {
        vec_size = v;
      }
      if (v == 1) {
        // No need to continue analyzing if we know we cannot vectorize
        break;
      }
    }
    return vec_size;
  };
  MatmulParams::SupportedVectorization supported_vec_size;
  supported_vec_size.a = getMinVectorization(MatmulRole::INPUT_A);
  supported_vec_size.b = getMinVectorization(MatmulRole::INPUT_B);
  // Currently we set epilogue to the max vectorization supported by all outputs
  // and all "C" type input dtypes.
  // See https://github.com/NVIDIA/Fuser/issues/2169
  supported_vec_size.epilogue = 16l;
  // We will write OUTPUT_D role tensors in the default stride order. So
  // vectorization is based on the inner dimension
  const auto d_it = roles_map.find(MatmulRole::OUTPUT_D);
  NVF_ERROR(d_it != roles_map.end(), "Could not find any output D tensors");
  for (TensorView* tv : d_it->second) {
    const int64_t N =
        runtime_info.expressionEvaluator()
            .evaluate(TensorDomain::noReductions(tv->getRootDomain())
                          .back()
                          ->extent())
            .as<int64_t>();
    supported_vec_size.epilogue =
        std::min(supported_vec_size.epilogue, 16l / dataTypeSize(tv->dtype()));
    supported_vec_size.epilogue = std::min(
        supported_vec_size.epilogue, scheduler_utils::maxVectorizationWidth(N));
  }
  // For INPUT_C role tensors, we do not necessarily know which axis we would
  // like to vectorize, so we set vectorization based on dtype instead here
  // until a more complete analysis is implemented.
  if (const auto c_it = roles_map.find(MatmulRole::INPUT_C);
      c_it != roles_map.end()) {
    for (TensorView* tv : c_it->second) {
      supported_vec_size.epilogue = std::min(
          supported_vec_size.epilogue, 16l / dataTypeSize(tv->dtype()));
    }
  }
  return supported_vec_size;
}

} // anonymous namespace

std::string getMatmulRunTimeRejectReason(
    Fusion* fusion,
    HeuristicSummary* data_cache,
    SchedulerRuntimeInfo& runtime_info) {
  // TODO: add proper set of checks
  return "";
}

// The analysis is based on mul-sum pair pattern, detected in the provided
//  fusion definition. If detected and properties of an instance of such
//  pattern are valid then it will be later replaced with MmaOp in
//  fusion definition.
// For the time being a direct instance of MmaOp is also accepted and handled
//  by the analysis.
std::string getMatmulCompileTimeRejectReason(Fusion* fusion) {
  // The plan:
  // 1. Check supported device
  // 2. Check if there is exactly one MmaOp or suitable mul sum pair
  // defined in the fusion.
  // 3. Check if matmul scheduler is enabled
  // 4. Check if inputs to the mma op or mul sum pair match any of
  // supported inputs layout
  // 5. Check if fusion represents expressions that are recognized by matmul
  // scheduler.

  // #1
  // Use a dummy problem shape to determine whether this is a supported device.
  {
    const auto device_prop = at::cuda::getCurrentDeviceProperties();
    const auto mma_op = getMmaOp(
        device_prop->major * 10 + device_prop->minor, {128, 128, 128, 1});
    if (!mma_op.has_value()) {
      return "Unsupported device compute capability";
    }
  }

  // #2
  // Initializing the machinery to check if there's a Mul-Sum pair
  // can be replaced by a Mma Op.
  std::vector<mma_utils::MatmulPattern> patterns =
      mma_utils::findMatmulPatterns(fusion);
  if (patterns.empty()) {
    return "No matmul patterns were found";
  }
  if (patterns.size() > 1) {
    return "Only a single matmul pattern can currently be fused";
  }

  // #1
  {
    if (!isOptionEnabled(EnableOption::FuseMatmul)) {
      // Check for MatmulOp or LinearOp. If found, then only fuse if option is
      // specified
      Expr* op = patterns.front().output->definition();
      if (op->isA<MatmulOp>() /* || op->isA<LinearOp>()*/) {
        return "Matmul fusion is disabled by default. Enable it using NVFUSER_ENABLE=fuse_matmul";
      }
    }
  }

  // Prepare an IdModel which will be reused to check remaining conditions
  IdModel id_model(fusion);
  const auto id_roles = patterns.front().getDimRoles(id_model);
  const mma_utils::RolesMapOpt roles_map_opt =
      mma_utils::getTensorsRoles(fusion, id_model, id_roles);
  if (!roles_map_opt.isValid()) {
    return {roles_map_opt.getErrorMsg()};
  }
  mma_utils::RolesMap roles_map = roles_map_opt.getData();

  // #4
  const auto input_layout_opt =
      mma_utils::getProblemLayout(id_model, id_roles, roles_map);
  if (!input_layout_opt.isValid()) {
    return input_layout_opt.getErrorMsg();
  }

  // #5
  {
    auto support_status = isMatmulFusionDefinitionSupported(
        fusion, patterns.front(), roles_map, id_roles);
    if (!support_status.empty()) {
      return support_status;
    }
  }

  return "";
}

bool isCpAsyncOperandLoadSupported(
    const MatmulParams* params,
    int64_t dtype_size_a,
    int64_t dtype_size_b) {
  if (!isAmpere(params->mma_macro)) {
    return false;
  }
  // Use cp.async for loading operands if vec size is compatible
  const auto& validCpAsyncVecSize = [](int64_t dtype_size,
                                       int64_t vec_size) -> bool {
    int64_t cp_bytes = dtype_size * vec_size;
    return cp_bytes == 16 || cp_bytes == 8 || cp_bytes == 4;
  };
  return params->double_buffer_options.smem_double_buffer_stage > 1 &&
      validCpAsyncVecSize(dtype_size_a, params->supported_vec_size.a) &&
      validCpAsyncVecSize(dtype_size_b, params->supported_vec_size.b);
}

std::shared_ptr<MatmulParams> getMatmulHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FusionGuard fg(fusion);
  (void)data_cache;
  auto params = std::make_shared<MatmulParams>();

  // Set kernel index mode
  params->cparams.index_type = runtime_info.getIndexType();

  // Check initial conditions
  std::vector<mma_utils::MatmulPattern> patterns =
      mma_utils::findMatmulPatterns(fusion);
  NVF_ERROR(!patterns.empty(), "No matmul patterns were found");
  NVF_ERROR(
      patterns.size() == 1,
      "Only a single matmul pattern can currently be fused");
  mma_utils::MatmulPattern& pattern = patterns.front();

  // IdModel is used to analyze problem shape & layout
  IdModel id_model(fusion);

  const std::unordered_map<ValGroup, MatmulDomain> id_roles =
      pattern.getDimRoles(id_model);

  const auto problem_shape = getProblemShape(id_roles, runtime_info);

  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  const auto mma_op =
      getMmaOp(device_prop->major * 10 + device_prop->minor, problem_shape);
  NVF_ERROR(
      mma_op.has_value(), "Failed to determine a MMA op for given problem.");
  params->mma_macro = mma_op.value();

  const auto& roles_map_opt =
      mma_utils::getTensorsRoles(fusion, id_model, id_roles);
  NVF_ERROR(roles_map_opt.isValid(), "Tensor roles map in mma is not valid.");
  const auto roles_map = roles_map_opt.getData();

  params->supported_vec_size =
      getSupportedVectorization(roles_map, runtime_info);

  if (matmul_heuristic_plugin::hasPlugin()) {
    const mma_utils::MatmulProblemLayoutOpt layout_opt =
        mma_utils::getProblemLayout(id_model, id_roles, roles_map);
    NVF_ERROR(layout_opt.isValid(), layout_opt.getErrorMsg());
    const MmaLayout layout = layout_opt.getData();

    // Fill in proper values using plugin
    matmul_heuristic_plugin::updateMatmulParams(
        *params,
        problem_shape[(size_t)MatmulDomain::M],
        problem_shape[(size_t)MatmulDomain::N],
        problem_shape[(size_t)MatmulDomain::K],
        problem_shape[(size_t)MatmulDomain::Batch],
        layout,
        roles_map);
  } else {
    TORCH_WARN_ONCE(
        "Scheduling a matmul without heuristic plugin. "
        "Specify plugin location like this: "
        "NVFUSER_MATMUL_HEURISTIC_PLUGIN=/path/to/libmatmulheuristic.so");
    // Populate heuristic details
    auto status = initCoreHeuristics(params, problem_shape, roles_map);
    NVF_ERROR(status, "Initialization of core part of heuristics failed.");
  }

  // Disable magic zero for matmul kernels
  params->cparams.enable_magic_zero = false;

  // Set whether to use shared memory for epilogue
  std::tie(params->use_smem_epilogue, params->promote_prologue_smem_reuse) =
      mma_utils::generateSharedMemoryEpilogueHeuristics(
          params->tile_sizes,
          params->double_buffer_options.smem_double_buffer_stage,
          roles_map);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << params->toString() << std::endl;
  }

  return params;
}

} // namespace nvfuser
