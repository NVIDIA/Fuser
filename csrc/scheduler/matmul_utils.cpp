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
    const mma_utils::TensorRolesMap& tensor_roles) {
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

  const auto& roleMinDtypeSize = [&tensor_roles](MatmulRole role) -> int64_t {
    const auto op_it = tensor_roles.find(role);
    NVF_ERROR(op_it != tensor_roles.end());
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
    const mma_utils::DimRolesMap& dim_roles,
    SchedulerRuntimeInfo& runtime_info) {
  ProblemShape shape{1, 1, 1, 1};
  for (const auto& [g, dom] : dim_roles) {
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

// Checks that this pattern:
//   - is a GEMM or batch GEMM
//   - has at least two inputs i.e. not A @ A.T
//   - has a single A and a single B operand i.e not A @ (B1 * B2)
//   - has a fusion output with OUTPUT_D role i.e. that has M, N dims
//   - includes all fusion inputs/outputs in its tensor roles
//   - has no fusion inputs with non-trivial allocation domain
std::string isMatmulFusionDefinitionSupported(
    Fusion* fusion,
    const mma_utils::MatmulPattern& pattern,
    const mma_utils::TensorRolesMap& tensor_roles,
    const mma_utils::DimRolesMap& id_roles) {
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
    constexpr int64_t expected_axes_numbers = 1;
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
    auto entry = tensor_roles.find(MatmulRole::INPUT_A);
    std::set<TensorView*> tvs_with_roles;

    if (entry != tensor_roles.end()) {
      if (MATMUL_CORE_ROLES_EXPECTED_COUNT == entry->second.size()) {
        tvs_with_roles.insert(entry->second.begin(), entry->second.end());
      } else {
        return "There is more than a single fusion input that can be MMA first input";
      }
    } else {
      return "No candidate in fusion inputs for MMA first input";
    }

    entry = tensor_roles.find(MatmulRole::INPUT_B);
    if (entry != tensor_roles.end()) {
      if (MATMUL_CORE_ROLES_EXPECTED_COUNT == entry->second.size()) {
        tvs_with_roles.insert(entry->second.begin(), entry->second.end());
      } else {
        return "There is more than a single fusion input that can be MMA second input";
      }
    } else {
      return "No candidate in fusion inputs for MMA second input";
    }

    entry = tensor_roles.find(MatmulRole::OUTPUT_D);
    if (entry != tensor_roles.end()) {
      tvs_with_roles.insert(entry->second.begin(), entry->second.end());
    } else {
      return "No candidate in fusion outputs MMA output";
    }

    // Non-core input roles are optional, no requirements for definitions
    entry = tensor_roles.find(MatmulRole::INPUT_C);
    if (entry != tensor_roles.end()) {
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

class VectorizationCalculator {
 public:
  VectorizationCalculator(
      const mma_utils::TensorRolesMap& tensor_roles,
      const mma_utils::DimRolesMap& dim_roles,
      const ValGraph& exact_graph,
      SchedulerRuntimeInfo& runtime_info)
      : runtime_info_(runtime_info),
        tensor_roles_(tensor_roles),
        dim_roles_(dim_roles),
        exact_graph_(exact_graph) {
    dim_ordering_ =
        mma_utils::canonicalDimOrdering(tensor_roles, dim_roles_, exact_graph_);
  }

  MatmulParams::SupportedVectorization compute() {
    return {operandVectorizations(), epilogueVectorization()};
  }

 private:
  std::vector<int64_t> operandVectorizations() {
    std::vector<int64_t> vec_sizes;
    for (MatmulRole role : {MatmulRole::INPUT_A, MatmulRole::INPUT_B}) {
      const auto op_it = tensor_roles_.find(role);
      if (op_it == tensor_roles_.end()) {
        continue;
      }
      for (TensorView* tv : op_it->second) {
        vec_sizes.push_back(operandVectorization(tv));
      }
    }
    return vec_sizes;
  }

  MatmulDomain dimRole(const ValGroup& g) const {
    auto dim_role_it = dim_roles_.find(g);
    NVF_ERROR(
        dim_role_it != dim_roles_.end(), "Found ValGroup with unknown role");
    return dim_role_it->second;
  }

  int64_t ptrAndDTypeVec(TensorView* tv) const {
    const int64_t data_ptr_int = (int64_t)runtime_info_.ptrOf(tv);
    int64_t vec_size = scheduler_utils::maxVectorizationWidth(data_ptr_int);
    vec_size = std::min(vec_size, 16l);
    vec_size /= dataTypeSize(tv->dtype());
    vec_size = std::max(vec_size, 1l);
    return vec_size;
  }

  // Note this is non-const because we use runtime_info_.expressionEvaluator()
  std::pair<std::vector<int64_t>, std::vector<int64_t>> getSizesAndStrides(
      TensorView* tv) {
    if (tv->isFusionInput()) {
      return {
          runtime_info_.getInputAllocationSizes(tv),
          runtime_info_.getInputAllocationStrides(tv)};
    }
    // For non-inputs, compute sizes using ExpressionEvaluator, then compute
    // strides based on allocation domain, assuming full contiguity regardless
    // of how it is marked in the TensorView.
    std::vector<int64_t> sizes, strides;
    for (IterDomain* id : tv->getMaybeAllocationDomain()) {
      if (id->isBroadcast() || id->isReduction()) {
        continue;
      }
      PolymorphicValue ext =
          runtime_info_.expressionEvaluator().evaluate(id->extent());
      NVF_ERROR(ext.hasValue());
      sizes.push_back(ext.as<int64_t>());
    }

    strides.resize(sizes.size(), 0l);
    int64_t stride = 1l;
    for (int64_t i = (int64_t)(sizes.size()) - 1l; i >= 0; --i) {
      strides[(size_t)i] = stride;
      stride *= sizes[(size_t)i];
    }
    return {sizes, strides};
  }

  // Given a TensorView and a vector of dimension ValGroups find vectorization.
  // The vector of dimensions indicates how the tensor will be scheduled;
  // dimensions in tv will be reordered if needed then the vector of dimensions
  // will be merged. We check the allocation domain of tv to tell how the
  // resulting merged TV can be vectorized.
  int64_t innerDimsVectorization(
      TensorView* tv,
      const std::vector<ValGroup>& inner_dims) {
    const auto& [sizes, strides] = getSizesAndStrides(tv);
    NVF_ERROR(sizes.size() == strides.size());

    // Position of the outermost vectorizable dimension, in allocation domain
    size_t inner_dim_pos = tv->getMaybeAllocationDomain().size();
    // Product of sizes of all vectorizable dims; i.e. the size of the merged
    // vectorized dimension.
    int64_t inner_dims_size = 1;
    std::vector<ValGroup> remaining_inner_dims(inner_dims);
    for (size_t i = tv->getMaybeAllocationDomain().size() - 1; i >= 0; --i) {
      IterDomain* id = tv->getMaybeAllocationDomain()[i];
      if (id->isReduction() || id->isBroadcast()) {
        continue;
      }

      ValGroup g = exact_graph_.toGroup(id);
      // Exit when this does not match the given ordered inner dimension
      if (g != remaining_inner_dims.back()) {
        break;
      }
      remaining_inner_dims.pop_back();

      std::optional<bool> c = tv->getContiguity().at(i);
      NVF_ERROR(c.has_value());
      if (!c.value()) {
        // axis is marked discontiguous; can't vectorize
        break;
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
        inner_dim_pos = i;
        inner_dims_size *= sizes[i];
      }
    }

    if (inner_dims_size == 1l) {
      return 1l;
    }

    // Since this is unpredicated vectorization, the size of the innermost
    // dimension must be a multiple of the vectorization factor.
    int64_t vec_size = scheduler_utils::maxVectorizationWidth(inner_dims_size);

    // Account for misaligned rows due to outer strides
    for (size_t i : c10::irange(inner_dim_pos)) {
      if (sizes[i] == 1) {
        // outer size-1 dimensions don't affect vectorizability
        continue;
      }
      vec_size = std::min(
          vec_size, scheduler_utils::maxVectorizationWidth(strides[i]));
    }

    return vec_size;
  }

  // Inspect the allocation domain of an operand input TensorView to determine
  // vectorization width.
  //
  // We canonicalize dimensions by reordering them with the given ordering
  // before merging all dimensions that have the same role. For a given operand,
  // this might mean that the inner-most dimension gets reordered to be outer,
  // even if it has the same role as the innermost dimension in the canonical
  // ordering.
  int64_t operandVectorization(TensorView* tv) {
    // Check data pointer alignment
    int64_t vec_size = ptrAndDTypeVec(tv);
    if (vec_size == 1l) {
      return vec_size;
    }

    // Find the inner-most non-batch role for this tensor, and collect all
    // ValGroups in that role, in the canonical ordering.
    std::optional<MatmulDomain> vec_dim_role = std::nullopt;
    for (int64_t i = (int64_t)(tv->getMaybeAllocationDomain().size()) - 1;
         i >= 0;
         --i) {
      IterDomain* id = tv->getMaybeAllocationDomain()[i];
      if (id->isReduction() || id->isBroadcast()) {
        continue;
      }

      ValGroup g = exact_graph_.toGroup(id);
      MatmulDomain dim_role = dimRole(g);
      if (dim_role == MatmulDomain::Batch) {
        // We cannot vectorize in batch dimensions
        break;
      }
      if (!vec_dim_role.has_value()) {
        vec_dim_role = dim_role;
      }
    }
    if (!vec_dim_role.has_value()) {
      // Didn't find any dimensions to vectorize
      return 1l;
    }

    // Extract dims with this role in the canonical ordering
    std::vector<ValGroup> ordered_inner_dims;
    for (const ValGroup& og : dim_ordering_) {
      if (dimRole(og) == vec_dim_role.value()) {
        ordered_inner_dims.push_back(og);
      }
    }

    return std::min(vec_size, innerDimsVectorization(tv, ordered_inner_dims));
  }

  int64_t epilogueVectorization() {
    // This is a vector of non-K dimensions sorted from inner to outer
    std::vector<ValGroup> inner_nonk_dims;
    std::optional<MatmulDomain> inner_nonk_role = std::nullopt;
    for (auto g_it = dim_ordering_.rbegin(); g_it != dim_ordering_.rend();
         ++g_it) {
      const ValGroup& g = *g_it;

      MatmulDomain dim_role = dimRole(g);
      if (dim_role == MatmulDomain::K) {
        // Skip K dims since they won't appear in epilogue loop nest
        continue;
      }
      if (!inner_nonk_role.has_value()) {
        inner_nonk_role = dim_role;
      }
      if (dim_role != inner_nonk_role.value()) {
        break;
      }
      inner_nonk_dims.push_back(g);
    }

    if (!inner_nonk_role.has_value() ||
        inner_nonk_role.value() == MatmulDomain::Batch) {
      // If the innermost non-K dimension is a batch dimension, then we cannot
      // vectorize the outputs since we parallelize batch dimensions across the
      // grid.
      return 1l;
    }

    // Match the innermost dimensions above to contiguous innermost dims in tv
    // from inner to outer. Determine supported vectorization based on product
    // of matching sizes along with all outer strides.
    const auto innerMostVec = [&](TensorView* tv) {
      int64_t vec_size = ptrAndDTypeVec(tv);
      if (vec_size == 1l) {
        return vec_size;
      }
      return std::min(vec_size, innerDimsVectorization(tv, inner_nonk_dims));
    };

    const auto d_it = tensor_roles_.find(MatmulRole::OUTPUT_D);
    NVF_ERROR(
        d_it != tensor_roles_.end(), "Could not find any output D tensors");
    int64_t vec_size = 16l;
    for (TensorView* tv : d_it->second) {
      vec_size = std::min(vec_size, innerMostVec(tv));
    }
    if (const auto c_it = tensor_roles_.find(MatmulRole::INPUT_C);
        c_it != tensor_roles_.end()) {
      for (TensorView* tv : c_it->second) {
        vec_size = std::min(vec_size, innerMostVec(tv));
      }
    }
    return vec_size;
  }

 private:
  SchedulerRuntimeInfo& runtime_info_;
  const mma_utils::TensorRolesMap& tensor_roles_;
  const mma_utils::DimRolesMap& dim_roles_;
  const ValGraph& exact_graph_;
  std::vector<ValGroup> dim_ordering_;
};

MatmulParams::SupportedVectorization getSupportedVectorization(
    const mma_utils::TensorRolesMap& tensor_roles,
    const mma_utils::DimRolesMap& dim_roles,
    const ValGraph& exact_graph,
    SchedulerRuntimeInfo& runtime_info) {
  VectorizationCalculator calc(
      tensor_roles, dim_roles, exact_graph, runtime_info);
  return calc.compute();
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
  // 0. Check if the current CUDA device is supported
  // 1. Check if there is exactly one matmul pattern defined in the fusion.
  // 2. Check if the input layout for the matmul pattern can be determined
  // 3. Check if fusion represents expressions that are recognized by matmul
  // scheduler.

  // #0
  {
    const auto device_prop = at::cuda::getCurrentDeviceProperties();
    // Use a dummy problem shape to determine whether this is a supported
    // device.
    const auto mma_op = getMmaOp(
        device_prop->major * 10 + device_prop->minor, {128, 128, 128, 1});
    if (!mma_op.has_value()) {
      return "Unsupported device compute capability";
    }
  }

  // #1
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

  // #3
  // Prepare an IdModel which will be reused to check remaining conditions
  IdModel id_model(fusion);
  const auto id_roles = patterns.front().getDimRoles(id_model);
  const mma_utils::TensorRolesMapOpt tensor_roles_opt =
      mma_utils::getTensorRoles(fusion, id_model, id_roles);
  if (!tensor_roles_opt.isValid()) {
    return {tensor_roles_opt.getErrorMsg()};
  }
  mma_utils::TensorRolesMap tensor_roles = tensor_roles_opt.getData();

  // #4
  const auto input_layout_opt =
      mma_utils::getProblemLayout(id_model, id_roles, tensor_roles);
  if (!input_layout_opt.isValid()) {
    return input_layout_opt.getErrorMsg();
  }

  // #5
  {
    auto support_status = isMatmulFusionDefinitionSupported(
        fusion, patterns.front(), tensor_roles, id_roles);
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
  NVF_ERROR(
      params->supported_vec_size.operands.size() == 2,
      "Only two operands supported");
  return params->double_buffer_options.smem_double_buffer_stage > 1 &&
      validCpAsyncVecSize(
             dtype_size_a, params->supported_vec_size.operands[0]) &&
      validCpAsyncVecSize(dtype_size_b, params->supported_vec_size.operands[1]);
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
  id_model.maybeBuildGraph(IdMappingMode::EXACT);

  const mma_utils::DimRolesMap id_roles = pattern.getDimRoles(id_model);

  const auto problem_shape = getProblemShape(id_roles, runtime_info);

  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  const auto mma_op =
      getMmaOp(device_prop->major * 10 + device_prop->minor, problem_shape);
  NVF_ERROR(
      mma_op.has_value(), "Failed to determine a MMA op for given problem.");
  params->mma_macro = mma_op.value();

  const auto& tensor_roles_opt =
      mma_utils::getTensorRoles(fusion, id_model, id_roles);
  NVF_ERROR(
      tensor_roles_opt.isValid(), "Tensor roles map in mma is not valid.");
  const auto tensor_roles = tensor_roles_opt.getData();

  params->supported_vec_size = getSupportedVectorization(
      tensor_roles,
      id_roles,
      id_model.idGraph(IdMappingMode::EXACT),
      runtime_info);

  if (matmul_heuristic_plugin::hasPlugin()) {
    const mma_utils::MatmulProblemLayoutOpt layout_opt =
        mma_utils::getProblemLayout(id_model, id_roles, tensor_roles);
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
        tensor_roles);
  } else {
    TORCH_WARN_ONCE(
        "Scheduling a matmul without heuristic plugin. "
        "Specify plugin location like this: "
        "NVFUSER_MATMUL_HEURISTIC_PLUGIN=/path/to/libmatmulheuristic.so");
    // Populate heuristic details
    auto status = initCoreHeuristics(params, problem_shape, tensor_roles);
    NVF_ERROR(status, "Initialization of core part of heuristics failed.");
  }

  // Disable magic zero for matmul kernels
  params->cparams.enable_magic_zero = false;

  // Set whether to use shared memory for epilogue
  std::tie(params->use_smem_epilogue, params->promote_prologue_smem_reuse) =
      mma_utils::generateSharedMemoryEpilogueHeuristics(
          params->tile_sizes,
          params->double_buffer_options.smem_double_buffer_stage,
          tensor_roles);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << params->toString() << std::endl;
  }

  return params;
}

} // namespace nvfuser
