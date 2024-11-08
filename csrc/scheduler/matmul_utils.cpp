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
#include <scheduler/runtime_info.h>

// NOTE: included to avoid compilation error caused by missing destructor in
// 'SchedulerRuntimeInfo'
#include <debug.h>
#include <id_model/id_model.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/utils.h>
#include <options.h>
#include <runtime/executor_utils.h>
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
namespace matmul_utils {
namespace {

//! Access to the structure should be done with labels defined in MatmulDimRole.
using ProblemShape = std::array<int64_t, 4>;

//! A helper for deciding the type of MMA op for given fusion and problem shape.
inline std::optional<MmaMacro> getMmaOp(
    const int dev_version,
    const ProblemShape& problem) {
  using MacroType = MmaMacro;

  // NOTE: A temp condition
  const ProblemShape::value_type n_extend = problem[(size_t)MatmulDimRole::N];
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

//! Find the number of circular buffer stages for shared memory operands, so
//! that the entire pipeline is filled given problem and heuristics.
void limitCircularBufferingSmemOperands(
    MatmulParams* mparams,
    const ProblemShape& problem_shape) {
  // Short-Circuit: Skip if matmul params do not use circular buffering
  if (!mparams->circular_buffer_options.circular_buffer_smem_write) {
    return;
  }

  // The axes of the mma tensorviews are permuted to [B, M, N, K],
  // so K / cta_tile_k is the circular buffer axis for both operands.
  int64_t numerator =
      ceilDiv(problem_shape[(size_t)MatmulDimRole::K], mparams->splitk_factor);
  int64_t k_stages = ceilDiv(numerator, mparams->tile_sizes.cta_tile.k);
  int64_t stages = std::min(
      k_stages,
      (int64_t)mparams->circular_buffer_options.smem_circular_buffer_stage);

  mparams->circular_buffer_options.circular_buffer_smem_write = (stages != 1);
  mparams->circular_buffer_options.smem_circular_buffer_stage = (int)stages;
}

//! A wrapper for core heuristics initialization.
//! We should have already set mparams->mma_macro before calling this function.
inline bool initCoreHeuristics(
    MatmulParams* mparams,
    const ProblemShape& problem_shape,
    const mma_utils::TensorRolesMap& tensor_roles) {
  const GemmTile instruction_tile = getMmaOpShape(mparams->mma_macro);
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

    const auto mn_ratio = (double)problem_shape[(size_t)MatmulDimRole::M] /
        (double)problem_shape[(size_t)MatmulDimRole::N];
    if (mn_ratio < 0.5) {
      m_ratio = 1;
      n_ratio = 4;
    } else if (mn_ratio > 2) {
      m_ratio = 4;
      n_ratio = 1;
    }

    cta_tile = {warp_tile.m * m_ratio, warp_tile.n * n_ratio, warp_tile.k};
  }

  mparams->tile_sizes = {cta_tile, warp_tile};

  // stages and async mem copy
  {
    // NOTE: compilation errors when async is enabled on Turing devices
    if (isAmpere(mparams->mma_macro)) {
      constexpr int stages = 3;

      mparams->circular_buffer_options.circular_buffer_smem_write = true;
      mparams->circular_buffer_options.circular_buffer_smem_read = true;
      mparams->circular_buffer_options.smem_circular_buffer_stage = stages;
    }
  }

  const auto roleMinDtypeSize =
      [&tensor_roles](MatmulTensorRole role) -> int64_t {
    const auto op_it = tensor_roles.find(role);
    NVF_ERROR(op_it != tensor_roles.end());
    int64_t min_size_bytes = 128LL;
    for (const TensorView* operand : op_it->second) {
      min_size_bytes = std::min(min_size_bytes, dataTypeSize(operand->dtype()));
    }
    return min_size_bytes;
  };
  mparams->async_gmem_load_operands = isCpAsyncOperandLoadSupported(
      mparams,
      std::min(
          roleMinDtypeSize(MatmulTensorRole::OPERAND_A),
          roleMinDtypeSize(MatmulTensorRole::OPERAND_B)));

  if (!mparams->async_gmem_load_operands) {
    // Circular buffering requires async load. If we cannot use async load due
    // to unsupported vectorization width, then we can only circular buffer at
    // most.
    mparams->circular_buffer_options.smem_circular_buffer_stage = std::min(
        2, mparams->circular_buffer_options.smem_circular_buffer_stage);
  }
  return true;
}

//! A helper for getting problem shape from fusion and runtime info.
//!
//! For a given domain, try to find the size by evaluating the extent of an
//! IterDomain in each group of that domain type. For example, if there are
//! multiple Batch dimensions, we find all ValGroups that are mapped as
//! MatmulDimRole::Batch, we evaluate the extent of each, then we multiply those
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
    const mma_utils::DimRolesMap& id_roles,
    const ValGraph& permissive_graph) {
  const auto& fusion_inputs = fusion->inputs();
  const auto& fusion_outputs = fusion->outputs();
  std::vector<TensorView*> mma_inputs = {pattern.A, pattern.B};
  const auto mma_output = pattern.output;

  const auto fusion_inputs_tvs =
      ir_utils::filterByType<TensorView>(fusion_inputs).vector();
  const auto fusion_outputs_tvs =
      ir_utils::filterByType<TensorView>(fusion_outputs).vector();

  constexpr size_t minimal_number_of_inputs = 2;

  // Quick checks
  {
    if (!mma_output->hasReduction()) {
      return "MMA output TV has no reduction domain";
    }

    // Check that there is a single K dimension
    if (std::count_if(
            mma_output->getLogicalDomain().begin(),
            mma_output->getLogicalDomain().end(),
            [](IterDomain* id) { return id->isReduction(); }) != 1) {
      return "MMA output TV must have exactly one reduction (K) dimension";
    }

    // Fusion should contain at least two inputs (for now)
    if (minimal_number_of_inputs > fusion_inputs.size()) {
      return "Fusion inputs contain at least one non-TensorView object";
    }
  }

  // Fusion topology check
  {
    // Track TensorViews with assigned roles so we can check that all inputs and
    // outputs have recognized roles
    std::set<TensorView*> tvs_with_roles;

    {
      for (MatmulTensorRole role :
           {MatmulTensorRole::OPERAND_A, MatmulTensorRole::OPERAND_B}) {
        auto entry = tensor_roles.find(role);
        if (entry != tensor_roles.end()) {
          if (1 == entry->second.size()) {
            tvs_with_roles.insert(entry->second.begin(), entry->second.end());
          } else {
            return "There is other than one fusion input that can be MMA operand";
          }
        } else {
          return "No candidate in fusion inputs for MMA operand";
        }
      }
    }

    auto entry = tensor_roles.find(MatmulTensorRole::OUTPUT);
    if (entry != tensor_roles.end()) {
      tvs_with_roles.insert(entry->second.begin(), entry->second.end());
    } else {
      return "No candidate in fusion outputs MMA output";
    }

    // Non-core input roles are optional, no requirements for definitions
    entry = tensor_roles.find(MatmulTensorRole::EPILOGUE_INPUT);
    if (entry != tensor_roles.end()) {
      tvs_with_roles.insert(entry->second.begin(), entry->second.end());
    }

    const auto in_out_tvs_count =
        fusion_inputs_tvs.size() + fusion_outputs_tvs.size();
    if (in_out_tvs_count != tvs_with_roles.size()) {
      return "Detected input/output TVs without assigned roles";
    }
  }

  // Check that canonical dim order is (B)MNK
  // TODO: Remove this check once we are confident that non-standard orders are
  // properly handled
  {
    std::vector<ValGroup> dim_ordering = mma_utils::canonicalDimOrdering(
        tensor_roles, id_roles, permissive_graph);
    VectorOfUniqueEntries<MatmulDimRole> role_order;
    for (const ValGroup& g : dim_ordering) {
      const auto it = id_roles.find(g);
      NVF_ERROR(it != id_roles.end());
      role_order.pushBack(it->second);
    }
    if (role_order.size() != 3 && role_order.size() != 4) {
      std::stringstream ss;
      ss << "Expected either {B,M,N,K} roles or {M,N,K} but role_order.size()="
         << role_order.size();
      return ss.str();
    }
    if (role_order.back() != MatmulDimRole::K) {
      return "Canonical dim order must be BMNK";
    }
    if (role_order.at(role_order.size() - 2) != MatmulDimRole::N) {
      return "Canonical dim order must be BMNK";
    }
    if (role_order.at(role_order.size() - 3) != MatmulDimRole::M) {
      return "Canonical dim order must be BMNK";
    }

    // Also check that dims within each role are consecutive with one another
    // for this pattern.
    // TODO: Lift this requirement by modifying the definition or setting
    // allocation domains to support this setting in MmaOp
    NVF_ERROR(pattern.output->definition() != nullptr);
    if (pattern.output->definition()->isA<MatmulOp>()) {
      if (TensorDomain::noReductions(
              TensorDomain::noDevices(pattern.B->getLogicalDomain()))
              .size() >
          TensorDomain::noReductions(
              TensorDomain::noDevices(pattern.A->getLogicalDomain()))
              .size()) {
        return "Implicit broadcast in MatmulOp causes new non-consecutive N dimension";
      }
    }
  }

  // TODO: Lift this requirement once we properly handle output allocation
  // domain
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
      const ValGraph& permissive_graph,
      SchedulerRuntimeInfo& runtime_info)
      : runtime_info_(runtime_info),
        tensor_roles_(tensor_roles),
        dim_roles_(dim_roles),
        permissive_graph_(permissive_graph),
        dim_ordering_(mma_utils::canonicalDimOrdering(
            tensor_roles,
            dim_roles_,
            permissive_graph_)) {}

  MatmulParams::SupportedVectorization compute() {
    const std::vector<int64_t> a_vecs =
        operandVectorizations(MatmulTensorRole::OPERAND_A);
    NVF_ERROR(a_vecs.size() == 1, "Expected exactly one A operand");
    const std::vector<int64_t> b_vecs =
        operandVectorizations(MatmulTensorRole::OPERAND_B);
    NVF_ERROR(b_vecs.size() == 1, "Expected exactly one B operand");
    return {a_vecs[0], b_vecs[0], epilogueVectorization()};
  }

 private:
  std::vector<int64_t> operandVectorizations(MatmulTensorRole role) {
    std::vector<int64_t> vec_sizes;
    const auto op_it = tensor_roles_.find(role);
    if (op_it != tensor_roles_.end()) {
      for (TensorView* tv : op_it->second) {
        vec_sizes.push_back(operandVectorization(tv));
      }
    }
    return vec_sizes;
  }

  MatmulDimRole dimRole(const ValGroup& g) const {
    auto dim_role_it = dim_roles_.find(g);
    NVF_ERROR(
        dim_role_it != dim_roles_.end(), "Found ValGroup with unknown role");
    return dim_role_it->second;
  }

  int64_t ptrAndDTypeVec(TensorView* tv) const {
    // TODO: ptrOf returns a fully aligned value of 16 for non-inputs.
    // However, we might be provided an output tensor. We should verify once
    // preallocated outputs are fully plumbed in that misaligned pointers are
    // respected in this calculation.
    const int64_t data_ptr_int = (int64_t)runtime_info_.ptrOf(tv);
    int64_t vec_size = scheduler_utils::maxVectorizationWidth(data_ptr_int);
    vec_size = std::min(vec_size, 16l);
    vec_size /= dataTypeSize(tv->dtype());
    vec_size = std::max(vec_size, 1l);
    return vec_size;
  }

  //! To analyze vectorization, we need to know pointer alignment, sizes, and
  //! strides. SchedulerRuntimeInfo contains all this info about fusion
  //! inputs, but fusion outputs are allocated by KernelExecutor so they are
  //! absent from SchedulerRuntimeInfo.
  //!
  //! This function just extracts sizes and strides from runtime_info_ when
  //! the argument is a fusion input. When the input is a fusion output, we
  //! respect the contiguity marked in the allocation domain. For
  //! discontiguous dimensions, we return a stride that has been padded to an
  //! odd value, which is the worst case scenario for vectorization.
  //!
  //! Note that this function is non-const because we use
  //! runtime_info_.expressionEvaluator() which caches intermediate values
  std::pair<std::vector<int64_t>, std::vector<int64_t>> getSizesAndStrides(
      TensorView* tv) {
    if (tv->isFusionInput()) {
      return {
          runtime_info_.getInputAllocationSizes(tv),
          runtime_info_.getInputAllocationStrides(tv)};
    }
    NVF_ERROR(
        tv->isFusionOutput(),
        "getSizesAndStrides should only be called with fusion inputs or outputs. Found ",
        tv->toString());
    // For outputs, compute sizes using ExpressionEvaluator, then compute
    // strides based on allocation domain, assuming contiguity as marked in
    // the TensorView. For discontiguous dimensions, we compute a stride that
    // is least favorable to vectorization, by padding to an odd value.
    std::vector<int64_t> sizes, strides;
    std::vector<bool> concrete_contig;
    for (size_t i : c10::irange(tv->getMaybeAllocationDomain().size())) {
      IterDomain* id = tv->getMaybeAllocationDomain().at(i);
      if (id->isBroadcast()) {
        sizes.push_back(1);
        continue;
      }
      if (id->isReduction()) {
        continue;
      }
      // Record contiguity of concrete dimensions
      std::optional<bool> contig_opt = tv->getContiguity().at(i);
      NVF_ERROR(contig_opt.has_value());
      concrete_contig.push_back(contig_opt.value());

      PolymorphicValue ext =
          runtime_info_.expressionEvaluator().evaluate(id->extent());
      NVF_ERROR(ext.hasValue());
      sizes.push_back(ext.as<int64_t>());
    }

    strides.resize(sizes.size(), 0l);
    int64_t stride = 1l;
    for (int64_t i = (int64_t)(sizes.size()) - 1l; i >= 0; --i) {
      strides[(size_t)i] = sizes[(size_t)i] == 1 ? 0 : stride;
      stride *= sizes[(size_t)i];
      if (!concrete_contig[(size_t)i]) {
        // pad non-concrete dims to next odd value
        stride |= 1l;
      }
    }
    return {sizes, strides};
  }

  // Given a TensorView and a vector of dimension ValGroups find
  // vectorization. The vector of dimensions indicates how the tensor will be
  // scheduled; dimensions in tv will be reordered if needed then the vector
  // of dimensions will be merged. We check the allocation domain of tv to
  // tell how the resulting merged TV can be vectorized. If the tensor does
  // not have any inner_dims, then it cannot be vectorized. In that case we
  // return 0 so that this tensor can be ignored in later computation.
  int64_t innerDimsVectorization(
      TensorView* tv,
      const std::vector<ValGroup>& inner_dims) {
    const auto& [sizes, strides] = getSizesAndStrides(tv);
    NVF_ERROR(sizes.size() == strides.size());

    // Position of the outermost vectorizable dimension, in allocation domain
    size_t inner_dim_pos = tv->getMaybeAllocationDomain().size();
    // Product of sizes of all vectorizable dims; i.e. the number of elements
    // in the merged vectorized dimension.
    int64_t inner_dims_numel = 1;
    std::vector<ValGroup> remaining_inner_dims(inner_dims);
    for (int64_t i = (int64_t)tv->getMaybeAllocationDomain().size() - 1; i >= 0;
         --i) {
      IterDomain* id = tv->getMaybeAllocationDomain()[i];
      if (id->isDeviceDim() || id->isReduction() || id->isBroadcast()) {
        continue;
      }

      ValGroup g = permissive_graph_.toGroup(id);
      // Exit when this does not match the given ordered inner dimension
      if (remaining_inner_dims.empty() || g != remaining_inner_dims.back()) {
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
            strides.at(i) == inner_dims_numel,
            "TensorView ",
            tv->toString(),
            " has marked contiguous inner dimension ",
            id->toString(),
            " but provided tensor has stride ",
            strides.at(i),
            " in that dimension.");
        inner_dim_pos = i;
        inner_dims_numel *= sizes.at(i);
      }
    }

    if (remaining_inner_dims.size() == inner_dims.size()) {
      // We didn't match any inner dims, so this tensor is not vectorizable.
      return 0l;
    }

    if (inner_dims_numel == 1l) {
      return 1l;
    }

    // Since this is unpredicated vectorization, the size of the innermost
    // dimension must be a multiple of the vectorization factor.
    int64_t vec_size = scheduler_utils::maxVectorizationWidth(inner_dims_numel);

    // Account for misaligned rows due to outer strides
    for (size_t i : c10::irange(inner_dim_pos)) {
      if (sizes.at(i) == 1) {
        // outer size-1 dimensions don't affect vectorizability
        continue;
      }
      vec_size = std::min(
          vec_size, scheduler_utils::maxVectorizationWidth(strides.at(i)));
    }

    return vec_size;
  }

  // Inspect the allocation domain of an operand input TensorView to determine
  // vectorization width.
  //
  // We canonicalize dimensions by reordering them with the given ordering
  // before merging all dimensions that have the same role. For a given
  // operand, this might mean that the inner-most dimension gets reordered to
  // be outer, even if it has the same role as the innermost dimension in the
  // canonical ordering.
  int64_t operandVectorization(TensorView* tv) {
    // Check data pointer alignment
    int64_t vec_size = ptrAndDTypeVec(tv);
    if (vec_size == 1l) {
      return vec_size;
    }

    // Find the inner-most non-batch role for this tensor, and collect all
    // ValGroups in that role, in the canonical ordering.
    std::optional<MatmulDimRole> vec_dim_role = std::nullopt;
    for (int64_t i = (int64_t)(tv->getMaybeAllocationDomain().size()) - 1;
         i >= 0;
         --i) {
      IterDomain* id = tv->getMaybeAllocationDomain()[i];
      if (id->isDeviceDim() || id->isReduction() || id->isBroadcast()) {
        continue;
      }

      ValGroup g = permissive_graph_.toGroup(id);
      MatmulDimRole dim_role = dimRole(g);
      if (dim_role == MatmulDimRole::Batch) {
        // We cannot vectorize in batch dimensions
        break;
      }
      if (!vec_dim_role.has_value()) {
        vec_dim_role = dim_role;
        break;
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
    std::optional<MatmulDimRole> inner_nonk_role = std::nullopt;
    for (auto g_it = dim_ordering_.rbegin(); g_it != dim_ordering_.rend();
         ++g_it) {
      const ValGroup& g = *g_it;

      MatmulDimRole dim_role = dimRole(g);
      if (dim_role == MatmulDimRole::K) {
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
        inner_nonk_role.value() == MatmulDimRole::Batch) {
      // If the innermost non-K dimension is a batch dimension, then we cannot
      // vectorize the outputs since we parallelize batch dimensions across
      // the grid.
      return 1l;
    }

    // Match the innermost dimensions above to contiguous innermost dims in tv
    // from inner to outer. Determine supported vectorization based on product
    // of matching sizes along with all outer strides.
    const auto innerMostVec = [&](TensorView* tv) {
      return std::min(
          ptrAndDTypeVec(tv), innerDimsVectorization(tv, inner_nonk_dims));
    };

    const auto d_it = tensor_roles_.find(MatmulTensorRole::OUTPUT);
    NVF_ERROR(
        d_it != tensor_roles_.end(), "Could not find any output D tensors");
    int64_t vec_size = 16l;
    for (TensorView* tv : d_it->second) {
      int64_t v = innerMostVec(tv);
      if (v == 0) {
        continue;
      }
      vec_size = std::min(vec_size, v);
    }
    if (const auto c_it = tensor_roles_.find(MatmulTensorRole::EPILOGUE_INPUT);
        c_it != tensor_roles_.end()) {
      for (TensorView* tv : c_it->second) {
        int64_t v = innerMostVec(tv);
        if (v == 0) {
          continue;
        }
        vec_size = std::min(vec_size, v);
      }
    }
    return vec_size;
  }

 private:
  SchedulerRuntimeInfo& runtime_info_;
  const mma_utils::TensorRolesMap& tensor_roles_;
  const mma_utils::DimRolesMap& dim_roles_;
  const ValGraph& permissive_graph_;
  std::vector<ValGroup> dim_ordering_;
};

MatmulParams::SupportedVectorization getSupportedVectorization(
    const mma_utils::TensorRolesMap& tensor_roles,
    const mma_utils::DimRolesMap& dim_roles,
    const ValGraph& permissive_graph,
    SchedulerRuntimeInfo& runtime_info) {
  VectorizationCalculator calc(
      tensor_roles, dim_roles, permissive_graph, runtime_info);
  return calc.compute();
}

} // anonymous namespace

std::unique_ptr<MatmulParams> getMatmulHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);
  (void)data_cache;
  auto mparams = std::make_unique<MatmulParams>();

  // Set kernel index mode
  mparams->cparams.index_type = runtime_info.getIndexType();

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
  id_model.maybeBuildGraph(IdMappingMode::PERMISSIVE);

  const mma_utils::DimRolesMap id_roles = pattern.getDimRoles(id_model);

  const auto problem_shape = getProblemShape(id_roles, runtime_info);

  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  const auto mma_op =
      getMmaOp(device_prop->major * 10 + device_prop->minor, problem_shape);
  NVF_ERROR(
      mma_op.has_value(), "Failed to determine a MMA op for given problem.");
  mparams->mma_macro = mma_op.value();

  const auto& tensor_roles_opt =
      mma_utils::getTensorRoles(fusion, id_model, id_roles);
  NVF_ERROR(
      tensor_roles_opt.isValid(), "Tensor roles map in mma is not valid.");
  const auto tensor_roles = tensor_roles_opt.getData();

  mparams->supported_vec_size = getSupportedVectorization(
      tensor_roles,
      id_roles,
      id_model.idGraph(IdMappingMode::PERMISSIVE),
      runtime_info);

  if (matmul_heuristic_plugin::hasPlugin()) {
    const mma_utils::MatmulOperandInnerDimsOpt inner_dims_opt =
        mma_utils::getOperandInnerDims(id_model, id_roles, tensor_roles);
    NVF_ERROR(inner_dims_opt.isValid(), inner_dims_opt.getErrorMsg());
    const mma_utils::MatmulOperandInnerDims inner_dims =
        inner_dims_opt.getData();

    // Fill in proper values using plugin
    matmul_heuristic_plugin::updateMatmulParams(
        mparams.get(),
        problem_shape[(size_t)MatmulDimRole::M],
        problem_shape[(size_t)MatmulDimRole::N],
        problem_shape[(size_t)MatmulDimRole::K],
        problem_shape[(size_t)MatmulDimRole::Batch],
        inner_dims,
        tensor_roles);
  } else {
    TORCH_WARN_ONCE(
        "Scheduling a matmul without heuristic plugin. "
        "Specify plugin location like this: "
        "NVFUSER_MATMUL_HEURISTIC_PLUGIN=/path/to/libmatmulheuristic.so");
    // Populate heuristic details
    auto status =
        initCoreHeuristics(mparams.get(), problem_shape, tensor_roles);
    NVF_ERROR(status, "Initialization of core part of heuristics failed.");
  }

  // Ensure that entire pipeline is filled for shared memory operands given
  // problem and heuristics.
  limitCircularBufferingSmemOperands(mparams.get(), problem_shape);

  // Disable magic zero for matmul kernels
  mparams->cparams.enable_magic_zero = false;

  // Set whether to use shared memory for epilogue
  std::tie(mparams->use_smem_epilogue, mparams->promote_prologue_smem_reuse) =
      mma_utils::generateSharedMemoryEpilogueHeuristics(
          mparams->tile_sizes,
          mparams->circular_buffer_options.smem_circular_buffer_stage,
          tensor_roles);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << mparams->toString() << std::endl;
  }

  return mparams;
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
  // 2. Check if fusion of MatmulOp and LinearOp is enabled, if applicable
  // 3. Check if inputs to the matmul pattern match any of
  // supported inputs layout
  // 4. Check if fusion represents expressions that are recognized by matmul
  // 5. Check if the input layout for the matmul pattern can be determined
  // scheduler.
  // 6. Check if the fusion is resharding.

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
  // Find matmul patterns
  std::vector<mma_utils::MatmulPattern> patterns =
      mma_utils::findMatmulPatterns(fusion);
  if (patterns.empty()) {
    return "No matmul patterns were found";
  }

  // #2
  {
    for (const mma_utils::MatmulPattern& pattern : patterns) {
      Expr* op = pattern.output->definition();
      if (op->isA<MatmulOp>() || op->isA<LinearOp>()) {
        if (!isOptionEnabled(EnableOption::FuseMatmul)) {
          // Check for MatmulOp or LinearOp. If found, then only fuse if option
          // is specified
          return "MatmulOp and LinearOp fusion is disabled by default. "
                 "Enable it using NVFUSER_ENABLE=fuse_matmul";
        }
        // Refuse patterns containing 1D inputs since these are mat-vec as
        // opposed to mat-mat products.
        if (pattern.A->nDims() < 2 || pattern.B->nDims() < 2) {
          return "Cannot fuse matrix-vector products";
        }
        for (TensorView* operand : {pattern.A, pattern.B}) {
          if (operand->dtype() != DataType::Half &&
              operand->dtype() != DataType::BFloat16) {
            return "Unsupported operand type. Operands must be fp16 or bf16";
          }
        }
      }
    }
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
  {
    auto support_status = isMatmulFusionDefinitionSupported(
        fusion,
        patterns.front(),
        tensor_roles,
        id_roles,
        id_model.idGraph(IdMappingMode::PERMISSIVE));
    if (!support_status.empty()) {
      return support_status;
    }
  }

  // #5
  const auto input_layout_opt =
      mma_utils::getOperandInnerDims(id_model, id_roles, tensor_roles);
  if (!input_layout_opt.isValid()) {
    return input_layout_opt.getErrorMsg();
  }

  // #6
  if (scheduler_utils::isResharding(fusion)) {
    return "Fusion is resharding.";
  }

  return "";
}

std::string getMatmulRunTimeRejectReason(
    Fusion* fusion,
    HeuristicDataCache* data_cache,
    SchedulerRuntimeInfo& runtime_info) {
  // TODO: add proper set of checks
  return "";
}

bool isCpAsyncOperandLoadSupported(
    const MatmulParams* mparams,
    int64_t min_dtype_size) {
  if (!isAmpere(mparams->mma_macro)) {
    return false;
  }
  // Use cp.async for loading operands if vec size is compatible
  const auto& validCpAsyncVecSize = [](int64_t dtype_size,
                                       int64_t vec_size) -> bool {
    int64_t cp_bytes = dtype_size * vec_size;
    return cp_bytes == 16 || cp_bytes == 8 || cp_bytes == 4;
  };
  // TODO: We should compute validCpAsyncVecSize for all the operand
  // dtype/vec_size pairs and AND them together
  return mparams->circular_buffer_options.smem_circular_buffer_stage > 1 &&
      validCpAsyncVecSize(
             min_dtype_size,
             std::min(
                 mparams->supported_vec_size.a, mparams->supported_vec_size.b));
}

void moveInnerBroadcastLeft(TensorView* tv, int64_t number_of_inner_pos) {
  NVF_ERROR(tv->nDims() >= number_of_inner_pos);
  std::vector<int64_t> broadcast_pos;
  std::vector<int64_t> nonbroadcast_pos;

  for (auto i : c10::irange(number_of_inner_pos)) {
    auto axis_idx = i - number_of_inner_pos;
    auto id = tv->axis(axis_idx);
    if (id->isBroadcast()) {
      broadcast_pos.push_back(axis_idx);
    } else {
      nonbroadcast_pos.push_back(axis_idx);
    }
  }

  auto combined_pos_vec = broadcast_pos;
  combined_pos_vec.insert(
      combined_pos_vec.end(), nonbroadcast_pos.begin(), nonbroadcast_pos.end());

  std::unordered_map<int64_t, int64_t> order_map;
  for (auto i : c10::irange(number_of_inner_pos)) {
    order_map[combined_pos_vec.at(i)] = i - number_of_inner_pos;
  }

  // Apply ordering.
  tv->reorder(order_map);
}
} // namespace matmul_utils
} // namespace nvfuser
