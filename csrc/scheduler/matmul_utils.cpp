// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/matmul_heuristic.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/registry.h>

// NOTE: included to avoid compilation error caused by missing destructor in
// 'SchedulerRuntimeInfo'
#include <executor_utils.h>
#include <algorithm>
#include <deque>
#include <iostream>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include "ATen/cuda/CUDAContext.h"
#include "c10/util/Optional.h"
#include "ir_base_nodes.h"
#include "ir_interface_nodes.h"
#include "ir_internal_nodes.h"
#include "ir_utils.h"
#include "mma_type.h"
#include "type.h"
#include "utils.h"

namespace nvfuser {
namespace {

using MatmulLayout = MmaOptions::MmaInputLayout;
using LayoutData =
    std::pair<c10::optional<MatmulLayout>, c10::optional<std::string>>;
using TensorShape = std::vector<int64_t>;
using ProblemShape = TensorShape;

//! A constant with position of M value (a number of columns in A tensor for TT
//!  layout) in problem in ProblemShape type.
constexpr size_t M_POS = 0;
//! A constant with position of N value (a number of rows in B tensor for TT
//!  layout) in problem in ProblemShape type.
constexpr size_t N_POS = 1;
//! A constant with position of K value (a number of rows in A tensor for TT
//!  layout) in problem in ProblemShape type.
constexpr size_t K_POS = 2;
//! A constant with expected number of dimensions in ProblemShape type.
constexpr size_t PROBLEM_DIMS = 3;

// TODO: helpers to be moved to 'iter_visitor.h'
std::deque<std::deque<Val*>> getAllDepndencyChains(
    const std::vector<Val*>& producers,
    const std::vector<Val*>& consumers) {
  std::deque<std::deque<Val*>> all_paths;
  for (auto* consumer : consumers) {
    for (auto* producer : producers) {
      auto paths = DependencyCheck::getAllDependencyChains(producer, consumer);
      if (paths.empty()) {
        continue;
      }
      all_paths.insert(
          all_paths.end(),
          std::make_move_iterator(paths.begin()),
          std::make_move_iterator(paths.end()));
    }
  }

  return all_paths;
}

//! A wrapper for printing debug details.
void printMsg(const std::string& msg) {
  std::cout << msg << std::endl;
}

//! A helper for deciding what kernel indexing mode use (int32_t or int64_t).
//!  TODO: add strides to handle non-continous tensors
PrimDataType getIndexType(const ProblemShape& problem_shape) {
  // based on collectIndexMode function
  constexpr int64_t most_positive_int32_index =
      std::numeric_limits<int>::max() / 2;

  const auto m = static_cast<int64_t>(problem_shape[M_POS]);
  const auto n = static_cast<int64_t>(problem_shape[N_POS]);
  const auto k = static_cast<int64_t>(problem_shape[K_POS]);

  const bool use_i64_index = m * k > most_positive_int32_index || // tensor A
      k * n > most_positive_int32_index || // tensor B
      m * n > most_positive_int32_index; // output tensor

  return use_i64_index ? PrimDataType::Int : PrimDataType::Int32;
}

//! A helper for deciding the type of MMA op for given fusion and problem shape.
inline c10::optional<MmaOptions::MacroType> getMmaOp(
    const int dev_version,
    const ProblemShape& problem) {
  using MacroType = MmaOptions::MacroType;

  TORCH_INTERNAL_ASSERT(
      problem.size() == PROBLEM_DIMS,
      "Invalid size of problem shape (number of dimensions)");

  // NOTE: A temp condition
  const bool use_small_n =
      ((problem[N_POS] % 8) == 0) && ((problem[N_POS] % 16) != 0);

  switch (dev_version) {
    case 70:
      return MacroType::Volta_16_16_4;
    case 75:
      return (use_small_n) ? MacroType::Turing_16_8_16
                           : MacroType::Turing_16_16_16;
    case 80:
      return (use_small_n) ? MacroType::Ampere_16_8_16
                           : MacroType::Ampere_16_16_16;
    default:
      break;
  }
  return c10::nullopt;
}

//! A helper for checking if layout of MMA op's inputs. It will return optional
//! message if check fails.
LayoutData getInputsLayout(const MmaOp* mma_expr) {
  std::stringstream ss;
  const auto& mmaExprInputs = mma_expr->inputs();

  const auto* in_A = mmaExprInputs[0]->as<TensorView>();
  const auto* in_B = mmaExprInputs[1]->as<TensorView>();

  // The number of IterDomains of MMA inputs must be the same
  if (in_A->nDims() != in_B->nDims()) {
    ss << "Mma op inputs don't have the same number of IterDomains, 1st input("
       << std::to_string(in_A->nDims()) << "), 2nd input("
       << std::to_string(in_B->nDims()) + ")";
    return {c10::nullopt, ss.str()};
  }

  // The currently supported number of IterDomains per MMA op input is 3
  constexpr size_t supportedDims = 3;
  if (in_A->nDims() != supportedDims) {
    ss << "Mma op inputs have unsupported number of IterDomains, got: "
       << std::to_string(in_A->nDims()) << ", expected "
       << std::to_string(supportedDims);
    return {c10::nullopt, ss.str()};
  }

  using AxisPos = decltype(std::declval<TensorView>().nDims());
  constexpr AxisPos unInitPos = -1;
  AxisPos bcastInApos = unInitPos;
  AxisPos bcastInBpos = unInitPos;

  // The first and the second input of MMA have the same number of
  // IterDomains
  for (AxisPos pos = 0; pos < in_A->nDims(); ++pos) {
    if (in_A->axis(static_cast<int>(pos))->isBroadcast()) {
      if (bcastInApos != unInitPos) {
        ss << "Mma op first input has more than one broadcast IterDomain: "
           << std::to_string(bcastInApos) << " and " << std::to_string(pos);
        return {c10::nullopt, ss.str()};
      }
      bcastInApos = pos;
    }
    if (in_B->axis(static_cast<int>(pos))->isBroadcast()) {
      if (bcastInBpos != unInitPos) {
        ss << "Mma op second input has more than one broadcast IterDomain: "
           << std::to_string(bcastInBpos) << " and " << std::to_string(pos);
        return {c10::nullopt, ss.str()};
      }
      bcastInBpos = pos;
    }
  }

  // MMA inputs need to have broadcast IterDomains
  if (bcastInApos == unInitPos || bcastInBpos == unInitPos) {
    ss << "The " << (bcastInApos == unInitPos ? "first" : "second")
       << " mma op has no broadcast IterDomain";
    return {c10::nullopt, ss.str()};
  }

  // MMA inputs must have supported data layout, defined in MatmulLayout
  // MatmulLayout::TT
  if (bcastInApos == static_cast<size_t>(2) &&
      bcastInBpos == static_cast<size_t>(0)) {
    return {MatmulLayout::TT, c10::nullopt};
  }
  // MatmulLayout::TN
  if (bcastInApos == static_cast<size_t>(1) &&
      bcastInBpos == static_cast<size_t>(0)) {
    return {MatmulLayout::TN, c10::nullopt};
  }
  // MatmulLayout::NT
  if (bcastInApos == static_cast<size_t>(2) &&
      bcastInBpos == static_cast<size_t>(1)) {
    return {MatmulLayout::NT, c10::nullopt};
  }

  ss << "Unsupported layout, broadcasts: inputA(" << bcastInApos << "), inputB("
     << bcastInBpos << ")";
  return {c10::nullopt, ss.str()};
}

//! A wrapper for core heuristics initialization
inline bool initCoreHeuristics(
    std::shared_ptr<MatmulParams> params,
    const MmaOptions::MacroType& mma_op,
    const MatmulLayout& layout,
    const ProblemShape& problem_shape) {
  const GemmTile instruction_tile = getMmaOpShape(mma_op);
  GemmTile warp_tile = {-1, -1, -1};
  GemmTile cta_tile = {-1, -1, -1};

  using DimType = decltype(GemmTile::m);

  // warp tile shape
  {
    if (isAmpere(mma_op)) {
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
    } else {
      // No support for Volta and Turing
      return false;
    }
  }

  // cta tile shape
  {
    // Initial target:
    // - 4 warp tiles per CTA
    // - CTA k-dim should be same as warp tile k-dim

    DimType m_ratio = 2;
    DimType n_ratio = 2;

    const auto mn_ratio =
        (double)problem_shape[M_POS] / (double)problem_shape[N_POS];
    if (mn_ratio < 0.5) {
      m_ratio = 1;
      n_ratio = 4;
    } else if (mn_ratio > 2) {
      m_ratio = 4;
      n_ratio = 1;
    }

    cta_tile = {warp_tile.m * m_ratio, warp_tile.n * n_ratio, warp_tile.k};
  }

  params->mma_op = mma_op;
  params->layout = layout;
  params->tile_sizes = {cta_tile, warp_tile, instruction_tile};

  return true;
}

//! A wrapper for additional heuristics initialization
inline bool initExtraHeuristics(
    std::shared_ptr<MatmulParams> params,
    const ProblemShape& problem_shape) {
  // TODO: add logic to calculate efficient number of stages
  constexpr int stages = 3;

  params->async_gmem_load_operands = true;
  params->double_buffer_options.double_buffer_smem_write = true;
  params->double_buffer_options.double_buffer_smem_read = true;
  params->double_buffer_options.smem_double_buffer_stage = stages;

  return true;
}

//! A helper for getting problem shape from fusion and runtime info. Operation
//! can fail and nullopt object is returned.
c10::optional<ProblemShape> getProblemShape(
    Fusion* fusion,
    const MmaOp* mma_expr,
    SchedulerRuntimeInfo& runtime_info,
    const MatmulLayout matmul_layout) {
  const auto& fusion_inputs = fusion->inputs();
  const auto& fusion_outputs = fusion->outputs();
  const auto& mma_inputs = mma_expr->inputs();
  const auto& mma_outputs = mma_expr->outputs();

  // It is an unsupported fusion if
  // - there are more than one fusion input TensorViews (producers)
  //   for MMA op input
  // - there are more than one fusion output TensorViews (consumers)
  //   MMA op output
  const auto getKeyTvFromPathBetween =
      [](const std::vector<Val*>& producers,
         const std::vector<Val*>& consumers) -> Val* {
    const auto paths = getAllDepndencyChains(producers, consumers);

    if (paths.empty()) {
      return nullptr;
    }

    std::vector<Val*> tvs;
    for (const auto& path : paths) {
      if (path.empty()) {
        continue;
      }
      if (path.front()->isA<TensorView>()) {
        tvs.push_back(path.front());
      }
    }
    return (tvs.size() == 1) ? tvs[0] : nullptr;
  };

  const auto* tv_input_A =
      getKeyTvFromPathBetween(fusion_inputs, {mma_inputs[0]});
  if (nullptr == tv_input_A) {
    return c10::nullopt;
  }

  const auto* tv_input_B =
      getKeyTvFromPathBetween(fusion_inputs, {mma_inputs[1]});
  if (nullptr == tv_input_B) {
    return c10::nullopt;
  }

  const auto* tv_output =
      getKeyTvFromPathBetween({mma_outputs[0]}, fusion_outputs);
  if (nullptr == tv_output) {
    return c10::nullopt;
  }

  // A helper for populating concrete domains from TensorView
  const auto getShape = [&runtime_info](const TensorView* tv) {
    TensorShape tv_shape;
    const auto concrete_domains = TensorDomain::noReductions(
        TensorDomain::noBroadcasts(tv->as<TensorView>()->domain()->domain()));
    for (const auto* domain : concrete_domains) {
      const auto domain_extend =
          runtime_info.expressionEvaluator().evaluate(domain->extent());
      if (domain_extend) {
        tv_shape.push_back(domain_extend->as<int64_t>());
      }
    }
    return tv_shape;
  };

  const auto& in_A = getShape(tv_input_A->as<TensorView>());
  const auto& in_B = getShape(tv_input_B->as<TensorView>());
  const auto& output = getShape(tv_output->as<TensorView>());

  constexpr size_t expected_dims = 2;
  if (in_A.size() != expected_dims || //
      in_B.size() != expected_dims || //
      output.size() != expected_dims) {
    return c10::nullopt;
  }

  switch (matmul_layout) {
    case MatmulLayout::TT: {
      // in_A := [M, K]
      // in_B := [K, N]
      // output := [M, N]
      const bool check_k = in_A[1] == in_B[0];
      const bool check_m = in_A[0] == output[0];
      const bool check_n = in_B[1] == output[1];
      if (!(check_k && check_m && check_n)) {
        return c10::nullopt;
      }
      // [M, N, K]
      return TensorShape{output[0], output[1], in_A[1]};
    }
    case MatmulLayout::NT: {
      // in_A := [K, M]
      // in_B := [K, N]
      // output := [M, N]
      const bool check_k = in_A[0] == in_B[0];
      const bool check_m = in_A[1] == output[0];
      const bool check_n = in_B[1] == output[1];
      if (!(check_k && check_m && check_n)) {
        return c10::nullopt;
      }
      // [M, N, K]
      return TensorShape{output[0], output[1], in_A[0]};
    }
    case MatmulLayout::TN: {
      // in_A := [M, K]
      // in_B := [N, K]
      // output := [M, N]
      const bool check_k = in_A[1] == in_B[1];
      const bool check_m = in_A[0] == output[0];
      const bool check_n = in_B[0] == output[1];
      if (!(check_k && check_m && check_n)) {
        return c10::nullopt;
      }
      // [M, N, K]
      return TensorShape{output[0], output[1], in_A[1]};
    }
    default:
      return c10::nullopt;
  }
  return c10::nullopt;
}

std::string checkMatmulType(Fusion* fusion, const MmaOp* mma_expr) {
  const auto& fusion_inputs = fusion->inputs();
  const auto& fusion_outputs = fusion->outputs();
  const auto& mma_inputs = mma_expr->inputs();
  const auto& mma_outputs = mma_expr->outputs();

  const auto fusion_inputs_tvs =
      ir_utils::filterByType<TensorView>(fusion_inputs).vector();
  const auto fusion_outputs_tvs =
      ir_utils::filterByType<TensorView>(fusion_outputs).vector();

  using DimSizeType = std::decay<decltype(fusion_inputs)>::type::size_type;

  static_assert(
      std::is_same<
          DimSizeType,
          std::decay<decltype(fusion_outputs)>::type::size_type>::value,
      "The type used to define the number of dimension in input and output TV must be the same.");

  constexpr DimSizeType expected_gemm_dims = static_cast<DimSizeType>(2);
  constexpr size_t expected_number_of_inputs = 2;
  constexpr size_t expected_number_of_outputs = 1;

  // Quick checks
  {
    // Fusion can only have two TV inputs
    if (fusion_inputs.size() != fusion_inputs_tvs.size()) {
      return "Fusion inputs contain at least one non-TensorView object";
    }
    if (expected_number_of_inputs != fusion_inputs.size()) {
      return "Fusion inputs contain at least one non-TensorView object";
    }

    // Fusion can only have TVs as outputs, and there can be only one output
    if (fusion_outputs_tvs.size() != fusion_outputs.size()) {
      return "Fusion has output which is not a TensorView object";
    }
    if ((expected_number_of_outputs != fusion_outputs_tvs.size())) {
      return "Fusion has more than a single TensorView object in outputs";
    }

    // Each of fusion input TVs must have:
    //  - 2 concrete domains,
    //  - no broadcasts domain,
    for (const auto tv : fusion_inputs_tvs) {
      if (tv->hasBroadcast()) {
        return "Fusion input TV has broadcast domain";
      }
      const auto result =
          TensorDomain::noReductions(
              TensorDomain::noBroadcasts(tv->domain()->domain()))
              .size();
      if (result != expected_gemm_dims) {
        return "Fusion input TV has unsupported number of domains";
      }
    }

    // Each of fusion output TVs must have:
    // - 2 concrete domains,
    // - reduction domain,
    // - no broadcast domain,
    for (const auto tv : fusion_outputs_tvs) {
      if (tv->hasBroadcast()) {
        return "Fusion output TV has broadcast domain";
      }
      if (!tv->hasReduction()) {
        return "Fusion output TV has no reduction domain";
      }
      const auto result =
          TensorDomain::noReductions(
              TensorDomain::noBroadcasts(tv->domain()->domain()))
              .size();
      if (result != expected_gemm_dims) {
        return "Fusion output TV has unsupported number of domains";
      }
    }
  }

  // MmaOp inputs/outputs dependencies check
  {
    // Check the expected path between MmaOp input and fusion inputs
    const auto areMmaOpInputDependeciesValid = [](const Val* val) {
      if (val->definition()->isA<BroadcastOp>()) {
        const auto& bcast_inputs = val->definition()->inputs();
        // BroadcastOp has single input/output, not need to check other things
        return bcast_inputs.front()->isFusionInput();
      }
      return false;
    };

    // MmaOp input is a result of broadcast op with input being fusion input
    for (const auto* mma_in : mma_inputs) {
      if (!areMmaOpInputDependeciesValid(mma_in)) {
        return "MmaOp input has unsupported dependency";
      }
    }

    // MmaOp output must be a fusion output
    if (!mma_outputs.front()->isFusionOutput()) {
      return "Mma op output does not belong to fusion outputs";
    }
  }

  return "";
}

} // anonymous namespace

std::string getMatmulRunTimeRejectReason(
    Fusion* fusion,
    HeuristicSummary* data_cache,
    SchedulerRuntimeInfo& runtime_info) {
  // TODO: add proper set of checks
  return "";
}

std::string getMatmulCompileTimeRejectReason(Fusion* fusion) {
  // The plan:
  // 1. check if there is exactly one MmaOp defined in the fusion
  // 2. check if MmaOp inputs match any of supported inputs layout
  // 3. check if fusion represents expressions that are recognized by matmul
  //    scheduler

  // #1
  auto mma_exprs = ir_utils::getMmaOps(fusion);
  if (mma_exprs.size() != 1) {
    std::stringstream ss;
    ss << "Matmul scheduler supports fusions only with a single MMA op, got: "
       << mma_exprs.size();
    return ss.str();
  }

  // #2
  {
    for (const auto* mma_expr : mma_exprs) {
      const auto layout_data = getInputsLayout(mma_expr);
      if (layout_data.second) {
        return layout_data.second.value();
      }
    }
  }

  // #3
  {
    for (auto mma_expr : mma_exprs) {
      auto matmul_status = checkMatmulType(fusion, mma_expr);
      if (!matmul_status.empty()) {
        return matmul_status;
      }
    }
  }

  return "";
}

std::shared_ptr<MatmulParams> getMatmulHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FusionGuard fg(fusion);
  (void)data_cache;
  (void)runtime_info;
  auto params = std::make_shared<MatmulParams>();

  // Check initial conditions
  const auto fusion_exprs = fusion->exprs();
  auto mma_exprs = ir_utils::filterByType<MmaOp>(fusion_exprs).vector();
  if (mma_exprs.size() != 1) {
    // Support only for fusion with a single mma op
    return nullptr;
  }

  const auto layout = getInputsLayout(mma_exprs.front());
  if (layout.second) {
    // Layout check returned an error message
    if (isDebugDumpEnabled(DebugDumpOption::MatmulChecks)) {
      printMsg(layout.second.value());
    }
    return nullptr;
  }

  const auto problem_shape = getProblemShape(
      fusion, mma_exprs[0]->as<MmaOp>(), runtime_info, layout.first.value());
  if (!problem_shape) {
    // Failed to acquire problem shape
    return nullptr;
  }

  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  const auto mma_op = getMmaOp(
      device_prop->major * 10 + device_prop->minor, problem_shape.value());
  if (!mma_op) {
    // No heuristics can be prepared if mma op request is empty
    return nullptr;
  }

  // Populate heuristic details
  auto status = initCoreHeuristics(
      params, mma_op.value(), layout.first.value(), problem_shape.value());
  if (!status) {
    // Core part of heuristics failed to initialize
    return nullptr;
  }

  status = initExtraHeuristics(params, problem_shape.value());
  if (!status) {
    // Additional pieces of heuristics failed to initialize
    return nullptr;
  }

  // set kernel index mode
  params->cparams.index_type = getIndexType(problem_shape.value());

  if (isDebugDumpEnabled(DebugDumpOption::MatmulChecks)) {
    printMsg(params->toString());
  }

  return params;
}

} // namespace nvfuser
