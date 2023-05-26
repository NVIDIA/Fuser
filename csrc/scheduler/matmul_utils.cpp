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
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/utils.h>
#include <algorithm>
#include <deque>
#include <iostream>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include "ATen/cuda/CUDAContext.h"
#include "mma_type.h"
#include "type.h"
#include "utils.h"

namespace nvfuser {
namespace {

using MatmulLayout = MmaOptions::MmaLayout;
using LayoutData =
    std::pair<std::optional<MatmulLayout>, std::optional<std::string>>;
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
// constexpr size_t K_POS = 2;
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

//! A helper for deciding the type of MMA op for given fusion and problem shape.
inline std::optional<MmaOptions::MacroType> getMmaOp(
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
    case 86:
    case 89:
      return (use_small_n) ? MacroType::Ampere_16_8_16
                           : MacroType::Ampere_16_16_16;
    default:
      break;
  }
  return std::nullopt;
}

//! A wrapper for core heuristics initialization
inline bool initCoreHeuristics(
    std::shared_ptr<MatmulParams> params,
    const MmaOptions::MacroType& mma_op,
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

  params->mma_macro = mma_op;
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
std::optional<ProblemShape> getProblemShape(
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
      if (path.size() >= 2 && path.at(1)->isA<TensorView>() &&
          path.at(1)->as<TensorView>()->hasRFactor()) {
        tvs.push_back(path.at(1));
      } else if (path.front()->isA<TensorView>()) {
        tvs.push_back(path.front());
      }
    }
    return (tvs.size() == 1) ? tvs[0] : nullptr;
  };

  const auto* tv_input_A =
      getKeyTvFromPathBetween(fusion_inputs, {mma_inputs[0]});
  if (nullptr == tv_input_A) {
    return std::nullopt;
  }

  const auto* tv_input_B =
      getKeyTvFromPathBetween(fusion_inputs, {mma_inputs[1]});
  if (nullptr == tv_input_B) {
    return std::nullopt;
  }

  const auto* tv_output =
      getKeyTvFromPathBetween({mma_outputs[0]}, fusion_outputs);
  if (nullptr == tv_output) {
    return std::nullopt;
  }

  // A helper for populating concrete domains from TensorView
  const auto getShape = [&runtime_info](const TensorView* tv) {
    TensorShape tv_shape;
    const auto concrete_domains = TensorDomain::noReductions(
        TensorDomain::noBroadcasts(tv->getLeafDomain()));
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
    return std::nullopt;
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
        return std::nullopt;
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
        return std::nullopt;
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
        return std::nullopt;
      }
      // [M, N, K]
      return TensorShape{output[0], output[1], in_A[1]};
    }
    default:
      return std::nullopt;
  }
  return std::nullopt;
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

  // Quick checks - MmaOp
  {
    // Check if MmaOp processes single gemm
    constexpr size_t expected_axes_numbers = 1;
    if (mma_expr->mAxes().size() != expected_axes_numbers ||
        mma_expr->nAxes().size() != expected_axes_numbers ||
        mma_expr->kAxes().size() != expected_axes_numbers ||
        !mma_expr->batchAxes().empty()) {
      return "MmaOp has unsupported number of one of M/N/K/Batch axes";
    }
  }

  // Quick checks - Fusion
  {
    // Fusion can only have two TV inputs
    if (fusion_inputs.size() != fusion_inputs_tvs.size()) {
      return "Fusion inputs contain at least one non-TensorView object";
    }
    if (expected_number_of_inputs != fusion_inputs.size()) {
      return "Fusion inputs contain at least one non-TensorView object";
    }

    // Fusion has only TVs as outputs, and we expect only one object in the list
    if ((expected_number_of_outputs != fusion_outputs_tvs.size())) {
      return "Fusion has more than a single TensorView object in its outputs";
    }

    // Each of fusion input TVs must have:
    //  - 2 concrete domains,
    //  - no broadcasts domain,
    for (const auto tv : fusion_inputs_tvs) {
      if (tv->hasBroadcast()) {
        return "Fusion input TV has broadcast domain";
      }
      const auto result = TensorDomain::noReductions(
                              TensorDomain::noBroadcasts(tv->getLeafDomain()))
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
      const auto result = TensorDomain::noReductions(
                              TensorDomain::noBroadcasts(tv->getLeafDomain()))
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
        return bcast_inputs.front()->isFusionInput() ||
            (dynamic_cast<LoadStoreOp*>(bcast_inputs.front()->definition()) !=
             nullptr);
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
      const auto input_layout = mma_expr->layout();
      if (!input_layout) {
        return "Failed to acquire inputs layout.";
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
  auto mma_exprs = ir_utils::getMmaOps(fusion);
  TORCH_INTERNAL_ASSERT(
      mma_exprs.size() == 1, "Support only fusion with a single mma op.");

  const auto layout = mma_exprs.front()->layout();
  TORCH_INTERNAL_ASSERT(layout.has_value(), "Failed to acquire inputs layout.");

  const auto problem_shape = getProblemShape(
      fusion, mma_exprs.front()->as<MmaOp>(), runtime_info, layout.value());
  TORCH_INTERNAL_ASSERT(
      problem_shape.has_value(), "Failed to acquire problem shape.");

  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  const auto mma_op = getMmaOp(
      device_prop->major * 10 + device_prop->minor, problem_shape.value());
  TORCH_INTERNAL_ASSERT(
      mma_op.has_value(), "Can not determine MMA op for problem.");

  // Populate heuristic details
  auto status =
      initCoreHeuristics(params, mma_op.value(), problem_shape.value());
  TORCH_INTERNAL_ASSERT(
      status, "Core part of heuristics failed to initialize.");

  status = initExtraHeuristics(params, problem_shape.value());
  TORCH_INTERNAL_ASSERT(
      status, "Additional part of heuristics failed to initialize.");

  // Set kernel index mode
  params->cparams.index_type = runtime_info.getIndexType();

  if (isDebugDumpEnabled(DebugDumpOption::MatmulChecks)) {
    printMsg(params->toString());
  }

  return params;
}

} // namespace nvfuser
