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
#include <debug.h>
#include <executor_utils.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/utils.h>
#include <options.h>
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

using MatmulLayout = MmaOptions::MmaLayout;
//! Access to the structure should be done with labels defined in
//!  MmaOptions::MmaDomains.
using ProblemShape = std::array<int64_t, 3>;

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
    const MmaMacro& mma_op,
    const ProblemShape& problem_shape) {
  const GemmTile instruction_tile = getMmaOpShape(mma_op);
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

  params->mma_macro = mma_op;
  params->tile_sizes = {cta_tile, warp_tile, instruction_tile};

  // stages and async mem copy
  {
    // NOTE: compilation errors when async is enabled on Turing devices
    if (isAmpere(mma_op)) {
      constexpr int stages = 3;

      params->async_gmem_load_operands = true;
      params->double_buffer_options.double_buffer_smem_write = true;
      params->double_buffer_options.double_buffer_smem_read = true;
      params->double_buffer_options.smem_double_buffer_stage = stages;
    }
  }

  return true;
}

//! A helper for getting problem shape from fusion and runtime info.
ProblemShape getProblemShape(
    Fusion* fusion,
    const MmaOp* mma_expr,
    SchedulerRuntimeInfo& runtime_info) {
  const auto mma_output_domains = mma_utils::getProblemIterDomains(fusion);
  if (!mma_output_domains.isValid()) {
    NVF_ERROR(false, mma_output_domains.getErrorMsg());
  }

  const auto [m, n, k] = mma_output_domains.getData();

  auto m_extend = runtime_info.expressionEvaluator().evaluate(m->extent());
  auto n_extend = runtime_info.expressionEvaluator().evaluate(n->extent());
  auto k_extend = runtime_info.expressionEvaluator().evaluate(k->extent());

  if (!(m_extend && n_extend && k_extend)) {
    NVF_ERROR(
        false,
        "Failed to acquire one of problem dimensions, M(",
        m_extend.hasValue(),
        "), N(",
        n_extend.hasValue(),
        " K(",
        k_extend.hasValue(),
        ")");
  }

  return ProblemShape{
      m_extend.as<int64_t>(), n_extend.as<int64_t>(), k_extend.as<int64_t>()};
}

std::string isMatmulFusionDefinitionSupported(
    Fusion* fusion,
    const MmaOp* mma_expr) {
  const auto& fusion_inputs = fusion->inputs();
  const auto& fusion_outputs = fusion->outputs();
  const auto& mma_inputs = mma_expr->inputs();
  const auto mma_output = mma_expr->out()->as<TensorView>();

  const auto fusion_inputs_tvs =
      ir_utils::filterByType<TensorView>(fusion_inputs).vector();
  const auto fusion_outputs_tvs =
      ir_utils::filterByType<TensorView>(fusion_outputs).vector();

  constexpr size_t minimal_number_of_inputs = 2;
  constexpr size_t expected_number_of_outputs = 1;

  // Quick checks - MmaOp
  {
    // Check if MmaOp represents gemm (requires M/N/K == 1, B == 0)
    //  or bgemm (requires M/N/K/B == 1)
    constexpr size_t expected_axes_numbers = 1;
    if (mma_expr->mAxes().size() != expected_axes_numbers ||
        mma_expr->nAxes().size() != expected_axes_numbers ||
        mma_expr->kAxes().size() != expected_axes_numbers ||
        mma_expr->batchAxes().size() > expected_axes_numbers) {
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

    // Fusion has only TVs as outputs, and we expect only one object in the list
    if ((expected_number_of_outputs != fusion_outputs_tvs.size())) {
      return "Fusion has more than a single TensorView object in its outputs";
    }
  }

  // Fusion topology check
  {
    const auto& roles_map_opt = mma_utils::getTensorsRoles(fusion);
    if (!roles_map_opt.isValid()) {
      return roles_map_opt.getErrorMsg();
    }

    const auto& roles_map = roles_map_opt.getData();
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
      if (MATMUL_CORE_ROLES_EXPECTED_COUNT == entry->second.size()) {
        tvs_with_roles.insert(entry->second.begin(), entry->second.end());
      } else {
        return "There is more than a single fusion output that can be MMA output";
      }
    } else {
      return "No candidate in fusion outputs MMA output";
    }

    // Non-core roles are optional, no requirements for their presence
    entry = roles_map.find(MatmulRole::INPUT_C);
    if (entry != roles_map.end()) {
      tvs_with_roles.insert(entry->second.begin(), entry->second.end());
    }

    const auto in_out_tvs_count =
        fusion_inputs_tvs.size() + fusion_outputs_tvs.size();
    if (in_out_tvs_count != tvs_with_roles.size()) {
      return "Detected input/output TVs without assigned roles";
    }
  }

  // MmaOp inputs/outputs dependencies check
  // TODO: check to be removed when more rules are added to TV roles
  //  calculations
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
  auto mma_exprs = ir_utils::getOpsOfType<MmaOp>(fusion);
  if (mma_exprs.size() != 1) {
    std::stringstream ss;
    ss << "Matmul scheduler supports fusions only with a single MMA op, got: "
       << mma_exprs.size();
    return ss.str();
  }

  // #2
  {
    const auto input_layout_opt = mma_utils::getMatmulLayout(fusion);
    if (!input_layout_opt.isValid()) {
      return input_layout_opt.getErrorMsg();
    }
  }

  // #3
  {
    for (auto mma_expr : mma_exprs) {
      auto support_status = isMatmulFusionDefinitionSupported(fusion, mma_expr);
      if (!support_status.empty()) {
        return support_status;
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
  auto mma_exprs = ir_utils::getOpsOfType<MmaOp>(fusion);
  NVF_ERROR(mma_exprs.size() == 1, "Support only fusion with a single mma op.");

  const auto problem_shape =
      getProblemShape(fusion, mma_exprs.front()->as<MmaOp>(), runtime_info);

  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  const auto mma_op =
      getMmaOp(device_prop->major * 10 + device_prop->minor, problem_shape);
  NVF_ERROR(
      mma_op.has_value(), "Failed to determine a MMA op for given problem.");

  // Populate heuristic details
  auto status = initCoreHeuristics(params, mma_op.value(), problem_shape);
  NVF_ERROR(status, "Initialization of core part of heuristics failed.");

  // Set kernel index mode
  params->cparams.index_type = runtime_info.getIndexType();

  // Disable magic zero for matmul kernels
  params->cparams.enable_magic_zero = false;

  // Set whether to use shared memory for epilogue
  const auto& roles_map_opt = mma_utils::getTensorsRoles(fusion);
  NVF_ERROR(roles_map_opt.isValid(), "Tensor roles map in mma is not valid.");

  const auto roles_map = roles_map_opt.getData();
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
