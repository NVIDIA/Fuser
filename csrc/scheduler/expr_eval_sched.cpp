// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <alias_analysis.h>
#include <ir/utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/expr_eval_sched.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>

namespace nvfuser {

namespace {
bool allOutputsArePointerArithmetics(Fusion* fusion) {
  const AliasAnalysisResult analysis =
      findAliases(fusion, EmptyAllocationAs::kLogical);

  auto is_pointer_arithmetic = [&](TensorView* out) -> bool {
    // Check out has an alias and out is not an inplace update target.
    if (fusion->getOutputAlias(out).type == AllocationType::ReuseBuffer) {
      return false;
    }

    // When `out` happens to be a fusion input (unlikely but possible), we
    // treat `out` as an alias. This check is necessary because getRoot(out)
    // never returns `out` itself due to the way it is implemented.
    if (out->isFusionInput()) {
      return true;
    }

    TensorView* root = analysis.getRoot(out);
    return root != nullptr && root->isFusionInput();
  };

  auto out_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
  return std::all_of(out_tvs.begin(), out_tvs.end(), is_pointer_arithmetic);
}
} // namespace

// Check if the fusion has a single MatmulOp/LinearOp node
bool ExprEvalScheduler::canScheduleCompileTime(Fusion* fusion) {
  if (scheduler_utils::isResharding(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Fusion is resharding.");
    return false;
  }

  if (allOutputsArePointerArithmetics(fusion)) {
    return true;
  }

  auto exprs = fusion->exprs();
  if (exprs.size() != 1) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Fusion must contain only a single expression.");
    return false;
  }

  // TODO: remove IndexPutAccumulateOp
  if (exprs.front()
          ->isOneOf<
              ScatterOp,
              SdpaFwdOp,
              SdpaBwdOp,
              EmbeddingFwdOp,
              IndexPutAccumulateOp>()) {
    return true;
  }

  if (exprs.front()->isOneOf<LinearOp, MatmulOp>()) {
    if (isOptionDisabled(DisableOption::MatmulExprEval)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(),
          "Matmul ATen evaluation was disabled by "
          "NVFUSER_DISABLE=matmul_expr_eval");
      return false;
    }
    return true;
  }

  scheduler_debug_utils::canScheduleRejectReason(
      schedulerType(),
      "Fusion must contain only a single expression of type "
      "MatmulOp/LinearOp/SdpaFwdOp/SdpaBwdOp");
  return false;
}

void ExprEvalScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  NVF_ERROR(
      params->scheduler_type == schedulerType(),
      "Invalid heuristic sent to ExprEval scheduler: ",
      params);

  std::for_each(
      fusion->outputs().begin(), fusion->outputs().end(), [&](Val* out) {
        fusion->aliasOutputToInput(
            out, /*input=*/nullptr, AllocationType::Evaluate);
      });
}

std::unique_ptr<HeuristicParams> ExprEvalScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  auto params = std::make_unique<HeuristicParams>(SchedulerType::ExprEval);
  params->cparams.index_type = runtime_info.getIndexType();
  return params;
}

} // namespace nvfuser
