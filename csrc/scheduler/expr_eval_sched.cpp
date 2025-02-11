// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/expr_eval_sched.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>

namespace nvfuser {

// Check if the fusion has a single MatmulOp/LinearOp node
bool ExprEvalScheduler::canScheduleCompileTime(Fusion* fusion) {
  if (scheduler_utils::isResharding(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Fusion is resharding.");
    return false;
  }

  auto expr_check = [](Expr* expr) {
    return expr->isOneOf<SdpaFwdOp, SdpaBwdOp, EmbeddingFwdOp, GetMetaData>() ||
        (expr->isOneOf<LinearOp, MatmulOp>() &&
         !isOptionDisabled(DisableOption::MatmulExprEval)) ||
        ir_utils::isScalarOp(expr);
  };

  auto exprs = fusion->exprs();

  for (auto expr : exprs) {
    if (!expr_check(expr)) {
      scheduler_debug_utils::canScheduleRejectReason(
          "Expr not supported in ExprEvalScheduler:", expr->toString());
      return false;
    }
  }

  return true;
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
