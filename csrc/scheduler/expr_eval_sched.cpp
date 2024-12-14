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

  auto exprs = fusion->exprs();
  if (exprs.size() != 1) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Fusion must contain only a single expression.");
    return false;
  }

  if (exprs.front()->isOneOf<SdpaFwdOp, SdpaBwdOp, EmbeddingOp>()) {
    return true;
  }

  if (exprs.front()->isOneOf<LinearOp, MatmulOp>()) {
    if (isOptionDisabled(DisableOption::MatmulExprEval)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(),
          "Matmul ATen evaluation was disabled by NVFUSER_DISABLE=matmul_expr_eval");
      return false;
    }
    return true;
  }

  scheduler_debug_utils::canScheduleRejectReason(
      schedulerType(),
      "Fusion must contain only a single expression of type MatmulOp/LinearOp/SdpaFwdOp/SdpaBwdOp");
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
