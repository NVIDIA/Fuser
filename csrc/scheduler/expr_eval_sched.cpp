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

namespace nvfuser {

// Check if the fusion has a single MatmulOp/LinearOp node
bool ExprEvalScheduler::canScheduleCompileTime(Fusion* fusion) {
  auto exprs = fusion->exprs();
  if (exprs.size() == 1 &&
      (exprs.front()->isA<MatmulOp>() || exprs.front()->isA<LinearOp>())) {
    return true;
  }
  scheduler_debug_utils::canScheduleRejectReason(
      heuristicType(),
      "Fusion must contain a single expression of type MatmulOp or LinearOp");
  return false;
}

void ExprEvalScheduler::schedule(Fusion* fusion) {
  fusion->aliasOutputToInput(
      fusion->outputs()[0], /*input=*/nullptr, AllocationType::Evaluate);
}

} // namespace nvfuser
