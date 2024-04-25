// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/no_op.h>
#include <scheduler/registry_utils.h>

namespace nvfuser {

template <typename... Args>
void vlog(const Args&... args) {
  scheduler_debug_utils::log("[Expression Evaluator Scheduler] ", args...);
}

ExprEvalScheduler::ExprEvalScheduler(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache)
    : SchedulerEntry(heuristicType()) {
  params_ = std::make_shared<ExprEvalHeuristic>("", runtime_info.getIndexType());
}

//! Check if the no-op heuristics apply in given fusion
bool ExprEvalScheduler::canScheduleCompileTime(Fusion* fusion) {
  // Check if the fusion has matmul node and accept
  if (fusion->outputs().size() == 1 && fusion->outputs().front()->isA<MatmulOp>()) {
    return true;
  }
  scheduler_debug_utils::canScheduleRejectReason(
          heuristicType(), "Only accepts MatmulOp");
  return false;
}

bool ExprEvalScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  return true;
}

void ExprEvalScheduler::schedule(Fusion* fusion) {
  fusion->aliasOutputToInput(
        fusion->outputs()[0], /*input=*/nullptr, AllocationType::Evaluate);
}

void ExprEvalScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  return;
}
} // namespace nvfuser