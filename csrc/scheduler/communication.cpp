// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <alias_analysis.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <scheduler/communication.h>
#include <scheduler/debug_utils.h>
#include <scheduler/mark_aliases.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>

namespace nvfuser {

//! Check if the given fusion is a single communication expression
bool CommunicationScheduler::canScheduleCompileTime(Fusion* fusion) {
  const std::vector<Expr*>& exprs = fusion->exprs();
  if (exprs.size() != 1) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Expected only one expression but found: ",
        exprs.size());
    return false;
  }
  Expr* e = exprs[0];

  if (!isResharding(e)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Expected the expression to be resharding: ",
        e->toString());
    return false;
  }

  return true;
}

bool CommunicationScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  return true;
}

std::unique_ptr<HeuristicParams> CommunicationScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  auto params = std::make_unique<HeuristicParams>(SchedulerType::Communication);
  params->cparams.index_type = runtime_info.getIndexType();
  return params;
}

void CommunicationScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  NVF_ERROR(
      params->scheduler_type == schedulerType(),
      "Invalid heuristic sent to Communication scheduler: ",
      params);
}

} // namespace nvfuser
