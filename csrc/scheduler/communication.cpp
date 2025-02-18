// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <alias_analysis.h>
#include <host_ir/lower.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/mark_aliases.h>
#include <scheduler/communication.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>

namespace nvfuser {

//! Check if the given fusion is a single communication expression
bool CommunicationScheduler::canScheduleCompileTime(Fusion* fusion) {
  const std::vector<Expr*>& exprs = fusion->exprs();
  return (exprs.size() == 1 && isResharding(exprs[0]) &&
      HostIrLower::canLower(exprs[0]));
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
  return std::make_unique<HeuristicParams>(SchedulerType::Communication);
}

void CommunicationScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  NVF_ERROR(
      params->scheduler_type == schedulerType(),
      "Invalid heuristic sent to Communication scheduler: ",
      params);
}

} // namespace nvfuser
