// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/heuristic.h>
#include <scheduler/registry.h>

namespace nvfuser {
HeuristicParamsList::HeuristicParamsList(
    HeuristicType schedule_heuristic,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache)
    : is_segmented_(false) {
  heuristics_.emplace_back(
      SchedulerEntry::makeSchedulerInstance(schedule_heuristic)
          ->computeHeuristics(runtime_info.fusion(), runtime_info, data_cache));
}

} // namespace nvfuser
