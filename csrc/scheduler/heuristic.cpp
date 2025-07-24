// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/heuristic.h>

#include <fusion.h>
#include <scheduler/registry.h>
#include <scheduler/runtime_info.h>

namespace nvfuser {
HeuristicParamsList::HeuristicParamsList(
    SchedulerType scheduler_type,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache)
    : is_segmented_(false) {
  heuristics_.emplace_back(
      SchedulerEntry::makeSchedulerInstance(scheduler_type)
          ->computeHeuristics(runtime_info.fusion(), runtime_info, data_cache));
}

} // namespace nvfuser
