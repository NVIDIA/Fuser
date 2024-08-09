// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <scheduler/heuristic.h>
#include <scheduler/registry.h>

namespace nvfuser {

class Fusion;
class SchedulerRuntimeInfo;
class HeuristicSummary;

// ExprEval scheduler represents the case where we allocate outputs directly
// using EE. No code is generated.
class ExprEvalScheduler : public SchedulerEntry {
 public:
  explicit ExprEvalScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(heuristicType()) {
    params_ =
        std::make_shared<HeuristicParams>("", runtime_info.getIndexType());
  }

  // This scheduler only accepts MatmulOp.
  static bool canScheduleCompileTime(Fusion* fusion);

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache) {
    return true;
  }

  constexpr static ScheduleHeuristic heuristicType() {
    return ScheduleHeuristic::ExprEval;
  }

  void schedule(Fusion* fusion) override;
};

} // namespace nvfuser
