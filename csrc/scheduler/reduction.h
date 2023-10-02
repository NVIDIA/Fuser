// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/ivalue.h>
#include <exceptions.h>
#include <fusion.h>
#include <scheduler/reduction_heuristic.h>
#include <scheduler/registry.h>

namespace nvfuser {

class SchedulerRuntimeInfo;
class HeuristicSummary;

std::shared_ptr<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache = nullptr);

std::shared_ptr<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache = nullptr);

void scheduleReduction(Fusion* fusion, const ReductionParams& rparams);

class ReductionScheduler : public SchedulerEntry {
 public:
  explicit ReductionScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  void schedule(Fusion* fusion) override;

  static bool canScheduleCompileTime(Fusion* fusion);

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);
};

} // namespace nvfuser
