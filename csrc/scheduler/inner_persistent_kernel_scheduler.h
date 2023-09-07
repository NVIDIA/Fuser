// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <scheduler/registry.h>
#include <scheduler/utils.h>

namespace nvfuser {

class HeuristicSummary;

class InnerPersistentKernelScheduler : public PersistentKernelScheduler {
 public:
  explicit InnerPersistentKernelScheduler(
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

  static bool checkReductionPattern(
      Fusion* fusion,
      const std::vector<TensorView*>& inner_reduction_tvs,
      const std::vector<TensorView*>& outer_reduction_tvs);

  static std::pair<int64_t, int64_t> getPersistentBufferSize(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache,
      const std::vector<TensorView*>& reduction_tvs);

  static bool canScheduleRunTimeOuter(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache,
      const std::vector<TensorView*>& reduction_tvs,
      const scheduler_utils::ReductionTvProperties& properties);
};

} // namespace nvfuser
