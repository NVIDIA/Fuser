// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <scheduler/normalization_helper.h>
#include <scheduler/registry.h>
#include <scheduler/utils.h>

namespace nvfuser {

class HeuristicSummary;

class InnerOuterPersistentKernelScheduler : public SchedulerEntry,
                                            public PersistentSchedulerHelper {
 public:
  explicit InnerOuterPersistentKernelScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  void schedule(Fusion* fusion) override;

  static bool canScheduleCompileTime(Fusion* fusion);

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  static std::shared_ptr<ReductionParams> getHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  static void scheduleKernel(Fusion* fusion, const ReductionParams& rparams);

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);
};

} // namespace nvfuser
