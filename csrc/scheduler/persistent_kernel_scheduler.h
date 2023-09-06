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

class PersistentKernelScheduler : public SchedulerEntry {
 public:
  explicit PersistentKernelScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  virtual ~PersistentKernelScheduler() = default;

  virtual void schedule(Fusion* fusion) = 0;

  static bool canScheduleCompileTime(Fusion* fusion) = 0;

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) = 0;

 private:
  virtual void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) = 0;
};

} // namespace nvfuser
