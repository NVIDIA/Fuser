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
class HeuristicDataCache;

class ResizeScheduler : public SchedulerEntry {
 public:
  bool canScheduleCompileTime(Fusion* fusion) override;
  bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache = nullptr) override {
    return true;
  }

  std::unique_ptr<HeuristicParams> computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache) override;

  void schedule(Fusion* fusion, const HeuristicParams* params) override;

  constexpr static SchedulerType schedulerType() {
    return SchedulerType::Resize;
  }

 private:
  int64_t getVersion() const;

  bool canScheduleCompileTimeV1(Fusion* fusion);
  bool canScheduleCompileTimeV2(Fusion* fusion);

  void scheduleV1(Fusion* fusion, const HeuristicParams* params);
  void scheduleV2(Fusion* fusion, const HeuristicParams* params);
};

} // namespace nvfuser
