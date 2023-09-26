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
#include <scheduler/utils.h>

// TODO: If caching inputs would require persistence we are sending it to the
// persistent kerenl scheduler. This isn't necessary if the only persistent
// buffers are inputs as we could re-read them from global memory. Need to
// consider if this is worth implementing.

namespace nvfuser {

class SchedulerRuntimeInfo;
class HeuristicSummary;

class OuterPersistentKernelScheduler : public SchedulerEntry {
 public:
  explicit OuterPersistentKernelScheduler(
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

std::shared_ptr<ReductionParams> getOuterPersistentHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache = nullptr);

std::shared_ptr<ReductionParams> getOuterPersistentHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache = nullptr);

void scheduleOuterPersistentKernel(
    Fusion* fusion,
    const ReductionParams& rparams);

} // namespace nvfuser
