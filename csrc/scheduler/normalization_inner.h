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
#include <visibility.h>

// TODO: If caching inputs would require persistence we are sending it to the
// persistent kerenl scheduler. This isn't necessary if the only persistent
// buffers are inputs as we could re-read them from global memory. Need to
// consider if this is worth implementing.

namespace nvfuser {

class SchedulerRuntimeInfo;
class HeuristicSummary;

class InnerPersistentKernelScheduler : public SchedulerEntry {
 public:
  void schedule(Fusion* fusion, const HeuristicParams* params) override;

  bool canScheduleCompileTime(Fusion* fusion) override;

  bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) override;

  constexpr static HeuristicType heuristicType() {
    return HeuristicType::InnerPersistent;
  }

  std::unique_ptr<HeuristicParams> computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache) override;
};

NVF_API std::unique_ptr<ReductionParams> getInnerPersistentHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache = nullptr);

NVF_API std::unique_ptr<ReductionParams> getInnerPersistentHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache = nullptr);

NVF_API void scheduleInnerPersistentKernel(
    Fusion* fusion,
    const ReductionParams* rparams);

} // namespace nvfuser
