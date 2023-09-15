// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <scheduler/registry.h>

namespace nvfuser {

// convenience function to get persistent kernel heuristics using
// runtime_inputs. Used in cpp tests.
TORCH_CUDA_CU_API std::shared_ptr<ReductionParams> getInnerPersistentHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache = nullptr);

class HeuristicSummary;

class InnerPersistentKernelScheduler : public SchedulerEntry {
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

  static std::shared_ptr<ReductionParams> getPersistentHeuristic(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  static void schedulePersistentKernel(
      Fusion* fusion,
      const ReductionParams& rparams);

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);
};

} // namespace nvfuser
