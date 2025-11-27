// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <scheduler/normalization_inner_tma.h>

#include <memory>

namespace nvfuser {
namespace normalization_inner {
namespace tma {

std::unique_ptr<ReductionParams> getInnerPersistentHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);
  auto params = std::make_unique<ReductionParams>(
      InnerPersistentKernelScheduler::schedulerType());
  params->tag = "Inner Persistent TMA heuristics";
  NVF_THROW("Schedule inner persistent using TMA");
  return params;
}

void scheduleInnerPersistent(Fusion* fusion, const ReductionParams* rparams) {
  FusionGuard fg(fusion);
  NVF_THROW("Schedule inner persistent using TMA");
}

} // namespace tma
} // namespace normalization_inner
} // namespace nvfuser
