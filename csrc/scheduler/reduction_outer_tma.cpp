// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <scheduler/reduction_outer_tma.h>

namespace nvfuser {
namespace reduction {
namespace outer_tma {

std::unique_ptr<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const reduction_scheduler_utils::FusionRuntimeProperties& props) {
  FusionGuard fg(fusion);
  auto params = std::make_unique<ReductionParams>();
  params->tag = "Outer Reduction TMA heuristics";
  NVF_THROW("Schedule outer reduction using TMA");
  return params;
}

void scheduleReduction(Fusion* fusion, const ReductionParams* rparams) {
  FusionGuard fg(fusion);
  NVF_THROW("Schedule outer reduction using TMA");
}
} // namespace outer_tma
} // namespace reduction
} // namespace nvfuser
