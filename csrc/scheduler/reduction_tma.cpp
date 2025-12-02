// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <scheduler/reduction_tma.h>

namespace nvfuser {
namespace reduction {
namespace tma {

std::unique_ptr<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);
  auto params = std::make_unique<ReductionParams>();
  params->tag = "Reduction TMA heuristics";
  params->use_tma_load = true;
  NVF_THROW("Schedule reduction using TMA");
  return params;
}

void scheduleReduction(Fusion* fusion, const ReductionParams* pparams) {
  FusionGuard fg(fusion);
  NVF_THROW("Reduction pointwise using TMA");
}
} // namespace tma
} // namespace reduction
} // namespace nvfuser
