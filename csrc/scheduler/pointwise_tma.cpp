// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <scheduler/pointwise_tma.h>

namespace nvfuser {
namespace pointwise {
namespace tma {

std::unique_ptr<PointwiseParams> getPointwiseHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);
  auto params = std::make_unique<PointwiseParams>();
  params->tag = "Pointwise TMA heuristics";
  params->use_tma_load = true;
  NVF_THROW("Schedule pointwise using TMA");
  return params;
}

// TODO: Inline intermediate operations (avoid inlining unrolled/vectorized
// input/output caches)
void schedulePointwise(Fusion* fusion, const PointwiseParams* pparams) {
  FusionGuard fg(fusion);
  NVF_THROW("Schedule pointwise using TMA");
}
} // namespace tma
} // namespace pointwise
} // namespace nvfuser
