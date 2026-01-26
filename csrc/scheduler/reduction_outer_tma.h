// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <scheduler/reduction.h>
#include <scheduler/reduction_utils.h>

namespace nvfuser {
namespace reduction {
namespace outer_tma {
std::unique_ptr<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const reduction_scheduler_utils::FusionRuntimeProperties& props);

void scheduleReduction(Fusion* fusion, const ReductionParams* rparams);
} // namespace outer_tma
} // namespace reduction
} // namespace nvfuser
