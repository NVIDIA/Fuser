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

namespace nvfuser {
namespace reduction {
namespace tma {
std::unique_ptr<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache);

void scheduleReduction(Fusion* fusion, const ReductionParams* pparams);
} // namespace tma
} // namespace reduction
} // namespace nvfuser
