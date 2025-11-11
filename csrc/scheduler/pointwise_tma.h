// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <scheduler/pointwise_heuristic.h>

namespace nvfuser {
namespace pointwise_tma {

void getHeuristics(
    PointwiseParams* pparams,
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache);

void scheduleFusion(Fusion* fusion, const PointwiseParams* pparams);

} // namespace pointwise_tma
} // namespace nvfuser
