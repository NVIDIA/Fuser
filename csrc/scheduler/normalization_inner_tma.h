// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <memory>

#include <fusion.h>
#include <scheduler/normalization_inner.h>
#include <scheduler/reduction_heuristic.h>

namespace nvfuser {

class SchedulerRuntimeInfo;
class HeuristicDataCache;

namespace inner_persistent {
namespace tma {

std::unique_ptr<ReductionParams> getInnerPersistentHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache);

void scheduleInnerPersistent(Fusion* fusion, const ReductionParams* rparams);

} // namespace tma
} // namespace inner_persistent
} // namespace nvfuser
