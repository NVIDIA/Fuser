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
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_heuristic.h>

namespace nvfuser {

class SchedulerRuntimeInfo;
class HeuristicDataCache;

namespace normalization_inner {
namespace non_tma {

using PersistentKernelProperties =
    normalization_scheduler_utils::PersistentKernelProperties;

std::unique_ptr<ReductionParams> getInnerPersistentHeuristics(
    Fusion* fusion,
    const PersistentKernelProperties& prop,
    HeuristicDataCache* data_cache);

void scheduleInnerPersistent(Fusion* fusion, const ReductionParams* rparams);

} // namespace non_tma
} // namespace normalization_inner
} // namespace nvfuser
