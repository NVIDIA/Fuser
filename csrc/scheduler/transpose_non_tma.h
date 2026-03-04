// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include "fusion.h"
#include "scheduler/transpose.h"

namespace nvfuser {
namespace transpose {
namespace non_tma {

std::unique_ptr<TransposeParams> getTransposeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache);

void scheduleTranspose(Fusion* fusion, const TransposeParams* tparams);

} // namespace non_tma
} // namespace transpose
} // namespace nvfuser
