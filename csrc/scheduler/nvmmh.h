// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <mma_type.h>
#include <scheduler/heuristic.h>
#include <scheduler/registry.h>
#include <scheduler/scheduler_types.h>
#include <visibility.h>

namespace nvfuser {

void fillNvMatmulHeuristicsParams(CutlassParams* params, Fusion* fusion, SchedulerRuntimeInfo& runtime_info);

} // namespace nvfuser
