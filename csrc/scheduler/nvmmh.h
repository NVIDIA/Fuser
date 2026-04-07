// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

namespace nvfuser {

class CutlassParams;
class Fusion;
class SchedulerRuntimeInfo;

//! This tries to load libnvMatmulHeuristics.so and, if successful, calls out to
//! it to retrieve a valid kernel config. It then translates it into a
//! CutlassParams, filling the appropriate values in the provided params
void fillNvMatmulHeuristicsParams(
    CutlassParams* params,
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info);

} // namespace nvfuser
