// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <scheduler/matmul_heuristic.h>
#include <visibility.h>

namespace nvfuser {

NVF_API void scheduleMultipleMatmuls(
    Fusion* fusion,
    const MatmulParams* mparams);

} // namespace nvfuser
