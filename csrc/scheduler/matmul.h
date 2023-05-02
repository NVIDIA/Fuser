// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/ivalue.h>

#include <fusion.h>
#include <mma_type.h>
#include <scheduler/matmul_heuristic.h>

namespace nvfuser {

TORCH_CUDA_CU_API void scheduleMatmul(
    Fusion* fusion,
    const MatmulParams& params);

} // namespace nvfuser
