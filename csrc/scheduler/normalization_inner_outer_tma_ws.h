// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <scheduler/reduction_heuristic.h>

namespace nvfuser {
namespace inner_outer_tma_warp_specialized {
void getHeuristics(
    ReductionParams* rparams,
    const int64_t outer_dim_numel,
    const int64_t inner_dim_numel,
    const int64_t regs_buffer_size_bit,
    const int64_t smem_buffer_size_bit,
    const int64_t smem_overhead_bit,
    const size_t tmp_gmem_dtype_size_bit,
    const size_t vectorize_factor,
    const int64_t hp_threads_per_block_min,
    const int64_t hp_threads_per_block_max,
    const bool project_to_input,
    const PrimDataType index_type);

void scheduleFusion(Fusion* fusion, const ReductionParams* rparams);
} // namespace inner_outer_tma_warp_specialized
} // namespace nvfuser
