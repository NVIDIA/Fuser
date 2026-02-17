// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include "fusion.h"
#include "scheduler/reduction.h"
#include "scheduler/reduction_utils.h"

namespace nvfuser {

class TmaOuterReductionParams : public HeuristicParams {
 public:
  TmaOuterReductionParams(
      SchedulerType scheduler_type = SchedulerType::Reduction)
      : HeuristicParams(scheduler_type) {};

  // Thread block dimensions for 2D thread block
  // bdimx covers the iteration dimension, bdimy covers the reduction dimension
  int64_t bdimx = 32;
  int64_t bdimy = 16;

  // TMA tile dimensions
  int64_t tma_tile_i = 128;
  int64_t tma_tile_r = 128;

  // Unroll factor for the iteration dimension (within TMA tile)
  int64_t iter_unroll_factor = 4;

  // Unroll factor for the reduction dimension (within TMA tile)
  int64_t redu_unroll_factor = 8;

  // Grid dimension for parallelizing the outer reduction across CTAs
  int64_t grdim = 1;
};

namespace reduction {
namespace outer_tma {
std::unique_ptr<TmaOuterReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const reduction_scheduler_utils::FusionRuntimeProperties& props);

void scheduleReduction(Fusion* fusion, const TmaOuterReductionParams* rparams);
} // namespace outer_tma
} // namespace reduction
} // namespace nvfuser
