// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <scheduler/reduction.h>
#include <scheduler/reduction_utils.h>

namespace nvfuser {

class TmaInnerReductionParams : public HeuristicParams {
 public:
  TmaInnerReductionParams(
      SchedulerType scheduler_type = SchedulerType::Reduction)
      : HeuristicParams(scheduler_type) {};

  // Inner serial split factor (similar to vectorization for non-TMA)
  int64_t vectorization_factor = 1;

  // Number of threads per block for TIDx parallelization
  int64_t threads_per_block = 1;

  // Unroll factor on top of TIDx split
  int64_t unroll_factor = 1;

  // Grid reduction: when reduction dim exceeds smem capacity, split and
  // parallelize outer reduction loop over BIDx, shifting iteration to BIDy
  bool grid_reduction = false;

  // Split size for grid reduction (fits into smem)
  int64_t grid_reduction_split_size = 0;
};

namespace reduction {
namespace tma {
std::unique_ptr<TmaInnerReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const reduction_scheduler_utils::FusionRuntimeProperties& props);

void scheduleReduction(Fusion* fusion, const TmaInnerReductionParams* rparams);
} // namespace tma
} // namespace reduction
} // namespace nvfuser
