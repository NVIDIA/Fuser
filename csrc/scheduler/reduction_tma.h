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

namespace nvfuser {

class TmaInnerReductionParams : public HeuristicParams {
 public:
  TmaInnerReductionParams(
      SchedulerType scheduler_type = SchedulerType::Reduction)
      : HeuristicParams(scheduler_type) {};

  // Unrolling/Vectorization factor
  int64_t unroll_factor = 1;

  int64_t target_threads_per_block;
};

namespace reduction {
namespace tma {
std::unique_ptr<TmaInnerReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache);

void scheduleReduction(Fusion* fusion, const TmaInnerReductionParams* pparams);
} // namespace tma
} // namespace reduction
} // namespace nvfuser
