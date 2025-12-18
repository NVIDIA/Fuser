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

  int64_t inner_unroll = 1;

  int64_t threads_per_block;
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
