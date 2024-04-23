// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/ivalue.h>
#include <exceptions.h>
#include <fusion.h>
#include <mma_type.h>
#include <scheduler/matmul_heuristic.h>
#include <scheduler/registry.h>
#include <visibility.h>

namespace nvfuser {

// Move the broadcast axes to the left on the specified number of inner
// dimensions e.g.  (when number_of_inner_pos == 3):
//      [... I0, B, I1] -> [... B, I0, I1]
//  should probably be only used to order innermost mnk axes.
void moveInnerBroadcastLeft(TensorView* tv, int64_t number_of_inner_pos = 3);

NVF_API void scheduleMatmul(Fusion* fusion, const MatmulParams& params);

class MatmulScheduler : public SchedulerEntry {
 public:
  explicit MatmulScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  void schedule(Fusion* fusion) override;

  static bool canScheduleCompileTime(Fusion* fusion);

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);
  constexpr static ScheduleHeuristic heuristicType() {
    return ScheduleHeuristic::Matmul;
  }

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);
};

} // namespace nvfuser
