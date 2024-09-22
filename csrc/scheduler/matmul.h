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

NVF_API void scheduleMatmul(Fusion* fusion, const MatmulParams* mparams);

class MatmulScheduler : public SchedulerEntry {
 public:
  void schedule(Fusion* fusion, const HeuristicParams* params) override;

  bool canScheduleCompileTime(Fusion* fusion) override;

  bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache = nullptr) override;

  constexpr static SchedulerType schedulerType() {
    return SchedulerType::Matmul;
  }

  std::unique_ptr<HeuristicParams> computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache = nullptr) override;
};

} // namespace nvfuser
