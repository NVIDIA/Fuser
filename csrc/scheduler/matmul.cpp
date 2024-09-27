// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <abstract_tensor.h>
#include <device_lower/analysis/circular_buffer.h>
#include <inlining.h>
#include <instrumentation.h>
#include <multidevice/utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/matmul.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/mma_utils.h>
#include <scheduler/multi_matmul.h>
#include <scheduler/utils.h>

// NOTE: included to avoid compilation error caused by missing destructor in
// 'SchedulerRuntimeInfo'
#include <fusion_executor/executor_utils.h>
#include "mma_type.h"

namespace nvfuser {

void MatmulScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  FUSER_PERF_SCOPE("MatmulScheduler::schedule");
  auto mparams = dynamic_cast<const MatmulParams*>(params);
  NVF_ERROR(
      mparams != nullptr,
      "Incorrect parameters sent to MatmulScheduler::schedule",
      params);
  scheduleMultipleMatmuls(fusion, mparams);
}

bool MatmulScheduler::canScheduleCompileTime(Fusion* fusion) {
  const auto msg = getMatmulCompileTimeRejectReason(fusion);
  if (!msg.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(schedulerType(), msg);
    return false;
  }

  return true;
}

bool MatmulScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("MatmulScheduler::canSchedule");
  auto reason = getMatmulRunTimeRejectReason(fusion, data_cache, runtime_info);
  if (!reason.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(schedulerType(), reason);
    return false;
  }
  return true;
}

std::unique_ptr<HeuristicParams> MatmulScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  auto mparams = getMatmulHeuristics(fusion, runtime_info, data_cache);
  NVF_ERROR(mparams != nullptr);
  return mparams;
}

void moveInnerBroadcastLeft(TensorView* tv, int64_t number_of_inner_pos) {
  NVF_ERROR(tv->nDims() >= number_of_inner_pos);
  std::vector<int64_t> broadcast_pos;
  std::vector<int64_t> nonbroadcast_pos;

  for (auto i : c10::irange(number_of_inner_pos)) {
    auto axis_idx = i - number_of_inner_pos;
    auto id = tv->axis(axis_idx);
    if (id->isBroadcast()) {
      broadcast_pos.push_back(axis_idx);
    } else {
      nonbroadcast_pos.push_back(axis_idx);
    }
  }

  auto combined_pos_vec = broadcast_pos;
  combined_pos_vec.insert(
      combined_pos_vec.end(), nonbroadcast_pos.begin(), nonbroadcast_pos.end());

  std::unordered_map<int64_t, int64_t> order_map;
  for (auto i : c10::irange(number_of_inner_pos)) {
    order_map[combined_pos_vec.at(i)] = i - number_of_inner_pos;
  }

  // Apply ordering.
  tv->reorder(order_map);
}

void scheduleMatmul(Fusion* fusion, const MatmulParams* mparams) {
  scheduleMultipleMatmuls(fusion, mparams);
}

} // namespace nvfuser
