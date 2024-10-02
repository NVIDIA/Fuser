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
#include <runtime/executor_utils.h>
#include "mma_type.h"

namespace nvfuser {

bool MatmulScheduler::canScheduleCompileTime(Fusion* fusion) {
  const auto msg = matmul_utils::getMatmulCompileTimeRejectReason(fusion);
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
  auto reason = matmul_utils::getMatmulRunTimeRejectReason(
      fusion, data_cache, runtime_info);
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
  auto mparams =
      matmul_utils::getMatmulHeuristics(fusion, runtime_info, data_cache);
  NVF_ERROR(mparams != nullptr);
  return mparams;
}

void MatmulScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  FUSER_PERF_SCOPE("MatmulScheduler::schedule");
  auto mparams = dynamic_cast<const MatmulParams*>(params);
  NVF_ERROR(
      mparams != nullptr,
      "Incorrect parameters sent to MatmulScheduler::schedule",
      params);
  scheduleMultipleMatmuls(fusion, mparams);
}

} // namespace nvfuser
