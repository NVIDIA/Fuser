// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/inner_persistent_kernel_scheduler.h>
#include <scheduler/persistent_scheduler_helper.h>
#include <scheduler/registry_utils.h>

#include <c10/util/irange.h>
#include <disjoint_set.h>
#include <executor_utils.h>
#include <expr_evaluator.h>
#include <instrumentation.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <root_domain_map.h>
#include <scheduler/debug_utils.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/pointwise.h>
#include <scheduler/transpose.h>
#include <scheduler/utils.h>
#include <tensor_metadata.h>

#include <limits>

#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {

InnerPersistentKernelScheduler::InnerPersistentKernelScheduler(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache)
    : SchedulerEntry(ScheduleHeuristic::InnerPersistent) {
  computeHeuristics(fusion, runtime_info, data_cache);
}

void InnerPersistentKernelScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  params_ = getInnerPersistentHeuristics(fusion, runtime_info, data_cache);
  NVF_ERROR(params_ != nullptr);
}

void InnerPersistentKernelScheduler::schedule(Fusion* fusion) {
  FUSER_PERF_SCOPE("Schedule InnerPersistent Fusion");
  scheduleInnerPersistentKernel(fusion, reductionParams());
}

bool InnerPersistentKernelScheduler::canScheduleCompileTime(Fusion* fusion) {
  return commonCompileTimeCheck(fusion, ScheduleHeuristic::InnerPersistent);
}

bool InnerPersistentKernelScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("InnerPersistentKernelScheduler::canSchedule");

  // (1) check persistent buffer size, ensure we can do persistent.
  auto persistent_buffer_info_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });
  auto& persistent_buffer_info = persistent_buffer_info_entry.get();
  const int64_t persistent_buffer_size =
      normalization_scheduler_utils::getPersistentBufferSize(
          fusion, runtime_info, data_cache, persistent_buffer_info);

  const int64_t available_shared_memory_size =
      normalization_scheduler_utils::getAvailableSmemSize(
          runtime_info, persistent_buffer_info.persistent_buffers);

  const int64_t available_persistent_buffer_size = std::max(
      scheduler_utils::register_file_size, available_shared_memory_size);

  if (persistent_buffer_size > available_persistent_buffer_size) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::InnerPersistent,
        "not enough registers or shared memory for persistence");
    return false;
  }

  // (2) check iteration size, ensure we can do persistent efficiently.
  auto reduction_tv_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });
  auto& reduction_tvs = reduction_tv_entry.get();
  auto properties = scheduler_utils::getReductionProperties(
      fusion, runtime_info, reduction_tvs[0]);

  if (!PersistentSchedulerHelper::runTimeCheckIterSize(
          properties, ScheduleHeuristic::InnerPersistent)) {
    return false;
  }

  return true;
}

} // namespace nvfuser
