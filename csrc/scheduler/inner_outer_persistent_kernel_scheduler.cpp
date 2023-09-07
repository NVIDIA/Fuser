// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/persistent_scheduler_helper.h>
#include <scheduler/inner_outer_persistent_kernel_scheduler.h>
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

InnerOuterPersistentKernelScheduler::InnerOuterPersistentKernelScheduler(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache)
    : SchedulerEntry(ScheduleHeuristic::InnerOuterPersistent) {
  computeHeuristics(fusion, runtime_info, data_cache);
}

void InnerOuterPersistentKernelScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  params_ = getInnerOuterPersistentHeuristics(fusion, runtime_info, data_cache);
  NVF_ERROR(params_ != nullptr);
}

void InnerOuterPersistentKernelScheduler::schedule(Fusion* fusion) {
  FUSER_PERF_SCOPE("Schedule InnerOuterPersistent Fusion");
  scheduleInnerOuterPersistentKernel(fusion, reductionParams());
}



bool InnerOuterPersistentKernelScheduler::canScheduleCompileTime(
    Fusion* fusion) {
  auto heuristic = ScheduleHeuristic::InnerOuterPersistent;

  // (1) leading common checks for all persistent kernels.
  if (!leadingCommonCompileTimeCheck(
          fusion, heuristic)) {
    return false;
  }

  // (2) check reduction type.
  const auto& reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  if (!checkReductionType(
          reduction_tvs, heuristic)) {
    return false;
  }

  // (3) special check for InnerOuter persistent kernel.
  std::vector<TensorView*> inner_reduction_tvs;
  std::vector<TensorView*> outer_reduction_tvs;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      inner_reduction_tvs.emplace_back(tv);
    } else {
      outer_reduction_tvs.emplace_back(tv);
    }
  }
  compileTimeCheckReductionAxis(
      fusion, inner_reduction_tvs, heuristic);
  compileTimeCheckReductionAxis(
      fusion, outer_reduction_tvs, heuristic);
  if (!normalization_scheduler_utils::checkIfReductionsAreInnerOuter(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic,
        "to use combined reduction, inner reduction tensor should be [I,I,...,R,R] and outer reduction tensor should be [R,R,...,I,I]");
    return false;
  }

  if (!normalization_scheduler_utils::hasSharedInput(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic,
        "to use combined reduction, inner reduction and outer reduction should have shared input.");
    return false;
  }

  if (!normalization_scheduler_utils::isConnectedOnlyThroughReductionProducer(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic,
        "to use combined reduction, inner reduction and outer reduction should not have shared consumer, their consumers should not have shared non-outer-reduction producer.");
    return false;
  }

  // (4) tailing common checks for all persistent kernels.
  if (!tailingCommonCompileTimeCheck(
          fusion, reduction_tvs, heuristic)) {
    return false;
  }

  return true;
}

bool InnerOuterPersistentKernelScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("InnerOuterPersistentKernelScheduler::canSchedule");

  // (1) check if there is enough shared memory or registers for persistent.
  auto reduction_tv_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });
  auto& reduction_tvs = reduction_tv_entry.get();

  auto persistent_buffer_info_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });
  auto& persistent_buffer_info = persistent_buffer_info_entry.get();

  const int64_t persistent_buffer_size =
      normalization_scheduler_utils::getPersistentBufferSize(
          fusion, runtime_info, data_cache, persistent_buffer_info) +
      normalization_scheduler_utils::partialReductionBufferSize(
          reduction_tvs, runtime_info);

  const int64_t available_shared_memory_size =
      normalization_scheduler_utils::getAvailableSmemSize(
          runtime_info, persistent_buffer_info.persistent_buffers);

  const int64_t available_persistent_buffer_size = std::max(
      scheduler_utils::register_file_size_full, available_shared_memory_size);

  if (persistent_buffer_size > available_persistent_buffer_size) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::InnerOuterPersistent,
        "not enough registers or shared memory for persistence");
    return false;
  }

  // (2) check if we can schedule the combined reductions with a reasonable
  // batch size without register spills.
  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  TensorView* reference_tv = nullptr;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      reference_tv = tv;
      break;
    }
  }
  NVF_ERROR(
      reference_tv,
      "reference_tv is nullptr in PersistentKernelScheduler::canScheduleRunTimeInnerOuter");
  auto properties = scheduler_utils::getReductionProperties(
      fusion, runtime_info, reference_tv);
  auto reduced_tv = ir_utils::getSoleProducerTv(reference_tv);
  const auto vectorize_factor = vectorize_helper::getVectorizationFactor(
      runtime_info,
      reduced_tv,
      data_cache,
      (int)(reduced_tv->nDims() - properties.inner_most_dimension_ndims));
  if (!normalization_scheduler_utils::
           getOptionalInnerOuterPersistentBufferBatches(
               properties.total_reduction_numel,
               properties.total_iteration_numel,
               persistent_buffer_size,
               (int64_t)vectorize_factor,
               (int64_t)device_prop->warpSize,
               false)
               .first.has_value()) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::InnerOuterPersistent,
        "Required batch number is larger than available batch number! Will cause register spills!");
    return false;
  }

  // (3) check iteration size
  // TODO: Needs check whether we need this check for innerOuter scheduler or
  // not.
  if (!runTimeCheckIterSize(
          properties, ScheduleHeuristic::InnerOuterPersistent)) {
    return false;
  }

  return true;
}

} // namespace nvfuser
