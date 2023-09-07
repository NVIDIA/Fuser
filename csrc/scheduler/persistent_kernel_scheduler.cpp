// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/cuda/CUDAContext.h>
#include <instrumentation.h>
#include <scheduler/debug_utils.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/persistent_kernel_scheduler.h>
#include <scheduler/registry_utils.h>
#include <scheduler/utils.h>

namespace nvfuser {

PersistentKernelScheduler::PersistentKernelScheduler(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache)
    : SchedulerEntry(ScheduleHeuristic::Persistent) {
  computeHeuristics(fusion, runtime_info, data_cache);
}

void PersistentKernelScheduler::schedule(Fusion* fusion) {
  FUSER_PERF_SCOPE("Schedule Persistent Fusion");
  schedulePersistentKernel(fusion, reductionParams());
}

bool PersistentKernelScheduler::canScheduleCompileTime(Fusion* fusion) {
  // Needs at least one reduction to consider.
  auto reduction_ops = ir_utils::getReductionOps(fusion);
  if (reduction_ops.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent, "needs a reduction op");
    return false;
  }

  if (ir_utils::filterByType<TensorView>(fusion->inputs()).empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent,
        "Scheduling not supported with no input");
    return false;
  }

  // Check that inputs of all select/gather-like ops are fusion inputs
  if (registry_utils::rejectScheduleForMemoryPromotion(
          fusion, ScheduleHeuristic::Persistent)) {
    return false;
  }

  // Fusions handled by persistent kernel scheduler cannot have MmaOp.
  if (!ir_utils::getMmaOps(fusion).empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent, "no support for mma ops.");
    return false;
  }

  if (registry_utils::hasNonUniqueBcast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent,
        "Broadcasting dimension might be broadcasting to multiple sizes.");
    return false;
  }

  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);

  if (reduction_tvs.empty()) {
    // Use pointwise logic
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent, "no reduction tv");
    return false;
  }

  std::vector<TensorView*> inner_reduction_tvs;
  std::vector<TensorView*> outer_reduction_tvs;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      inner_reduction_tvs.emplace_back(tv);
    } else {
      outer_reduction_tvs.emplace_back(tv);
    }
  }
  bool combined_inner_outer =
      !inner_reduction_tvs.empty() && !outer_reduction_tvs.empty();
  if (!checkReductionPattern(
          fusion, inner_reduction_tvs, outer_reduction_tvs)) {
    return false;
  }
  // If there is both inner and outer reduction, we use the first inner
  // reduction tv as reference, otherwise we use the first reduction tv,
  // whether it is inner or outer.
  TensorView* reference_tv =
      combined_inner_outer ? inner_reduction_tvs[0] : reduction_tvs[0];

  if (!ir_utils::getViewOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "Fusion requires view being reversible.");
      return false;
    }

    // Persistent scheduler simply uses reference_tv as the reference, if
    // that changes, this needs to be changed.
    if (registry_utils::reductionInterferingView(
            fusion, ca_map, reference_tv)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "View may interfere with normalization scheduling.");
      return false;
    }
  }

  // Before examining the reduction axes want to quickly
  //   check the reductions have the same axis width
  //   to avoid building root domain map in easier cases
  bool valid_axis_count = false;
  size_t axis_count = 0;
  auto reduction_root_size = [](TensorView* red_tv) {
    size_t count = 0;
    for (auto id : red_tv->getRootDomain()) {
      if (!id->isBroadcast()) {
        count++;
      }
    }
    return count;
  };

  for (auto red : reduction_tvs) {
    if (!valid_axis_count) {
      valid_axis_count = true;
      axis_count = reduction_root_size(red);
    } else {
      if (reduction_root_size(red) != axis_count) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Persistent,
            "inconsistent reduction root size: ",
            red->toString(),
            ", expected: ",
            axis_count);
        return false;
      }
    }
  }

  // Only accept persistent kernels
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
  if (persistent_buffer_info.persistent_buffers.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent, "no persistent buffer identified");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasNonNormalizePostReductionBCast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent,
        "unsupported post reduction normalization");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasGatherToBroadcastBeforeReduction(fusion, reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent,
        "has unsupported gather-like ops before normalization");
    return false;
  }

  return true;
}

bool PersistentKernelScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("PersistentKernelScheduler::canSchedule");
  auto reduction_tv_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });

  auto& reduction_tvs = reduction_tv_entry.get();
  bool inner_reduction = false;
  bool outer_reduction = false;
  TensorView* first_inner_reduction_tv = nullptr;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      first_inner_reduction_tv = tv;
      inner_reduction = true;
    } else {
      outer_reduction = true;
    }
  }

  // If there is both inner and outer reduction, we use the first inner
  // reduction tv to get properties, otherwise we use the first reduction tv,
  // whether it is inner or outer.
  auto reference_tv = inner_reduction && outer_reduction
      ? first_inner_reduction_tv
      : reduction_tvs[0];

  auto properties = scheduler_utils::getReductionProperties(
      fusion, runtime_info, reference_tv);

  const int64_t warp_size = at::cuda::getCurrentDeviceProperties()->warpSize;

  if (!properties.fastest_dim_reduction) {
    return canScheduleRunTimeOuter(
        fusion, runtime_info, data_cache, reduction_tvs, properties);
  }

  // pair of persistent_buffer_size and available_persistent_buffer_size
  const std::pair<int64_t, int64_t> buffer_size =
      getPersistentBufferSize(fusion, runtime_info, data_cache, reduction_tvs);
  const int64_t persistent_buffer_size = buffer_size.first;
  const int64_t available_persistent_buffer_size = buffer_size.second;

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  if (persistent_buffer_size > available_persistent_buffer_size) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent,
        "not enough registers or shared memory for persistence");
    return false;
  }

  if (inner_reduction && outer_reduction) {
    // get vectorize_factor, same process to that in getPersistentHeuristics
    auto reduced_tv = ir_utils::getSoleProducerTv(reference_tv);
    const auto vectorize_factor = vectorize_helper::getVectorizationFactor(
        runtime_info,
        reduced_tv,
        data_cache,
        (int)(reduced_tv->nDims() - properties.inner_most_dimension_ndims));
    // check if we can schedule the combined reductions with a reasonable
    // batch size without register spills.
    if (!normalization_scheduler_utils::
             getOptionalInnerOuterPersistentBufferBatches(
                 properties.total_reduction_numel,
                 properties.total_iteration_numel,
                 persistent_buffer_size,
                 (int64_t)vectorize_factor,
                 warp_size,
                 false)
                 .first.has_value()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "Required batch number is larger than available batch number! Will cause register spills!");
      return false;
    }
  }

  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor;

  // Maximum number of iteration dimensions we can have and still be
  // persistent.
  const int64_t max_multi_reduction_factor = scheduler_utils::safeDiv(
      available_persistent_buffer_size, persistent_buffer_size);

  const int64_t required_sm_per_norm =
      ceilDiv(persistent_buffer_size, scheduler_utils::register_file_size);

  // If the persistence requires over half the device don't do grid
  // persistence as we can't overlap the grid comms.
  if (required_sm_per_norm >
      scheduler_utils::safeDiv(device_multiprocessor_count, 3)) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent, "requires over half GPU persistence.");
    return false;
  }

  const int64_t norm_per_sm =
      ceilDiv(scheduler_utils::register_file_size, persistent_buffer_size);

  // If outer reduction, don't go persistent if we can't fit half a warp in
  // the iter domain of the persistent reduction.
  if (!properties.fastest_dim_reduction &&
      !(norm_per_sm >= warp_size / 2 ||
        max_multi_reduction_factor >= warp_size)) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent, "not enough threads");
    return false;
  }

  // Don't go persistent if we can't use a small fraction of the
  // available SMs yet have a large reduction size.
  if ( // Large reduction dim
      properties.total_reduction_numel >=
          device_max_threads_per_multiprocessor * 4 &&
      properties.total_iteration_numel <
          (properties.fastest_dim_reduction
               ? scheduler_utils::safeDiv(device_multiprocessor_count, 8)
               // Make sure we at least use a quarter of the device * a
               // half warp
               : (warp_size / 8) * device_multiprocessor_count)) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent, "not enough blocks");
    return false;
  }

  return true;
}

void PersistentKernelScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  params_ = getPersistentHeuristics(fusion, runtime_info, data_cache);
  NVF_ERROR(params_ != nullptr);
}

bool PersistentKernelScheduler::checkReductionPattern(
    Fusion* fusion,
    const std::vector<TensorView*>& inner_reduction_tvs,
    const std::vector<TensorView*>& outer_reduction_tvs) {
  // Use root domain map to check the reduction ops have the same axes
  FusionGuard fg(fusion);
  ComputeAtRootDomainMap root_map;
  root_map.build(true);

  // check inner and outer reductions seperately
  for (const auto& rtvs : {inner_reduction_tvs, outer_reduction_tvs}) {
    for (const auto it : c10::irange(1, rtvs.size())) {
      if (!registry_utils::checkPatternEquivalence(
              rtvs[it - 1], rtvs[it], root_map)) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Persistent,
            "unmapped reduction ",
            rtvs[it - 1],
            " and ",
            rtvs[it]);
        return false;
      }
    }
  }
  // combined inner and outer reduction is of general purpose but only tested
  // for layer norm backward
  if (!inner_reduction_tvs.empty() && !outer_reduction_tvs.empty()) {
    if (!normalization_scheduler_utils::checkIfReductionsAreInnerOuter(
            inner_reduction_tvs, outer_reduction_tvs)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "to use combined reduction, inner reduction tensor should be [I,I,...,R,R] and outer reduction tensor should be [R,R,...,I,I]");
      return false;
    }

    if (!normalization_scheduler_utils::hasSharedInput(
            inner_reduction_tvs, outer_reduction_tvs)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "to use combined reduction, inner reduction and outer reduction should have shared input.");
      return false;
    }

    if (!normalization_scheduler_utils::isConnectedOnlyThroughReductionProducer(
            inner_reduction_tvs, outer_reduction_tvs)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "to use combined reduction, inner reduction and outer reduction should not have shared consumer, their consumers should not have shared non-outer-reduction producer.");
      return false;
    }
  }
  return true;
}

std::pair<int64_t, int64_t> PersistentKernelScheduler::getPersistentBufferSize(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache,
    const std::vector<TensorView*>& reduction_tvs) {
  auto persistent_buffer_info_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });

  auto& persistent_buffer_info = persistent_buffer_info_entry.get();

  auto persistent_buffer_size_info = scheduler_utils::persistentBufferSize(
      fusion, runtime_info, persistent_buffer_info, data_cache);

  // Note that projected buffer size can be zero
  auto persistent_buffer_size =
      persistent_buffer_size_info.projected_persistent_buffer_size == 0
      ? persistent_buffer_size_info.persistent_buffer_size
      : std::min(
            persistent_buffer_size_info.persistent_buffer_size,
            persistent_buffer_size_info.projected_persistent_buffer_size);

  // in combined_inner_outer_reduction, the partial results of outer
  // reductions must be persistent, allow register spill avoid segmentation
  int64_t inner_reduction_count = 0;
  int64_t outer_reduction_count = 0;
  std::vector<TensorView*> outer_reduction_tvs;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      inner_reduction_count++;
    } else {
      outer_reduction_count++;
      outer_reduction_tvs.emplace_back(tv);
    }
  }
  const bool combined_inner_outer_reduction =
      inner_reduction_count && outer_reduction_count;
  if (combined_inner_outer_reduction) {
    persistent_buffer_size +=
        normalization_scheduler_utils::partialReductionBufferSize(
            outer_reduction_tvs, runtime_info);
  }

  // At this point, we use the full register file size only for the
  // inner-outer case. It does not mean the full size shouldn't be used
  // otherwise, but more detailed tuning of the heuristics would be required.
  int64_t available_persistent_buffer_size = combined_inner_outer_reduction
      ? scheduler_utils::register_file_size_full
      : scheduler_utils::register_file_size;

  // Use shared memory for persistent buffer is only tested for inner
  // reduction
  // TODO: extend to outer reduction and combined reduction
  const bool allow_shared_memory =
      inner_reduction_count > 0 && outer_reduction_count == 0;
  if (allow_shared_memory) {
    const auto dev_prop = at::cuda::getCurrentDeviceProperties();
    const int64_t max_shared_memory_size =
        (int64_t)dev_prop->sharedMemPerBlockOptin;
    // Some shared memories are reserved for kernel launch overhead and
    // reduction_broadcast_workspace. Estimation is conservative, but should
    // be good enough. The actual threads per block is set in the heuristics
    // and it may be smaller than maxThreadsPerBlock.
    // TODO: More accurate estimation of available shared memory size.
    const int64_t kernel_overhead =
        (int64_t)dev_prop->reservedSharedMemPerBlock;
    int64_t max_buffer_dtype_size = 1;
    for (auto tv : persistent_buffer_info.persistent_buffers) {
      max_buffer_dtype_size = std::max(
          max_buffer_dtype_size,
          dataTypeSize(tv->getDataType().value(), runtime_info.getIndexType()));
    }
    const int64_t reduction_broadcast_workspace =
        (int64_t)(dev_prop->maxThreadsPerBlock) * max_buffer_dtype_size;
    const int64_t available_shared_memory_size = max_shared_memory_size -
        kernel_overhead - reduction_broadcast_workspace;
    available_persistent_buffer_size = std::max(
        available_persistent_buffer_size, available_shared_memory_size);
  }

  return std::make_pair(
      persistent_buffer_size, available_persistent_buffer_size);
}

bool PersistentKernelScheduler::canScheduleRunTimeOuter(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache,
    const std::vector<TensorView*>& reduction_tvs,
    const scheduler_utils::ReductionTvProperties& properties) {
  FUSER_PERF_SCOPE("PersistentKernelScheduler::canScheduleRuntimeOuter");
  FusionGuard fg(fusion);

  const auto device_prop = at::cuda::getCurrentDeviceProperties();

  const int64_t sm_register_file_size =
      static_cast<int64_t>(device_prop->regsPerBlock * sizeof(int));

  auto persistent_buffer_info_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });

  const auto& persistent_buffer_info = persistent_buffer_info_entry.get();

  auto persistent_buffer_size_info = scheduler_utils::persistentBufferSize(
      fusion, runtime_info, persistent_buffer_info, data_cache);

  // Note that projected buffer size can be zero
  auto persistent_buffer_size =
      persistent_buffer_size_info.projected_persistent_buffer_size == 0
      ? persistent_buffer_size_info.persistent_buffer_size
      : std::min(
            persistent_buffer_size_info.persistent_buffer_size,
            persistent_buffer_size_info.projected_persistent_buffer_size);

  const int64_t device_multiprocessor_count =
      (int64_t)device_prop->multiProcessorCount;

  const auto available_persistent_buffer_size =
      sm_register_file_size * device_multiprocessor_count;

  if (persistent_buffer_size > available_persistent_buffer_size) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent, "not enough registers for persistence");
    return false;
  }

  auto reduced_tv = ir_utils::getSoleProducerTv(reduction_tvs.at(0));

  const int64_t vectorization_factor =
      (int64_t)vectorize_helper::getVectorizationFactor(
          runtime_info,
          reduced_tv,
          data_cache,
          (int)reduced_tv->nDims() -
              (int)properties.inner_most_dimension_ndims);

  // Minimum required multi reduction factor.
  const int64_t min_multi_reduction_factor = vectorization_factor *
      normalization_scheduler_utils::PreferredLaunchConfig::kMinBdimx;

  const int64_t required_sm_per_norm = ceilDiv(
      persistent_buffer_size * min_multi_reduction_factor,
      sm_register_file_size);

  // If the persistence requires over half the device don't do grid
  // persistence as we can't overlap the grid comms.
  if (required_sm_per_norm >
      scheduler_utils::safeDiv(device_multiprocessor_count, 2)) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent,
        "requires over half GPU persistence.",
        " required SMs per normalization: ",
        required_sm_per_norm);
    return false;
  }

  const bool is_cross_grid = required_sm_per_norm > 1;

  std::optional<normalization_scheduler_utils::GridOuterNormalizationParams>
      cross_grid_params;

  if (is_cross_grid) {
    // Don't try to be persistent unless at least 4-way vectorized
    // as register usage is hard to control
    // TODO: Is this necessary for block persistence as well?
    if (vectorization_factor < 4) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "not enough vectorized");
      return false;
    }

    // Make sure there's a valid grid persistence launch config
    cross_grid_params =
        normalization_scheduler_utils::getGridOuterNormalizationParams(
            properties.total_reduction_numel,
            properties.total_iteration_numel,
            vectorization_factor,
            persistent_buffer_size);

    if (!cross_grid_params.has_value()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "no valid launch config found");
      return false;
    }
  }

  NVF_ERROR(!is_cross_grid || cross_grid_params.has_value())

  // Maximum number of iteration dimensions we can have and still be
  // persistent.
  const int64_t max_multi_reduction_factor = scheduler_utils::safeDiv(
      is_cross_grid ? available_persistent_buffer_size : sm_register_file_size,
      persistent_buffer_size);

  // Don't go persistent if we can't fit the minimum multi reduction
  // factor
  if (max_multi_reduction_factor < min_multi_reduction_factor) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent,
        "Not enough threads.",
        " Multi reduction factor, ",
        max_multi_reduction_factor,
        ", is smaller than minimum multi reduction factor, ",
        min_multi_reduction_factor);
    return false;
  }

  const int64_t max_used_sms = is_cross_grid
      ? ceilDiv(
            ceilDiv(properties.total_iteration_numel, vectorization_factor),
            cross_grid_params->launch_params.bdimx()) *
          cross_grid_params->launch_params.gdimy()
      : ceilDiv(
            properties.total_iteration_numel * persistent_buffer_size,
            sm_register_file_size);

  // Bandwidth suffers if the number of used SMs is small. This is
  // particularly impactful in the case of cross grid, so at least
  // half of the SMs are required to be used. In the case of cross
  // block, keep using the existing heuristics for now.
  if (is_cross_grid &&
      max_used_sms < scheduler_utils::safeDiv(device_multiprocessor_count, 2)) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent,
        "cross grid - not enough used SMs: ",
        max_used_sms);
    return false;
  }

  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)device_prop->maxThreadsPerMultiProcessor;
  const int64_t min_fraction_of_sms =
      scheduler_utils::safeDiv(device_multiprocessor_count, 8);
  if (properties.total_reduction_numel >=
          device_max_threads_per_multiprocessor * 4 && // Large reduction dim
      max_used_sms < min_fraction_of_sms) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent, "not enough used SMs");
    return false;
  }

  // The runtime kernel for grouped normal grid reductions is not
  // well tuned, and it turned out to be quite difficult to get
  // consistently better performances than non-persistent
  // schedules. Disabled for now.
  // TODO: Enable non-welford persistent reductions
  if (is_cross_grid &&
      std::any_of(
          reduction_tvs.begin(),
          reduction_tvs.end(),
          [](TensorView* reduction_tv) {
            return !reduction_tv->definition()->isA<WelfordOp>();
          })) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent, "non-Welford not enabled yet");
    return false;
  }

  // Had a hard time tuning on Titan RTX and V100 when the iteration
  // space is not evenly divided by threads and thread blocks. It
  // doesn't seem to be noticeably bad on A100, though. For now,
  // disable the schedule if not evenly divisible on Titan RTX and
  // V100, i.e., compute architecture version 7.
  // TODO: Revisit
  if (is_cross_grid &&
      (properties.total_iteration_numel %
           (vectorization_factor * cross_grid_params->launch_params.bdimx() *
            cross_grid_params->launch_params.gdimx()) !=
       0) &&
      device_prop->major == 7) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Persistent, "iteration not evenly divided");
    return false;
  }

  return true;
}

} // namespace nvfuser
