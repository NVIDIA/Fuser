// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <instrumentation.h>
#include <options.h>
#include <scheduler/debug_utils.h>
#include <scheduler/normalization_inner_outer_multi_wave.h>
#include <scheduler/normalization_inner_outer_tma_ws.h>
#include <scheduler/normalization_inner_outer_utils.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>

#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {
namespace {
std::unique_ptr<ReductionParams> getInnerOuterPersistentHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);

  auto reduction_tv_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });

  auto& reduction_tvs = reduction_tv_entry.get();

  NVF_ERROR(!reduction_tvs.empty(), "Need reduction tensor views to schedule.");

  // Get dtype used to store partial outer reduction
  // Get the first inner reduction tv and use it as the reference tv
  int64_t max_outer_reduction_dtype_size = 1;
  TensorView* first_inner_reduction_tv = nullptr;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      first_inner_reduction_tv = tv;
    } else {
      max_outer_reduction_dtype_size = std::max(
          max_outer_reduction_dtype_size,
          dataTypeSize(tv->getDataType().value()));
    }
  }
  auto ref_red_tv = first_inner_reduction_tv;

  // Verify the presence of a reduction TensorView connected to a Fusion input
  normalization_scheduler_utils::checkReductionTvForScheduling(
      fusion, ref_red_tv);

  auto properties =
      scheduler_utils::getReductionProperties(fusion, runtime_info, ref_red_tv);
  auto reduced_tv = ir_utils::getSoleProducerTv(ref_red_tv);

  // Although properties contains runtime information
  // "inner_most_dimension_ndims" is a compile time value
  auto vec_break_point = HeuristicDataCacheEntry<
      HeuristicCompileTime::VectorizationBreakPointOfReductionProducer>(
      data_cache, [&ref_red_tv, &reduced_tv, &properties]() {
        return std::make_unique<int64_t>(
            vectorize_helper::getVectorizationBreakPointOfReductionProducer(
                ref_red_tv, reduced_tv, properties.inner_most_dimension_ndims));
      });

  const auto vectorize_factor = vectorize_helper::getVectorizationFactor(
      runtime_info, reduced_tv, data_cache, vec_break_point.get());

  auto persistent_buffer_info_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });

  auto scheduler_hyperparameters_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::SchedulerHyperParameters>(
          data_cache, [&]() {
            return std::make_unique<scheduler_utils::SchedulerHyperParameters>(
                /*vectorize_factor=*/vectorize_factor,
                /*unroll_factor=*/1,
                /*threads_per_block_min=*/
                InnerOuterPersistentKernelScheduler::threads_per_block_min,
                /*threads_per_block_max=*/
                InnerOuterPersistentKernelScheduler::threads_per_block_max);
          });
  scheduler_utils::SchedulerHyperParameters& hp =
      scheduler_hyperparameters_entry.get();

  auto& persistent_buffer_info = persistent_buffer_info_entry.get();
  NVF_ERROR(
      !persistent_buffer_info.persistent_buffers.empty(),
      "Persistent scheduler requires persistent buffers.");
  auto buffer_params = inner_outer_utils::getPersistentBufferStorageParams(
      fusion,
      runtime_info,
      data_cache,
      reduction_tvs,
      hp.vectorize_factor,
      hp.threads_per_block_min,
      hp.threads_per_block_max);

  auto rparams = std::make_unique<ReductionParams>(
      InnerOuterPersistentKernelScheduler::schedulerType());

  // save persistent tvs should use shared memory, to avoid calling
  // getPersistentBufferStorageParams again during the scheduling.
  rparams->smem_persistent_buffers = buffer_params.smem_persistent_buffers;

  // Ultimately, we want the heuristic to decide between using the
  // warp-specialized version or the multi-wave version. The enable option is a
  // temporary configuration to facilitate testing during development without
  // disrupting existing behavior.
  if (isOptionEnabled(EnableOption::WarpSpecializedNormalization)) {
    inner_outer_tma_warp_specialized::getHeuristics(
        rparams.get(),
        properties.total_iteration_numel,
        properties.total_reduction_numel,
        buffer_params.regs_buffer_size,
        buffer_params.smem_buffer_size,
        buffer_params.smem_overhead,
        max_outer_reduction_dtype_size,
        hp.vectorize_factor,
        hp.threads_per_block_min,
        hp.threads_per_block_max,
        buffer_params.project_to_input,
        runtime_info.getIndexType());
  } else {
    inner_outer_multi_wave::getHeuristics(
        rparams.get(),
        properties.total_iteration_numel,
        properties.total_reduction_numel,
        buffer_params.regs_buffer_size,
        buffer_params.smem_buffer_size,
        buffer_params.smem_overhead,
        max_outer_reduction_dtype_size,
        hp.vectorize_factor,
        hp.threads_per_block_min,
        hp.threads_per_block_max,
        buffer_params.project_to_input,
        runtime_info.getIndexType());
  }

  return rparams;
}

} // namespace

bool InnerOuterPersistentKernelScheduler::canScheduleCompileTime(
    Fusion* fusion) {
  FUSER_PERF_SCOPE(
      "InnerOuterPersistentKernelScheduler::canScheduleCompileTime");
  // common checks for all persistent heuristics
  if (!normalization_scheduler_utils::checkOpsAndInputs(
          fusion, schedulerType())) {
    return false;
  }

  // check reduction type
  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  if (reduction_tvs.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "no reduction tv");
    return false;
  }
  auto reduction_type =
      reduction_scheduler_utils::getReductionType(reduction_tvs);
  const SchedulerType persistent_heuristic =
      normalization_scheduler_utils::getPersistentHeuristicFor(reduction_type);
  if (persistent_heuristic != schedulerType()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "schedulerType() doesn't match with reduction type `",
        persistent_heuristic,
        "`.");
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

  // check connections between inner reduction and outer reduction tvs.
  if (!normalization_scheduler_utils::checkIfReductionsAreInnerOuter(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "to use combined reduction, inner reduction tensor should be [I,I,...,R,R] and outer reduction tensor should be [R,R,...,I,I]");
    return false;
  }

  if (!normalization_scheduler_utils::hasSharedInput(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "to use combined reduction, inner reduction and outer reduction should have shared input.");
    return false;
  }

  if (!normalization_scheduler_utils::isConnectedOnlyThroughReductionProducer(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "to use combined reduction, inner reduction and outer reduction should not have shared consumer, their consumers should not have shared non-outer-reduction producer.");
    return false;
  }

  if (!ir_utils::getViewOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "Fusion requires view being reversible.");
      return false;
    }
    // Persistent scheduler simply uses reference_tv as the reference, if
    // that changes, this needs to be changed.
    auto reference_tv = inner_reduction_tvs[0];
    if (registry_utils::reductionInterferingView(
            fusion, ca_map, reference_tv)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "View may interfere with normalization scheduling.");
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
    for (auto id : red_tv->getMaybeRootDomain()) {
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
            schedulerType(),
            "inconsistent reduction root size: ",
            red->toString(),
            ", expected: ",
            axis_count);
        return false;
      }
    }
  }

  // the reduction axis of outer reduction tv should match to the iteration axis
  // of the inner reduction tv.
  if (!normalization_scheduler_utils::isReductionIterationAxisMatched(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "to use combined reduction, every iteration axis in inner reduction tv should match to a reduction domain in outer reduction tv.");
    return false;
  }

  if (!normalization_scheduler_utils::checkReductionPattern(
          fusion, schedulerType(), inner_reduction_tvs, outer_reduction_tvs)) {
    return false;
  }

  // Only accept persistent kernels
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
  if (persistent_buffer_info.persistent_buffers.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "no persistent buffer identified");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasNonNormalizePostReductionBCast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "unsupported post reduction normalization");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasGatherToBroadcastBeforeReduction(fusion, reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "has unsupported gather-like ops before normalization");
    return false;
  }

  return true;
}

bool InnerOuterPersistentKernelScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("InnerOuterPersistentKernelScheduler::canScheduleRunTime");
  auto reduction_tv_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });

  auto& reduction_tvs = reduction_tv_entry.get();
  TensorView* first_inner_reduction_tv = nullptr;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      first_inner_reduction_tv = tv;
      break;
    }
  }
  auto reference_tv = first_inner_reduction_tv;

  auto properties = scheduler_utils::getReductionProperties(
      fusion, runtime_info, reference_tv);

  const int64_t warp_size = at::cuda::getCurrentDeviceProperties()->warpSize;

  auto reduced_tv = ir_utils::getSoleProducerTv(reference_tv);
  const auto vectorize_factor = vectorize_helper::getVectorizationFactor(
      runtime_info,
      reduced_tv,
      data_cache,
      (int)(reduced_tv->nDims() - properties.inner_most_dimension_ndims));

  auto scheduler_hyperparameters_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::SchedulerHyperParameters>(
          data_cache, [&]() {
            return std::make_unique<scheduler_utils::SchedulerHyperParameters>(
                /*vectorize_factor=*/vectorize_factor,
                /*unroll_factor=*/1,
                /*threads_per_block_min=*/
                InnerOuterPersistentKernelScheduler::threads_per_block_min,
                /*threads_per_block_max=*/
                InnerOuterPersistentKernelScheduler::threads_per_block_max);
          });
  scheduler_utils::SchedulerHyperParameters& hp =
      scheduler_hyperparameters_entry.get();

  // check if there is enough register and shared memory for persistence
  const auto buffer_params =
      inner_outer_utils::getPersistentBufferStorageParams(
          fusion,
          runtime_info,
          data_cache,
          reduction_tvs,
          hp.vectorize_factor,
          hp.threads_per_block_min,
          hp.threads_per_block_max);

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  if (!buffer_params.has_enough_regs_and_smem) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "not enough registers or shared memory for persistence");
    return false;
  }

  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor;

  const int64_t required_sm_per_norm = ceilDiv(
      buffer_params.regs_buffer_size, scheduler_utils::register_file_size);

  // If the persistence requires over half the device don't do grid
  // persistence as we can't overlap the grid comms.
  if (required_sm_per_norm >
      scheduler_utils::safeDiv(device_multiprocessor_count, 2)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "requires over half GPU persistence.");
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
        schedulerType(), "not enough blocks");
    return false;
  }

  return true;
}

std::unique_ptr<HeuristicParams> InnerOuterPersistentKernelScheduler::
    computeHeuristics(
        Fusion* fusion,
        SchedulerRuntimeInfo& runtime_info,
        HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("InnerOuterPersistentKernelScheduler::computeHeuristics");
  auto rparams =
      getInnerOuterPersistentHeuristics(fusion, runtime_info, data_cache);
  NVF_ERROR(rparams != nullptr);
  return rparams;
}

void InnerOuterPersistentKernelScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("InnerOuterPersistentKernelScheduler::schedule");
  auto rparams = dynamic_cast<const ReductionParams*>(params);
  NVF_ERROR(
      rparams != nullptr && rparams->scheduler_type == schedulerType(),
      "Incorrect parameters sent to InnerOuterPersistentKernelScheduler::schedule",
      params);
  if (rparams->tma_warp_specialized) {
    inner_outer_tma_warp_specialized::scheduleFusion(fusion, rparams);
  } else {
    inner_outer_multi_wave::scheduleFusion(fusion, rparams);
  }
}
} // namespace nvfuser
