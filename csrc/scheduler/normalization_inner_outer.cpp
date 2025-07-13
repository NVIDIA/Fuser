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
#include <scheduler/utils.h>

#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {
namespace {

// Prioritize warp specialized approach if possible
bool preferWarpSpecialized(
    Fusion* fusion,
    int64_t total_iteration_numel,
    int64_t n_inner_reduction_tvs) {
  // Temporary disable warp specialized approach
  // It can only be involed by NVFUSER_ENABLE=WarpSpecializedNormalization
  return false;

  // False, for pre-Blackwell GPUs
  if (at::cuda::getCurrentDeviceProperties()->major < 10) {
    return false;
  }
  // False, if any of the inputs is dynamically shaped
  // TODO: extend to support dynamic inputs, warp specialization requires
  // static CTA size
  auto inp_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
  if (std::any_of(inp_tvs.begin(), inp_tvs.end(), [](TensorView* tv) {
        return scheduler_utils::isSymbolicTensor(tv);
      })) {
    return false;
  }

  // False, when iteration dimension is too small.
  // (1) Benefit from amortizing weight tensor loading overhead is minimal.
  // (2) Can't create deep circular buffering.
  // (3) Empirically determined thresholds on B200:
  // - RMS Norm Bwd: ≤16 rows per SM, Layer Norm Bwd: ≤4 rows per SM
  int64_t sm_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int64_t rows_per_sm = ceilDiv(total_iteration_numel, sm_count);
  bool is_layer_norm = n_inner_reduction_tvs > 1;
  int64_t prefered_rows_per_sm = is_layer_norm ? 16 : 4;
  if (rows_per_sm <= prefered_rows_per_sm) {
    return false;
  }
  // Try to use warp specialized, but if the heuristic fails, fall back to
  // multi-wave
  return true;
}

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
  int64_t max_outer_reduction_dtype_size_bit = 1;
  int64_t n_inner_reduction_tvs = 0;
  TensorView* first_inner_reduction_tv = nullptr;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      first_inner_reduction_tv = tv;
      n_inner_reduction_tvs++;
    } else {
      max_outer_reduction_dtype_size_bit = std::max(
          max_outer_reduction_dtype_size_bit,
          dataTypeSizeBit(tv->getDataType().value()));
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
                InnerOuterPersistentKernelScheduler::threads_per_block_max,
                /*is_warp_specialized=*/
                isOptionEnabled(EnableOption::WarpSpecializedNormalization));
          });
  scheduler_utils::SchedulerHyperParameters& hp =
      scheduler_hyperparameters_entry.get();

  auto& persistent_buffer_info = persistent_buffer_info_entry.get();
  NVF_ERROR(
      !persistent_buffer_info.persistent_buffers.empty(),
      "Persistent scheduler requires persistent buffers.");

  auto getBufferParams = [&](bool warp_specialized) {
    return inner_outer_utils::getPersistentBufferStorageParams(
        fusion,
        runtime_info,
        data_cache,
        reduction_tvs,
        hp.vectorize_factor,
        hp.threads_per_block_min,
        hp.threads_per_block_max,
        warp_specialized);
  };

  auto makeRParams = [&]() {
    return std::make_unique<ReductionParams>(
        InnerOuterPersistentKernelScheduler::schedulerType());
  };
  if (hp.is_warp_specialized ||
      preferWarpSpecialized(
          fusion, properties.total_iteration_numel, n_inner_reduction_tvs)) {
    auto buffer_params = getBufferParams(/*is_warp_specialized=*/true);
<<<<<<< HEAD
    // Current implementation assumes persistent buffers are projected to
    // inputs, making TMA loading beneficial. If not, shared memory persistent
    // buffers cannot use TMA since their producers are not inputs.
    if (buffer_params.project_to_input) {
      auto rparams = makeRParams();
      rparams->smem_persistent_buffers = buffer_params.smem_persistent_buffers;

      inner_outer_tma_warp_specialized::getHeuristics(
          rparams.get(),
          properties.total_iteration_numel,
          properties.total_reduction_numel,
          buffer_params.regs_buffer_size,
          buffer_params.circular_buffered_smem_size,
          buffer_params.non_circular_buffered_smem_size,
          max_outer_reduction_dtype_size,
          hp.vectorize_factor,
          hp.threads_per_block_min,
          hp.threads_per_block_max,
          buffer_params.project_to_input,
          runtime_info.getIndexType());

=======

    // Current implementation assumes persistent buffers are projected to
    // inputs, making TMA loading beneficial. If not, shared memory persistent
    // buffers cannot use TMA since their producers are not inputs.
    if (buffer_params.project_to_input) {
      auto rparams = makeRParams();
      rparams->smem_persistent_buffers = buffer_params.smem_persistent_buffers;

      inner_outer_tma_warp_specialized::getHeuristics(
          rparams.get(),
          properties.total_iteration_numel,
          properties.total_reduction_numel,
          buffer_params.regs_buffer_size_bit,
          buffer_params.circular_buffered_smem_size_bit,
          buffer_params.non_circular_buffered_smem_size_bit,
          max_outer_reduction_dtype_size_bit,
          hp.vectorize_factor,
          hp.threads_per_block_min,
          hp.threads_per_block_max,
          buffer_params.project_to_input,
          runtime_info.getIndexType());

>>>>>>> llu/ws_tma_vect_check
      // If warp specialized is enabled, or the heuristic is successful, return
      if (hp.is_warp_specialized || rparams->is_good_ws_heuristic) {
        return rparams;
      }
    }
  }

  // Fallback to multi-wave
  auto buffer_params = getBufferParams(/*is_warp_specialized=*/false);
  auto rparams = makeRParams();
  rparams->smem_persistent_buffers = buffer_params.smem_persistent_buffers;

  inner_outer_multi_wave::getHeuristics(
      rparams.get(),
      properties.total_iteration_numel,
      properties.total_reduction_numel,
      buffer_params.regs_buffer_size_bit,
      buffer_params.smem_buffer_size_bit,
      buffer_params.smem_overhead_bit,
      max_outer_reduction_dtype_size_bit,
      hp.vectorize_factor,
      hp.threads_per_block_min,
      hp.threads_per_block_max,
      buffer_params.project_to_input,
      runtime_info.getIndexType());

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
        "to use combined reduction, inner reduction tensor should be "
        "[I,I,...,R,R] and outer reduction tensor should be [R,R,...,I,I]");
    return false;
  }

  if (!normalization_scheduler_utils::hasSharedInput(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "to use combined reduction, inner reduction and outer reduction should "
        "have shared input.");
    return false;
  }

  if (!normalization_scheduler_utils::isConnectedOnlyThroughReductionProducer(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "to use combined reduction, inner reduction and outer reduction should "
        "not have shared consumer, their consumers should not have shared "
        "non-outer-reduction producer.");
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
        "to use combined reduction, every iteration axis in inner reduction tv "
        "should match to a reduction domain in outer reduction tv.");
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
                InnerOuterPersistentKernelScheduler::threads_per_block_max,
                /*is_warp_specialized=*/
                isOptionEnabled(EnableOption::WarpSpecializedNormalization));
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
          hp.threads_per_block_max,
          hp.is_warp_specialized);

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
      buffer_params.regs_buffer_size_bit,
      scheduler_utils::register_file_size_bit);

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
      "Incorrect parameters sent to "
      "InnerOuterPersistentKernelScheduler::schedule",
      params);
  if (rparams->tma_warp_specialized) {
    inner_outer_tma_warp_specialized::scheduleFusion(fusion, rparams);
  } else {
    inner_outer_multi_wave::scheduleFusion(fusion, rparams);
  }
}
} // namespace nvfuser
