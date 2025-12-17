// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <instrumentation.h>
#include <scheduler/debug_utils.h>
#include <scheduler/normalization_inner.h>
#include <scheduler/normalization_inner_non_tma.h>
#include <scheduler/normalization_inner_tma.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>
#include <scheduler/utils.h>

#include <ATen/cuda/CUDAContext.h>

#include <memory>

namespace nvfuser {
using PersistentKernelProperties =
    normalization_scheduler_utils::PersistentKernelProperties;

namespace {

bool mayUseTma(Fusion* fusion, const PersistentKernelProperties& prop) {
  // Hardware requirement: TMA is only available on Hopper (SM 9.0) and later
  if (at::cuda::getCurrentDeviceProperties()->major < 9) {
    return false;
  }

  // TMA requires 16-byte alignment (128 bits) for memory transactions
  if (prop.vectorize_factor * prop.max_dtype_size_bit % 128 != 0) {
    return false;
  }

  // Fall back to non-TMA version when persistent buffer size exceeds register
  // file capacity, as cluster reduction (using shared memory) is not yet
  // supported in the TMA version
  if (prop.max_persistent_buffer_size_bit >
      scheduler_utils::register_file_size_bit) {
    return false;
  }

  // TMA scheduler requires at least 128 threads (4 warps) after vectorization
  // to ensure sufficient parallelism
  // TODO: Refine this heuristic based on actual performance measurements, as
  // small reduction sizes may not benefit from TMA overhead
  if (prop.total_reduction_numel / prop.vectorize_factor < 128) {
    return false;
  }

  return true;
}

} // namespace

bool InnerPersistentKernelScheduler::canScheduleCompileTime(Fusion* fusion) {
  FUSER_PERF_SCOPE("InnerPersistentKernelScheduler::canScheduleCompileTime");
  return normalization_scheduler_utils::compileTimeCheck(
      fusion, schedulerType());
}

bool InnerPersistentKernelScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("InnerPersistentKernelScheduler::canScheduleRunTime");
  auto reduction_tv_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });

  auto& reduction_tvs = reduction_tv_entry.get();

  auto reference_tv = reduction_tvs[0];

  auto properties = scheduler_utils::getReductionProperties(
      fusion, runtime_info, reference_tv);

  const int64_t warp_size = at::cuda::getCurrentDeviceProperties()->warpSize;

  // check reduction properties, don't use shared memory persistent if 3D
  // reduction or device supports cluster reduction. Test of cross entropy loss
  // shows using cluster reduction with register persistent is faster than block
  // reduction using shared memory persistent.
  bool can_use_smem_persistent = (properties.total_reduction_numel ==
                                  properties.inner_most_dimension_numel) &&
      at::cuda::getCurrentDeviceProperties()->major < 9;
  bool is_3d_reduction =
      properties.total_reduction_numel != properties.inner_most_dimension_numel;

  // Get persistent buffer size - delegate to non_tma implementation for now
  auto persistent_buffer_info_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });

  auto& persistent_buffer_info = persistent_buffer_info_entry.get();

  auto persistent_buffer_size_info = scheduler_utils::persistentBufferSizeBit(
      fusion, runtime_info, persistent_buffer_info, data_cache);

  normalization_scheduler_utils::BufferProjectionStrategy project_strategy =
      normalization_scheduler_utils::isProjectBufferToInputs(
          fusion,
          runtime_info,
          reduction_tvs,
          persistent_buffer_info,
          persistent_buffer_size_info,
          InnerPersistentKernelScheduler::schedulerType(),
          can_use_smem_persistent);
  bool project_persistent_buffers =
      (project_strategy ==
       normalization_scheduler_utils::BufferProjectionStrategy::
           ProjectToInputs);
  auto persistent_buffer_size_bit = project_persistent_buffers
      ? persistent_buffer_size_info.projected_persistent_buffer_size_bit
      : persistent_buffer_size_info.persistent_buffer_size_bit;

  int64_t available_persistent_buffer_size_bit = normalization_scheduler_utils::
      getMaxRegOrSharedMemorySizeBitForPersistentBuffer(
          fusion,
          runtime_info,
          reduction_tvs,
          persistent_buffer_info,
          can_use_smem_persistent,
          project_persistent_buffers);
  // If one SM doesn't have enough persistent buffer size, try multiple SMs.
  // Current implementation doesn't support 3D reduction with multiple SMs.
  if (!is_3d_reduction &&
      available_persistent_buffer_size_bit < persistent_buffer_size_bit) {
    available_persistent_buffer_size_bit *=
        scheduler_utils::getMaxClusterSize();
  }

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  if (persistent_buffer_size_bit > available_persistent_buffer_size_bit) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        can_use_smem_persistent
            ? "not enough registers or shared memory for persistence."
            : "not enough registers for persistence and shared memory "
              "persistence is not supported yet.");
    return false;
  }

  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor;

  const int64_t required_sm_per_norm = ceilDiv(
      persistent_buffer_size_bit, scheduler_utils::register_file_size_bit);

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

std::unique_ptr<HeuristicParams> InnerPersistentKernelScheduler::
    computeHeuristics(
        Fusion* fusion,
        SchedulerRuntimeInfo& runtime_info,
        HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("InnerPersistentKernelScheduler::computeHeuristics");

  // Get properties of the fusion
  const auto& prop =
      normalization_scheduler_utils::getPersistentKernelProperties(
          fusion,
          runtime_info,
          data_cache,
          InnerPersistentKernelScheduler::schedulerType());

  // Check if TMA can be used
  bool use_tma = mayUseTma(fusion, prop) &&
      isOptionEnabled(EnableOption::TmaInnerPersistent);

  std::unique_ptr<HeuristicParams> rparams = nullptr;
  if (use_tma) {
    rparams = normalization_inner::tma::getInnerPersistentHeuristics(
        fusion, prop, data_cache);
  }

  // Fallback to non-TMA scheduler if TMA is not applicable
  if (rparams == nullptr) {
    rparams = normalization_inner::non_tma::getInnerPersistentHeuristics(
        fusion, prop, data_cache);
  }

  NVF_ERROR(rparams != nullptr);
  return rparams;
}

void InnerPersistentKernelScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("InnerPersistentKernelScheduler::schedule");

  // Check if this is TMA params
  if (auto tma_params = dynamic_cast<const InnerNormTmaParams*>(params)) {
    NVF_ERROR(
        tma_params->scheduler_type == schedulerType(),
        "Incorrect scheduler type in InnerNormTmaParams");
    normalization_inner::tma::scheduleInnerPersistent(fusion, tma_params);
    return;
  }

  // Otherwise, use non-TMA implementation with ReductionParams
  auto rparams = dynamic_cast<const ReductionParams*>(params);
  NVF_ERROR(
      rparams != nullptr && rparams->scheduler_type == schedulerType(),
      "Incorrect parameters sent to InnerPersistentKernelScheduler::schedule",
      params);
  normalization_inner::non_tma::scheduleInnerPersistent(fusion, rparams);
}

} // namespace nvfuser
