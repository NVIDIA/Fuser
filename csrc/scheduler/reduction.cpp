// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/cuda/CUDAContext.h>
#include <debug.h>
#include <instrumentation.h>
#include <scheduler/debug_utils.h>
#include <scheduler/reduction.h>
#include <scheduler/reduction_non_tma.h>
#include <scheduler/reduction_tma.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>

namespace nvfuser {

//! Check if the reduction heuristics apply in given fusion
bool ReductionScheduler::canScheduleCompileTime(Fusion* fusion) {
  FUSER_PERF_SCOPE("ReductionScheduler::canScheduleCompileTime");

  for (auto tv : fusion->allTvs()) {
    if (tv->dtype() != DataType::Index &&
        dataTypeSizeBit(tv->dtype()) % 8 != 0) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "Does not support sub-byte data types.");
      return false;
    }
  }

  if (scheduler_utils::isResharding(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Fusion is resharding.");
    return false;
  }

  // Needs at least one reduction to consider.
  if (!ir_utils::hasAnyReductionOps(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "No reduction op to schedule");
    return false;
  }

  if (ir_utils::filterByType<TensorView>(fusion->inputs()).empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Scheduling not supported with no input");
    return false;
  }

  // Check that inputs of all select/gather-like ops are fusion inputs
  if (registry_utils::rejectScheduleForMemoryPromotion(
          fusion, schedulerType())) {
    return false;
  }

  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);

  if (reduction_tvs.empty()) {
    // Use pointwise logic
    return false;
  }

  // Reject when output IDs are not covered by reference tv. Assuming reduction
  // scheduler simply uses reduction_tvs[0] as the reference, if that changes,
  // this needs to be changed. see issue
  // https://github.com/NVIDIA/Fuser/issues/3811
  scheduler_tools::DomainMap domain_map(fusion);
  if (!domain_map.isValidReference(reduction_tvs[0], /*check_inputs=*/true)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Output contains ID that's not scheduled by reference tv.");
    return false;
  }

  if (registry_utils::hasNonUniqueBcast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Broadcasting dimension might be broadcasting to multiple sizes.");
    return false;
  }

  if (!ir_utils::getReshapeOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "Fusion requires view being reversible.");
      return false;
    }

    // Reduction scheduler simply uses reduction_tvs[0] as the reference, if
    // that changes, this needs to be changed.
    if (registry_utils::reductionInterferingView(
            fusion, ca_map, reduction_tvs[0])) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "View may interfere with reduction scheduling.");
      return false;
    }
  }

  // Make sure reduction axes are consistent through the fusion
  auto reduction_ops = ir_utils::getAllTypesOfReductionOps(fusion);
  if (reduction_ops.size() > 1) {
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
              "Inconsistent reduction root size: ",
              red->toString(),
              ", expected: ",
              axis_count);
          return false;
        }
      }
    }

    // Use root domain map to check the reduction ops have the same axes
    FusionGuard fg(fusion);
    ComputeAtLogicalDomainMap logical_map;
    logical_map.build(true);

    // red_ops.size()>1 checked before
    for (size_t it = 1; it < reduction_tvs.size(); it++) {
      if (!registry_utils::checkPatternEquivalence(
              reduction_tvs[it - 1], reduction_tvs[it], logical_map)) {
        scheduler_debug_utils::canScheduleRejectReason(
            schedulerType(),
            "Un-mapped multi-reduction: ",
            reduction_tvs[it - 1]->toString(),
            " and ",
            reduction_tvs[it]->toString());
        return false;
      }
    }
  }

  // Doesn't allow persistent kernels in this scheduler
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
  if (!persistent_buffer_info.persistent_buffers.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "need persistent buffers that reduction scheduler doesn't handle");
    return false;
  }

  if (!registry_utils::SchedulerTopologyChecker::supportedPostReductionFusion(
          fusion, reduction_tvs) ||
      registry_utils::SchedulerTopologyChecker::hasPostReductionBCast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "has unsupported post reduction fusion");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasGatherToBroadcastBeforeReduction(fusion, reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "has unsupported gather-like ops before reduction");
    return false;
  }

  return true;
}

bool ReductionScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("ReductionScheduler::canScheduleRunTime");
  return true;
}

namespace {

bool mayUseTma(
    const reduction_scheduler_utils::FusionRuntimeProperties& props) {
  auto dev_prop = at::cuda::getCurrentDeviceProperties();

  if (dev_prop->major < 9) {
    return false;
  }

  // Require the reduction shape is 2D inner reduction: [I, R]
  if (!props.fastest_dim_reduction) {
    return false;
  }

  if (props.total_reduction_numel != props.inner_most_dimension_numel) {
    return false;
  }

  // Skip TMA for small reductions
  if (props.total_reduction_numel < 128) {
    return false;
  }

  // Require reduction dim fits into smem, until we add iteration over large
  // reduction dim.
  const int64_t smem_elems = dev_prop->sharedMemPerBlockOptin /
      props.max_dtype_size_bit_for_vectorization;

  if (props.inner_most_dimension_numel > smem_elems) {
    return false;
  }

  // Smem check assumes only one input tensor.
  if (props.n_tensor_inputs != 1) {
    return false;
  }

  // Like vectorization, TMA requires 16-bytes alignment
  if (props.vectorize_factor <= 1) {
    return false;
  }

  return true;
}
} // namespace

std::unique_ptr<HeuristicParams> ReductionScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("ReductionScheduler::computeHeuristics");

  auto props = reduction_scheduler_utils::getFusionRuntimeProperties(
      fusion, runtime_info, data_cache);

  bool use_tma =
      mayUseTma(props) && isOptionEnabled(EnableOption::TmaReduction);

  std::unique_ptr<HeuristicParams> rparams = nullptr;
  if (use_tma) {
    rparams = reduction::tma::getReductionHeuristics(
        fusion, runtime_info, data_cache, props);
  }
  // Fallback to non-TMA scheduler if TMA is not applicable
  if (rparams == nullptr) {
    rparams = reduction::non_tma::getReductionHeuristics(
        fusion, runtime_info, data_cache, props);
  }
  NVF_ERROR(rparams != nullptr);
  return rparams;
}

void ReductionScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("ReductionScheduler::schedule");
  if (auto* tma_params = dynamic_cast<const TmaInnerReductionParams*>(params)) {
    reduction::tma::scheduleReduction(fusion, tma_params);
  } else {
    auto rparams = dynamic_cast<const ReductionParams*>(params);
    NVF_ERROR(
        rparams != nullptr,
        "Incorrect parameters sent to ReductionScheduler::schedule",
        params);
    reduction::non_tma::scheduleReduction(fusion, rparams);
  }
}
} // namespace nvfuser
