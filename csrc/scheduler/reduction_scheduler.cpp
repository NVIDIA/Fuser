// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <instrumentation.h>
#include <scheduler/debug_utils.h>
#include <scheduler/reduction_scheduler.h>
#include <scheduler/registry_utils.h>
#include <scheduler/utils.h>

namespace nvfuser {

ReductionScheduler::ReductionScheduler(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache)
    : SchedulerEntry(ScheduleHeuristic::Reduction) {
  computeHeuristics(fusion, runtime_info, data_cache);
}

void ReductionScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  params_ = getReductionHeuristics(fusion, runtime_info, data_cache);
  NVF_ERROR(params_ != nullptr);
}

void ReductionScheduler::schedule(Fusion* fusion) {
  FUSER_PERF_SCOPE("Schedule Single Reduction");
  scheduleReduction(fusion, reductionParams());
}

//! Check if the reduction heuristics apply in given fusion
bool ReductionScheduler::canScheduleCompileTime(Fusion* fusion) {
  // Needs at least one reduction to consider.
  if (ir_utils::getReductionOps(fusion).empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Reduction, "No reduction op to schedule");
    return false;
  }

  if (ir_utils::filterByType<TensorView>(fusion->inputs()).empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Reduction, "Scheduling not supported with no input");
    return false;
  }

  // Check that inputs of all select/gather-like ops are fusion inputs
  if (registry_utils::rejectScheduleForMemoryPromotion(
          fusion, ScheduleHeuristic::Reduction)) {
    return false;
  }

  // Fusions handled by reduction scheduler cannot have MmaOp.
  if (!ir_utils::getMmaOps(fusion).empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Reduction, "no support for mma ops.");
    return false;
  }

  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);

  if (reduction_tvs.empty()) {
    // Use pointwise logic
    return false;
  }

  if (registry_utils::hasNonUniqueBcast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Reduction,
        "Broadcasting dimension might be broadcasting to multiple sizes.");
    return false;
  }

  if (!ir_utils::getViewOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction,
          "Fusion requires view being reversible.");
      return false;
    }

    // Reduction scheduler simply uses reduction_tvs[0] as the reference, if
    // that changes, this needs to be changed.
    if (registry_utils::reductionInterferingView(
            fusion, ca_map, reduction_tvs[0])) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction,
          "View may interfere with reduction scheduling.");
      return false;
    }
  }

  // Make sure reduction axes are consistent through the fusion
  auto reduction_ops = ir_utils::getReductionOps(fusion);
  if (reduction_ops.size() > 1) {
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
              ScheduleHeuristic::Reduction,
              "Inconsistent reduction axes ",
              red,
              "is not ",
              axis_count);
          return false;
        }
      }
    }

    // Use root domain map to check the reduction ops have the same axes
    FusionGuard fg(fusion);
    ComputeAtRootDomainMap root_map;
    root_map.build(true);

    // red_ops.size()>1 checked before
    for (size_t it = 1; it < reduction_tvs.size(); it++) {
      if (!registry_utils::checkPatternEquivalence(
              reduction_tvs[it - 1], reduction_tvs[it], root_map)) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Reduction,
            "Un-mapped multi-reduction: ",
            reduction_tvs[it - 1],
            " ",
            reduction_tvs[it]);
        return false;
      }
    }
  }

  // Doesn't allow persistent kernels in this scheduler
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
  if (!persistent_buffer_info.persistent_buffers.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Reduction,
        "need persistent buffers that reduction scheduler doesn't handle");
    return false;
  }

  if (!registry_utils::SchedulerTopologyChecker::supportedPostReductionFusion(
          fusion, reduction_tvs) ||
      registry_utils::SchedulerTopologyChecker::hasPostReductionBCast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Reduction, "has unsupported post reduction fusion");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasGatherToBroadcastBeforeReduction(fusion, reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        ScheduleHeuristic::Reduction,
        "has unsupported gather-like ops before reduction");
    return false;
  }

  return true;
}

bool ReductionScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  return true;
}

} // namespace nvfuser