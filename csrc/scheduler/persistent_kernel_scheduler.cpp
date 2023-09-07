// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/persistent_kernel_scheduler.h>
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

PersistentKernelScheduler::PersistentKernelScheduler(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache)
    : SchedulerEntry(ScheduleHeuristic::None) {}

bool PersistentKernelScheduler::canScheduleCompileTime(Fusion* fusion) {
  // (1) common checks shared by all sub classes
  leadingCommonCompileTimeCheck(fusion, ScheduleHeuristic::PersistentKernel);

  // (2) specific checks for each scheduler
  std::unique_ptr<PersistentScheduler> scheduler;
  switch (getReductionType(fusion)) {
    case ReductionType::Inner:
      scheduler = std::make_unique<InnerPersistentScheduler>();
      break;
    case ReductionType::Outer:
      scheduler = std::make_unique<OuterPersistentScheduler>();
      break;
    case ReductionType::InnerOuter:
      scheduler = std::make_unique<InnerOuterPersistentScheduler>();
      break;
    default:
      return false;
  }
  bool can_schedule = scheduler->canScheduleCompileTime(fusion);

}

bool PersistentKernelScheduler::compileTimeCheckReductionAxis(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs,
    ScheduleHeuristic heuristic) {
  // Use root domain map to check the reduction ops have the same axes
  FusionGuard fg(fusion);
  ComputeAtRootDomainMap root_map;
  root_map.build(true);
  for (const auto it : c10::irange(1, reduction_tvs.size())) {
    if (!checkPatternEquivalence(
            reduction_tvs[it - 1], reduction_tvs[it], root_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          heuristic,
          "unmapped reduction ",
          reduction_tvs[it - 1],
          " and ",
          reduction_tvs[it]);
      return false;
    }
  }
  return true;
}

ReductionType PersistentKernelScheduler::getReductionType(Fusion* fusion) {
  const auto& reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  bool is_inner_reduction = false;
  bool is_outer_reduction = false;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      is_inner_reduction = true;
    } else {
      is_outer_reduction = true;
    }
  }
  if (is_inner_reduction && is_outer_reduction) {
    return ReductionType::InnerOuter;
  } else if (is_inner_reduction) {
    return ReductionType::Inner;
  } else if (is_outer_reduction) {
    return ReductionType::Outer;
  } else {
    return ReductionType::None;
  }
}

bool PersistentKernelScheduler::leadingCommonCompileTimeCheck(
    Fusion* fusion,
    ScheduleHeuristic heuristic) {
  // Needs at least one reduction to consider.
  auto reduction_ops = ir_utils::getReductionOps(fusion);
  if (reduction_ops.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic, "needs a reduction op");
    return false;
  }

  if (ir_utils::filterByType<TensorView>(fusion->inputs()).empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic, "Scheduling not supported with no input");
    return false;
  }

  // Check that inputs of all select/gather-like ops are fusion inputs
  if (rejectScheduleForMemoryPromotion(fusion, heuristic)) {
    return false;
  }

  // Fusions handled by persistent kernel scheduler cannot have MmaOp.
  if (!ir_utils::getMmaOps(fusion).empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic, "no support for mma ops.");
    return false;
  }

  if (hasNonUniqueBcast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic,
        "Broadcasting dimension might be broadcasting to multiple sizes.");
    return false;
  }
  return true;
}

bool PersistentKernelScheduler::tailingCommonCompileTimeCheck(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs,
    ScheduleHeuristic heuristic) {
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
  TensorView* reference_tv =
      combined_inner_outer ? inner_reduction_tvs[0] : reduction_tvs[0];
  if (!ir_utils::getViewOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    if (requiresForwardViewReplay(fusion, ca_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          heuristic, "Fusion requires view being reversible.");
      return false;
    }

    // Persistent scheduler simply uses reference_tv as the reference, if
    // that changes, this needs to be changed.
    if (reductionInterferingView(fusion, ca_map, reference_tv)) {
      scheduler_debug_utils::canScheduleRejectReason(
          heuristic, "View may interfere with normalization scheduling.");
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
            heuristic,
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
        heuristic, "no persistent buffer identified");
    return false;
  }

  if (SchedulerTopologyChecker::hasNonNormalizePostReductionBCast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic, "unsupported post reduction normalization");
    return false;
  }

  if (SchedulerTopologyChecker::hasGatherToBroadcastBeforeReduction(
          fusion, reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic, "has unsupported gather-like ops before normalization");
    return false;
  }
  return true;
}

} // namespace nvfuser
