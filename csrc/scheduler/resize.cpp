// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <debug.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <scheduler/cache_policy_refiner.h>
#include <scheduler/debug_utils.h>
#include <scheduler/mark_aliases.h>
#include <scheduler/pointwise_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/resize.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/tools/loop_domain_scheduler.h>
#include <scheduler/tools/resize_utils.h>
#include <val_graph_visitor.h>

namespace nvfuser {

namespace {

// Just use the pointwise version for now
TensorView* getReferenceTensor(Fusion* fusion) {
  return pointwise_utils::getReferenceTensor(fusion);
}

} // namespace

bool ResizeScheduler::canScheduleCompileTime(Fusion* fusion) {
  if (!isOptionEnabled(EnableOption::ResizeScheduler)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Not enabled");
    return false;
  }

  if (!ir_utils::hasOpsOfType<SliceOp, PadOp>(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "No resize op to schedule");
    return false;
  }

  if (scheduler_utils::isResharding(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Fusion is resharding.");
    return false;
  }

  if (ir_utils::hasAnyReductionOps(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "No support for reduction ops");
    return false;
  }

  if (registry_utils::hasNonUniqueBcast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Broadcasting dimension might be broadcasting to multiple sizes.");
    return false;
  }

  // For now, the resize scheduler is only allowed for a limited set
  // of fusion patterns. The restrictions are planned to be
  // incrementally relaxed.

  IdModel id_model(fusion, /*build_graphs=*/false);
  const auto& broadcast_graph = id_model.buildBroadcastGraph();

  // For now, only a single resize op is allowed to exist.
  auto resize_based_tensor_ops = ir_utils::getOpsOfType<SliceOp, PadOp>(fusion);
  if (resize_based_tensor_ops.size() != 1) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Only a single resize op is allowed.");
    return false;
  }

  auto resize_out_tv =
      resize_based_tensor_ops.at(0)->output(0)->as<TensorView>();

  auto all_dep_vals = DependencyCheck::getAllValsBetween(
      {fusion->inputs().begin(), fusion->inputs().end()}, {resize_out_tv});
  for (auto tv : ir_utils::filterByType<TensorView>(all_dep_vals)) {
    if (tv == resize_out_tv) {
      continue;
    }
    if (tv->isFusionOutput()) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(),
          "Dependency to fusion output not allowed: ",
          tv->toString());
      return false;
    }
    for (auto consumer_of_tv : ir_utils::consumerTvsOf(tv)) {
      if (std::find(all_dep_vals.begin(), all_dep_vals.end(), consumer_of_tv) ==
          all_dep_vals.end()) {
        scheduler_debug_utils::canScheduleRejectReason(
            schedulerType(),
            "Resize inputs must be exclusively consumed by resize: ",
            consumer_of_tv->toString());
        return false;
      }
    }
  }

  // Slicing of or to a broadcast ID is not allowed yet.
  for (auto tensor_op : resize_based_tensor_ops) {
    TensorView* out_tv = tensor_op->output(0)->as<TensorView>();
    for (auto logical_id : out_tv->getLogicalDomain()) {
      Resize* resize = dynamic_cast<Resize*>(logical_id->definition());
      if (resize == nullptr) {
        continue;
      }

      if (resize->out()->isBroadcast()) {
        scheduler_debug_utils::canScheduleRejectReason(
            schedulerType(), "Resize to a broadcast ID is not allowed.");
        return false;
      }

      // Need to check the broadcast group rather than just the input
      // ID only. For example,
      //
      // t0: [i0]
      // t1: [b1]
      // t2 = t0 + t1
      // t3 = slice(t2)
      //
      // Then, propagating the slice to its inputs would try to
      // propagate the resize op to b1 as well, which would fail due
      // to issue #3571
      const auto& input_group = broadcast_graph.toGroup(resize->in());
      if (std::any_of(
              input_group->begin(), input_group->end(), [](Val* inp_val) {
                return inp_val->as<IterDomain>()->isBroadcast();
              })) {
        scheduler_debug_utils::canScheduleRejectReason(
            schedulerType(), "Resize of a broadcast ID is not allowed.");
        return false;
      }
    }
  }

  // This doesn't work yet due to issue #3571
  auto ref_tv = getReferenceTensor(fusion);
  if (std::any_of(
          ref_tv->getLogicalDomain().begin(),
          ref_tv->getLogicalDomain().end(),
          [](IterDomain* logical_id) { return logical_id->isBroadcast(); })) {
    return false;
  }

  // Disable the scheduler if there's a squeeze op. The loop option
  // may also need to be enabled in that case, but that option is not
  // turned on automatically yet.
  if (ir_utils::hasOpsOfType<SqueezeOp>(fusion)) {
    return false;
  }

  return true;
}

std::unique_ptr<HeuristicParams> ResizeScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("ResizeScheduler::computeHeuristics");
  auto params = std::make_unique<HeuristicParams>(SchedulerType::Resize);
  params->cparams.index_type = runtime_info.getIndexType();
  return params;
}

void ResizeScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  FUSER_PERF_SCOPE("ResizeScheduler::schedule");

  FusionGuard fg(fusion);

  scheduler_utils::clearMemorySpace(fusion);

  scheduler_utils::cacheInputs(fusion, true);
  scheduler_utils::cacheAndForkOutputs(fusion, true);

  for (auto expr : fusion->exprs()) {
    if (!expr->isOneOf<SliceOp, PadOp>()) {
      continue;
    }

    scheduler_tools::propagateResizeToInputs(expr);
  }

  auto ref_tv = getReferenceTensor(fusion);

  // Just simple scheduling for now.
  // TODO: Do something smarter. Can just use the pointwise scheduler?

  // Make sure the DID ID located at the outermost position
  const auto outermost_pos = scheduler_utils::reorderDevicesToOuter(ref_tv);

  // Schedule only the remaining IDs
  ref_tv->flatten(outermost_pos);
  ref_tv->split(outermost_pos, 128);
  ref_tv->split(outermost_pos, 1 << 14);
  ref_tv->axis(-1)->parallelize(ParallelType::TIDx);
  ref_tv->axis(-2)->parallelize(ParallelType::BIDx);

  // Propagate the reference to the other tensors
  scheduler_tools::scheduleLoopDomainsLike(
      fusion->allTvs(), ref_tv->getLoopDomain());

  inlineMost();

  markAliases(fusion);
}

} // namespace nvfuser
