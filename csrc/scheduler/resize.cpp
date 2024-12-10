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

bool ResizeScheduler::canScheduleCompileTime(Fusion* fusion) {
  std::cerr << "ResizeScheduler::canScheduleCompileTime\n";

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

  // For now, only a single resize op is allowed to exist.
  auto resize_based_tensor_ops = ir_utils::getOpsOfType<SliceOp, PadOp>(fusion);
  if (resize_based_tensor_ops.size() != 1) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Only a single resize op is allowed.");
    return false;
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
      if (resize->in()->isBroadcast()) {
        scheduler_debug_utils::canScheduleRejectReason(
            schedulerType(), "Resize of a broadcast ID is not allowed.");
        return false;
      }
    }
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

namespace {

TensorView* getReferenceTensor(Fusion* fusion) {
  return nullptr;
}

} // namespace

void ResizeScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  FUSER_PERF_SCOPE("ResizeScheduler::schedule");

  DebugStreamGuard dsg(std::cerr);

  FusionGuard fg(fusion);

  std::cerr << "ResizeScheduler::schedule\n";

  scheduler_utils::clearMemorySpace(fusion);

  scheduler_utils::cacheInputs(fusion, true);

  fusion->printMath();

  const auto exprs = fusion->exprs();
  for (auto expr : exprs) {
    if (!expr->isOneOf<SliceOp, PadOp>()) {
      continue;
    }

    std::cerr << "Propagating resize tensor op: " << expr->toString();
    scheduler_tools::propagateResizeToInputs(expr);
  }

  auto ref_tv = getReferenceTensor(fusion);

  std::cerr << "Reference: " << ref_tv->toString() << "\n";

  ref_tv->flatten();
  ref_tv->split(0, 128);
  ref_tv->split(0, 1 << 14);
  ref_tv->axis(-1)->parallelize(ParallelType::TIDx);
  ref_tv->axis(-2)->parallelize(ParallelType::BIDx);

  std::cerr << "Scheduled reference:\n";
  ref_tv->printTransforms();

  scheduler_tools::scheduleLoopDomainsLike(
      fusion->allTvs(), ref_tv->getLoopDomain());

  {
    std::cerr << "All done\n";
    fusion->printMath();
    for (auto tv : fusion->allTvs()) {
      std::cerr << "Final scheduled T" << tv->name() << "\n";
      if (tv->hasRoot()) {
        std::cerr << "\tRoot: " << toDelimitedString(tv->getRootDomain())
                  << "\n";
      }
      std::cerr << "\tLogical: " << toDelimitedString(tv->getLogicalDomain())
                << "\n";
      std::cerr << "\tLoop: " << toDelimitedString(tv->getLoopDomain()) << "\n";
      std::cerr << "\tAdditional ids: "
                << toDelimitedString(tv->domain()->additionalIDs()) << "\n";
      for (auto expr : tv->domain()->allExprs()) {
        std::cerr << expr->toString(4);
      }
    }
  }

  inlineMost();

  fusion->printMath();

  return;
}

} // namespace nvfuser
