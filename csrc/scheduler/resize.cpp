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
#if 0
  fusion->printMath();
  fusion->print();
  std::cout << std::endl;
#endif
  scheduler_utils::clearMemorySpace(fusion);

  scheduler_utils::cacheInputs(fusion, true);

  for (auto expr : fusion->exprs()) {
    if (!expr->isOneOf<SliceOp, PadOp>()) {
      continue;
    }

    scheduler_tools::propagateResizeToInputs(expr);
  }
#if 0
  fusion->print();
  std::cout << std::endl;

  for (auto tv : fusion->allTvs()) {
    std::cerr << "scheduled T" << tv->name() << "\n";
    if (tv->hasRoot()) {
      std::cerr << "\tRoot: " << toDelimitedString(tv->getRootDomain()) << "\n";
    }
    std::cerr << "\tLogical: " << toDelimitedString(tv->getLogicalDomain())
              << "\n";
    std::cerr << "\tLoop: " << toDelimitedString(tv->getLoopDomain()) << "\n";
    std::cerr << "\tAdditional ids: "
              << toDelimitedString(tv->domain()->additionalIDs()) << "\n";
    std::cerr << "\tInitial loop ids: "
              << toDelimitedString(tv->domain()->initialLoop()) << "\n";
    for (auto expr : tv->domain()->allExprs()) {
      std::cerr << expr->toString(4);
    }
  }
#endif
  auto ref_tv = getReferenceTensor(fusion);

  // Just simple scheduling for now.
  // TODO: Do something smarter. Can just use the pointwise scheduler?

  // Make sure the DID ID located at the outermost position
  int64_t reorder_pos = 0;
  std::unordered_map<int64_t, int64_t> old2new;
  for (const auto i : c10::irange(ref_tv->getLoopDomain().size())) {
    if (isParallelTypeDeviceDim(ref_tv->axis((int64_t)i)->getParallelType())) {
      old2new.emplace((int64_t)i, reorder_pos);
      ++reorder_pos;
    }
  }
  ref_tv->reorder(old2new);

  // Schedule only the remaining IDs
  const auto outermost_pos = (int64_t)old2new.size();
  ref_tv->flatten(outermost_pos);
  ref_tv->split(outermost_pos, 128);
  ref_tv->split(outermost_pos, 1 << 14);
  ref_tv->axis(-1)->parallelize(ParallelType::TIDx);
  ref_tv->axis(-2)->parallelize(ParallelType::BIDx);

  // Propagate the reference to the other tensors
  scheduler_tools::scheduleLoopDomainsLike(
      fusion->allTvs(), ref_tv->getLoopDomain());

  inlineMost();

  // TODO: Alias support doesn't seem to be working. For example, see
  // AliasTest.AliasOutputBeforeNonAliasOutput.
  markAliases(fusion);

  return;
}

} // namespace nvfuser
