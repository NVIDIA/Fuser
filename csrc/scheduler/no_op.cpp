// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <scheduler/debug_utils.h>
#include <scheduler/no_op.h>
#include <scheduler/registry_utils.h>

namespace nvfuser {

NoOpScheduler::NoOpScheduler(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache)
    : SchedulerEntry(ScheduleHeuristic::NoOp) {
  params_ = std::make_shared<NoOpHeuristic>("", runtime_info.getIndexType());
}

//! Check if the no-op heuristics apply in given fusion
bool NoOpScheduler::canScheduleCompileTime(Fusion* fusion) {
  if (fusion->isNoOp()) {
    return true;
  }
  // Check there're no non-trivial reduction ops.
  for (auto reduction : ir_utils::getAllTypesOfReductionOps(fusion)) {
    for (auto output :
         ir_utils::filterByType<TensorView>(reduction->outputs())) {
      auto concrete_dimension =
          TensorDomain::noReductions(output->getRootDomain());
      auto all_nonzero = std::none_of(
          concrete_dimension.begin(),
          concrete_dimension.end(),
          [](IterDomain* id) { return id->extent()->isZeroInt(); });
      if (all_nonzero) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::NoOp,
            "reduction of non-zero elements is not supported");
        return false;
      }
    }
  }

  // Check that all outputs are either broadcast or ignored reduction.
  for (auto out_tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    auto concrete_dimension = TensorDomain::noReductions(
        TensorDomain::noBroadcasts(out_tv->getLeafDomain()));
    if (!concrete_dimension.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::NoOp, "output has a concrete dimension");
      return false;
    }
  }

  // Check that inputs of all select/gather-like ops are fusion inputs
  if (registry_utils::rejectScheduleForMemoryPromotion(
          fusion, ScheduleHeuristic::NoOp)) {
    return false;
  }

  // We have verified that all iterdomains on all output tv's are trivial
  // reductions,
  //  broadcasts or zero-sized. Therefore accepting this fusion for NoOp
  //  scheduling.
  return true;
}

bool NoOpScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  // TODO:
  //  Pipe through dynamic zero checks.
  return true;
}

void NoOpScheduler::schedule(Fusion* fusion) {
  // Schedule is no-op.
  return;
}

void NoOpScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  // Heuristics is no-op.
  return;
}
} // namespace nvfuser
