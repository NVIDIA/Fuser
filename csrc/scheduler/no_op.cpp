// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/utils.h>
#include <multidevice/utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/mark_aliases.h>
#include <scheduler/no_op.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>

namespace nvfuser {

template <typename... Args>
void vlog(const Args&... args) {
  scheduler_debug_utils::log("[no_op] ", args...);
}

//! Check if the no-op heuristics apply in given fusion
bool NoOpScheduler::canScheduleCompileTime(Fusion* fusion) {
  if (fusion->isNoOp()) {
    return true;
  }

  if (scheduler_utils::isResharding(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Fusion is resharding.");
    return false;
  }

  // Check there're no non-trivial reduction ops.
  for (auto reduction : ir_utils::getAllTypesOfReductionOps(fusion)) {
    for (auto output :
         ir_utils::filterByType<TensorView>(reduction->outputs())) {
      auto concrete_dimension =
          TensorDomain::noReductions(output->getLogicalDomain());
      auto all_nonzero = std::none_of(
          concrete_dimension.begin(),
          concrete_dimension.end(),
          [](IterDomain* id) { return id->extent()->isZeroInt(); });
      if (all_nonzero) {
        scheduler_debug_utils::canScheduleRejectReason(
            schedulerType(), "reduction of non-zero elements is not supported");
        return false;
      }
    }
  }

  // Check that all outputs are either broadcast or ignored reduction.
  for (auto out_tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    auto concrete_dimension = TensorDomain::noReductions(
        TensorDomain::noBroadcasts(out_tv->getLoopDomain()));
    if (!concrete_dimension.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "output has a concrete dimension");
      return false;
    }
  }

  // Check that inputs of all select/gather-like ops are fusion inputs
  if (registry_utils::rejectScheduleForMemoryPromotion(
          fusion, schedulerType())) {
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
    HeuristicDataCache* data_cache) {
  // TODO:
  //  Pipe through dynamic zero checks.
  return true;
}

std::unique_ptr<HeuristicParams> NoOpScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  auto params = std::make_unique<HeuristicParams>(SchedulerType::NoOp);
  params->cparams.index_type = runtime_info.getIndexType();
  return params;
}

void NoOpScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  NVF_ERROR(
      params->scheduler_type == schedulerType(),
      "Invalid heuristic sent to NoOp scheduler: ",
      params);

  if (scheduler_utils::isResharding(fusion)) {
    return;
  }

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation. Otherwise, the generated kernel may unnecessarily
  // access intermediate buffers. See NoOpTest.ExpandedReduction.
  scheduler_utils::clearMemorySpace(fusion);

  markAliases(fusion);
}

} // namespace nvfuser
