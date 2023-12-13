// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <alias_analysis.h>
#include <ir/utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/mark_aliases.h>
#include <scheduler/no_op.h>
#include <scheduler/registry_utils.h>

namespace nvfuser {

template <typename... Args>
void vlog(const Args&... args) {
  scheduler_debug_utils::log("[no_op] ", args...);
}

NoOpScheduler::NoOpScheduler(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache)
    : SchedulerEntry(heuristicType()) {
  params_ = std::make_shared<NoOpHeuristic>("", runtime_info.getIndexType());
}

namespace {
bool allOutputsArePointerArithmetics(Fusion* fusion) {
  const AliasAnalysisResult analysis =
      findAliases(fusion, /*can_override_empty_allocation_domain=*/false);
  auto out_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
  return std::all_of(
      out_tvs.begin(), out_tvs.end(), [&analysis](TensorView* out) {
        return analysis.getNearestAliasedIo(out) != nullptr;
      });
}
} // namespace

//! Check if the no-op heuristics apply in given fusion
bool NoOpScheduler::canScheduleCompileTime(Fusion* fusion) {
  if (fusion->isNoOp()) {
    return true;
  }

  if (allOutputsArePointerArithmetics(fusion)) {
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
            heuristicType(), "reduction of non-zero elements is not supported");
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
          heuristicType(), "output has a concrete dimension");
      return false;
    }
  }

  // Check that inputs of all select/gather-like ops are fusion inputs
  if (registry_utils::rejectScheduleForMemoryPromotion(
          fusion, heuristicType())) {
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
  markAliases(fusion);
}

void NoOpScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  // Heuristics is no-op.
  return;
}
} // namespace nvfuser
