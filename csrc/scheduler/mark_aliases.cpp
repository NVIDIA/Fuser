// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <alias_analysis.h>
#include <ir/internal_nodes.h>
#include <ir/utils.h>
#include <options.h>
#include <scheduler/debug_utils.h>
#include <scheduler/mark_aliases.h>

namespace nvfuser {

template <typename... Args>
void vlog(const Args&... args) {
  scheduler_debug_utils::log("[mark_aliases] ", args...);
}

void markAliases(Fusion* fusion) {
  if (isDebugDumpEnabled(DebugDumpOption::SchedulerVerbose)) {
    vlog("Input fusion:");
    fusion->printMath();
  }

  const AliasAnalysisResult analysis =
      findAliases(fusion, EmptyAllocationAs::kLogical);
  if (isDebugDumpEnabled(DebugDumpOption::SchedulerVerbose)) {
    vlog("Alias analysis result:\n", analysis.toString(/*indent_size=*/1));
  }

  for (auto* out : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    // AllocationType::ReuseBuffer requires the output to be updated in place
    // so it can't be computed as an alias.
    if (fusion->getOutputAlias(out).type == AllocationType::ReuseBuffer) {
      continue;
    }

    if (TensorView* aliased_io = analysis.getRoot(out)) {
      if (aliased_io->isFusionInput() || aliased_io->isFusionOutput()) {
        fusion->aliasOutputToInput(out, aliased_io, AllocationType::Evaluate);
        vlog(
            "Marked ",
            ir_utils::varName(out),
            " as an alias of ",
            ir_utils::varName(aliased_io));
      }
    }
  }
}

} // namespace nvfuser
