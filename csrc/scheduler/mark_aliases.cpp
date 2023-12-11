// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/internal_nodes.h>
#include <ir/utils.h>
#include <optimization/alias_analysis.h>
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

  // TODO(wujingyue): as a cleanup, move alias analysis out of
  // csrc/optimization.
  const optimization::AliasAnalysisResult analysis = optimization::findAliases(
      fusion, /*can_override_empty_allocation_domain=*/false);
  if (isDebugDumpEnabled(DebugDumpOption::SchedulerVerbose)) {
    vlog("Alias analysis result:\n", analysis.toString(/*indent_size=*/1));
  }

  for (TensorView* out :
       ir_utils::filterByType<TensorView>(fusion->outputs())) {
    const Val* in = analysis.getAliasedInput(out);
    if (in == nullptr) {
      continue;
    }

    fusion->aliasOutputToInput(
        out,
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<Val*>(in),
        AliasType::PointerArithmetic);
    vlog(
        "Marked ",
        ir_utils::varName(out),
        " as an alias of ",
        ir_utils::varName(in));
  }
}

} // namespace nvfuser
