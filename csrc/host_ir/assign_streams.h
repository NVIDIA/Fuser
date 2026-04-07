// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include "optimization_pass.h"

namespace nvfuser::hir {

// A host IR pass that assigns streams to stream-parallel loops.
class AssignStreams : public OptimizationPass<AssignStreams> {
  friend class OptimizationPass<AssignStreams>;

 protected:
  static void runPass(Fusion* fusion);

  static constexpr std::string_view name() {
    return "AssignStreams";
  }
};

} // namespace nvfuser::hir
