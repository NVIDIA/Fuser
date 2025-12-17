// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include "host_ir/container.h"
#include "optimization_pass.h"

namespace nvfuser::hir {

// A host IR pass that inserts allocations and deallocations. Some allocations
// are already inserted by lowerSegmentedFusionToHostIr. This pass inserts more
// allocations for functionality and/or performance.
class AllocateAndDeallocate : public OptimizationPass<AllocateAndDeallocate> {
  friend class OptimizationPass<AllocateAndDeallocate>;

 protected:
  static void runPass(Fusion* fusion);

  static constexpr std::string_view name() {
    return "AllocateAndDeallocate";
  }
};

} // namespace nvfuser::hir
