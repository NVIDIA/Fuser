// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <preseg_passes/optimization_pass.h>

#include <string>

#include <fusion.h>

namespace nvfuser::preseg_passes {

// This can only run after InsertReshardingPass.  Assumes all resharding ops
// are either a set or reduction.  For each resharding operation that requires
// communication over a noncontiguous slices of the tensor, this pass inserts
// permutations necessary to push the device parallel axis to the front so that
// communication operations are contiguous.
class ReorderShardedAxisPass : public OptimizationPass<ReorderShardedAxisPass> {
  friend class OptimizationPass<ReorderShardedAxisPass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "ReorderShardedAxisPass";
  }
};

} // namespace nvfuser::preseg_passes
