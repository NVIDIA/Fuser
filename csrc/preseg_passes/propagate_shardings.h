// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

// Very simple sharding propagation pass that identifies tvs without a
// DeviceMesh and shards it like its first producer tv with a sharding.  This
// assumes that all global inputs are sharded.  This cannot be done when the Op
// is inserted into the fusion, because the multidevice shcheduling hasn't been
// applied.
//
// TODO: Re-implement a robust and smatert sharding propagation pass.
class PropagateShardingsPass : public OptimizationPass<PropagateShardingsPass> {
  friend class OptimizationPass<PropagateShardingsPass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "PropagateShardingsPass";
  }
};

} // namespace nvfuser::preseg_passes
