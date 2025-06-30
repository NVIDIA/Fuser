// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

// Given patterns like this:
//
//   T3 = broadcast(T2)
//   T4 = expand(T3)
//   ...
//   T7 = broadcast(T2)
//   T8 = expand(T7)
//   T9 = set(T8)
//
// such that the expanded dims of T3 and T8 match, we will modify the definition of T9 to
//
//   T9 = set(T4)
//
// Note that the broadcasts do not need to be expanded. We use the permissive map to check whether dimensions match and can be re-used, so if a broadcast is concretized against a concrete dimension that matches an earlier broadcast, we can reuse it.
//
// TODO: example of the above comment
//
// See https://github.com/NVIDIA/Fuser/issues/4670
class ReuseBroadcasts : public OptimizationPass<ReuseBroadcasts> {
  friend class OptimizationPass<ReuseBroadcasts>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "ReuseBroadcasts";
  }
};

} // namespace nvfuser::preseg_passes
