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

//! RemoveEmptyPass removes intermediate empty tensors (those with at least one
//! extent zero thar are neither a fusion output or input).
class InsertSegmentSetPass : public OptimizationPass<InsertSegmentSetPass> {
  friend class OptimizationPass<InsertSegmentSetPass>;

 protected:
  static void runPass(Fusion* fusion);
  static std::string name() {
    return "InsertSegmentSet";
  }
};

} // namespace nvfuser::preseg_passes
