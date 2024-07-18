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

namespace nvfuser::preseg_passes {
// Runs through the fusion and inserts a resharding Set Op after
// any resharding Expr that is not directly lowerable to a series of
// communications
class InsertReshardingsPass : public OptimizationPass<InsertReshardingsPass> {
  friend class OptimizationPass<InsertReshardingsPass>;

 protected:
  static void runPass(Fusion* fusion);
  static std::string name() {
    return "InsertReshardingsPass";
  }
};

} // namespace nvfuser::preseg_passes
