// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

class TranslateRepeatToExpand : public OptimizationPass<TranslateRepeatToExpand> {
  friend class OptimizationPass<TranslateRepeatToExpand>;

 protected:
  static void runPass(Fusion* fusion);
  static std::string name() {
    return "TranslateRepeatToExpand";
  }
};

} // namespace nvfuser::preseg_passes
