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

//! AddAxiomsPass adds extent > 0 as axioms of the IR container for all tensors
class AddAxiomsPass : public OptimizationPass<AddAxiomsPass> {
  friend class OptimizationPass<AddAxiomsPass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "AddAxiomsPass";
  }
};

} // namespace nvfuser::preseg_passes
