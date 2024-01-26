// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/optimization_pass.h>
#include <visibility.h>

namespace nvfuser::optimization {

//! AddAxiomsPass adds extent > 0 as axioms of the IR container for all tensors
class AddAxiomsPass : public OptimizationPass<AddAxiomsPass> {
  friend class OptimizationPass<AddAxiomsPass>;

 protected:
  NVF_API static void runPass(Fusion* fusion);
};

} // namespace nvfuser::optimization
