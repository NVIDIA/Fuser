// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/optimization_pass.h>

namespace nvfuser::optimization {

// Fusion may have tensors with const extents and symbolic extents. This pass
// replaces symbolic extents with const extents if they are mapped to the exact
// same root domain set. See https://github.com/NVIDIA/Fuser/issues/1590.
class ConcretizeSymbolicRootDomainPass
    : public OptimizationPass<ConcretizeSymbolicRootDomainPass> {
  friend class OptimizationPass<ConcretizeSymbolicRootDomainPass>;

 protected:
  static void runPass(Fusion* fusion);
};

} // namespace nvfuser::optimization
