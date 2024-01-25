// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <exceptions.h>
#include <optimization/optimization_pass.h>

namespace nvfuser::optimization {

//! Check whether a TensorView is empty. During concretization, we traverse to
//! find a minimal set of TensorViews that have zero extents, and we then set
//! their extents to a constant 0. Here we check for those constant zero
//! extents.
bool isTVEmpty(TensorView* tv);

//! RemoveEmptyPass removes intermediate empty tensors (those with at least one
//! extent zero thar are neither a fusion output or input).
class RemoveEmptyPass : public OptimizationPass<RemoveEmptyPass> {
  friend class OptimizationPass<RemoveEmptyPass>;

 protected:
  static void runPass(Fusion* fusion);
};

} // namespace nvfuser::optimization
