// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir_interface_nodes.h>

namespace nvfuser::optimization {

//! [experimental API]
//! enum class to group optimization pass groups that runs at certain time in
//! the fusion execution.
enum class TORCH_CUDA_CU_API OptimizationPassCategory { PreSegmenter, Null };

using FusionPass = std::function<void(Fusion*)>;

//! [experimental API]
//! OptimizationPass is the base class to unify optimization pass APIs.
class TORCH_CUDA_CU_API OptimizationPass {
 public:
  virtual FusionPass func() = 0;
  virtual std::string name() = 0;
  virtual ~OptimizationPass() = default;
};

//! [experimental API]
//! OptimizationPassGuard is used to temporarily switch enable/disable on a
//! certain pass. Original status will be restored at exit.
class TORCH_CUDA_CU_API OptimizationPassGuard {
 public:
  OptimizationPassGuard(const OptimizationPassCategory& category, bool enable);
  ~OptimizationPassGuard();

 protected:
  OptimizationPassCategory cat_;
  bool prev_status_ = false;
};

//! [experimental API]
//! Register optimization pass with the `OptimizationPassCategroty`.
//!
//! all registered passes will run in order, where:
//! higher priority pass runs earlier;
//! newly registered pass runs at the end of all passes with identical priority.
TORCH_CUDA_CU_API void registerOptimizationPass(
    const OptimizationPassCategory& category,
    OptimizationPass* pass,
    int priority = 0);

//! [experimental API]
//! Run `category` group of optimization passes to `fusion`.
TORCH_CUDA_CU_API void applyOptimizationPass(
    const OptimizationPassCategory& category,
    Fusion* fusion);

//! [experimental API]
//! Switch the enable flag for a `category` group of optimization passes.
//! Returns the previous `enabled` status. Argument `std::optional<bool> enable`
//! is used to update the enable flag. An std::nullopt arg will leave the flag
//! unchanged.
TORCH_CUDA_CU_API bool switchOptimizationPass(
    const OptimizationPassCategory& category,
    std::optional<bool> enable);

} // namespace nvfuser::optimization
