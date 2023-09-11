// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <ir/interface_nodes.h>
#include <ir/utils.h>

#include <atomic>

namespace nvfuser::optimization {

using FusionPass = std::function<void(Fusion*)>;

//! [experimental API]
//! Base class to unify optimization pass APIs.
//! OptimizationPass can be turned on/off programmatically with the `setEnabled`
//! API. There's helper template OptimizationPassGuard to temporarily switch the
//! enablement within the context. Note the we are using a curiously recurring
//! template pattern here to ensure that static objects are unique for each
//! DerivedClass. In order to apply OptimizationPass with the switch enabled,
//! you need to run the function with
//! `OptimizationPass<DerivedClass>::runPass(...)`
//!
//! Specific optimization pass needs to be created like:
//!
//!   class TORCH_CUDA_CU_API Pass0 : public OptimizationPass<Pass0> {
//!     friend class OptimizationPass<Pass0>;
//!
//!    protected:
//!     static void runPass(Fusion* fusion);
//!   };
template <typename DerivedClass>
class TORCH_CUDA_CU_API OptimizationPass {
 public:
  static void setEnabled(bool enabled) {
    flag_.store(enabled);
  }

  static bool getEnabled() {
    return flag_.load();
  }

  static void runPass(Fusion* fusion) {
    if (!flag_.load()) {
      return;
    }
    DerivedClass::runPass(fusion);
#ifndef NDEBUG
    // cycle detection is only enabled on debug run
    NVF_ERROR(
        ir_utils::checkCycle(fusion).empty(), "cycle detected in fusion IR");
#endif
  }

  virtual ~OptimizationPass() = default;

 protected:
  static inline std::atomic<bool> flag_{true};
};

//! [experimental API]
//! OptimizationPassGuard is used to temporarily switch enable/disable on a
//! certain pass. Original status will be restored at destruction.
template <typename OptPass>
class TORCH_CUDA_CU_API OptimizationPassGuard {
 public:
  OptimizationPassGuard(bool enabled) : prev_status_(OptPass::getEnabled()) {
    if (prev_status_ != enabled) {
      OptPass::setEnabled(enabled);
    }
  }
  ~OptimizationPassGuard() {
    OptPass::setEnabled(prev_status_);
  }

 protected:
  bool prev_status_ = false;
};

} // namespace nvfuser::optimization
