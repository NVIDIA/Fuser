// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <debug.h>
#include <exceptions.h>
#include <host_ir/container.h>
#include <instrumentation.h>
#include <ir/interface_nodes.h>
#include <options.h>
#include <string_view>

#include <atomic>

namespace nvfuser::hir_pass {

//! Base class to unify host IR optimization pass APIs.
//! OptimizationPass can be turned on/off programmatically with the `setEnabled`
//! API. There's helper template OptimizationPassGuard to temporarily switch the
//! enablement within the context. Note the we are using a curiously recurring
//! template pattern here to ensure that static objects are unique for each
//! DerivedClass.
//!
//! Specific host IR optimization pass needs to be created like:
//!
//!   class Pass0 : public OptimizationPass<Pass0> {
//!     friend class OptimizationPass<Pass0>;
//!
//!    protected:
//!     void runPass(Fusion* fusion);
//!   };
template <typename DerivedClass>
class OptimizationPass {
 public:
  static void setEnabled(bool enabled) {
    flag_.store(enabled);
  }

  static bool getEnabled() {
    return flag_.load();
  }

  void runPass(Fusion* fusion) {
    if (!flag_.load()) {
      return;
    }

    FUSER_PERF_SCOPE(DerivedClass::name().data());
    static_cast<DerivedClass*>(this)->passImplementation(fusion);

    if (isDebugDumpEnabled(DebugDumpOption::HostIrLoweringLogging)) {
      debug() << "Fusion after pass: " << DerivedClass::name() << std::endl;
      if (fusion->isA<hir::HostIrContainer>()) {
        fusion->as<hir::HostIrContainer>()->print(debug());
      } else {
        fusion->printMath();
      }
      debug() << "========================================" << std::endl;
    }
  }

 protected:
  static inline std::atomic<bool> flag_{true};
};

//! OptimizationPassGuard is used to temporarily switch enable/disable on a
//! certain pass. Original status will be restored at destruction.
template <typename OptPass>
class OptimizationPassGuard {
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

} // namespace nvfuser::hir_pass
