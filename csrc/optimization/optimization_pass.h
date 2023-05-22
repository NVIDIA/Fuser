// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/interface_nodes.h>

#include <mutex>

namespace nvfuser::optimization {

using FusionPass = std::function<void(Fusion*)>;

//! [experimental API]
//! Base class to unify optimization pass APIs.
//! OptimizationPass is functional and defines the granularity of mutation
//! passes that is used to compose OptimizationGroups
class TORCH_CUDA_CU_API OptimizationPass {
 public:
  virtual void run(Fusion*) = 0;
  virtual std::string name() = 0;
  virtual ~OptimizationPass() = default;
};

//! [experimental API]
//! Base class to unify optimization group APIs.
//! OptimizationGroup composes optimization passes that is used at certain stage
//! in the runtime system. OptimizationGroup can be turned on/off
//! programmatically with the `setEnabled/flipEnabled` API. There's helper
//! template OptimizationGroupGuard to temporarily switch the enablement within
//! the context. Note the we are using a curiously recurring template pattern
//! here to ensure that static objects are unique for each DerivedClass. In
//! order to apply OptimizationGroup with the switch enabled, you need to run
//! the function with `OptimizationGroup<DerivedClass>::runPass(...)`
template <typename DerivedClass>
class TORCH_CUDA_CU_API OptimizationGroup {
 public:
  static bool flipEnabled(bool flip) {
    static std::mutex mutex_;
    static bool enable_flag_ = true;

    std::lock_guard<std::mutex> guard(mutex_);
    enable_flag_ = enable_flag_ ^ flip;
    return enable_flag_ ^ flip;
  }

  static bool setEnabled(bool enabled) {
    auto tmp = flipEnabled(false);
    if (enabled != tmp) {
      flipEnabled(true);
    }
    return tmp;
  }

  static void runPass(Fusion* fusion) {
    if (!flipEnabled(false)) {
      return;
    }
    DerivedClass::runPass(fusion);
  }

  virtual ~OptimizationGroup() = default;
};

//! [experimental API]
//! OptimizationGroupGuard is used to temporarily switch enable/disable on a
//! certain pass. Original status will be restored at destruction.
template <typename OptGroup>
class TORCH_CUDA_CU_API OptimizationGroupGuard {
 public:
  OptimizationGroupGuard(bool enabled)
      : prev_status_(OptGroup::setEnabled(enabled)) {}
  ~OptimizationGroupGuard() {
    OptGroup::setEnabled(prev_status_);
  }

 protected:
  bool prev_status_ = false;
};

} // namespace nvfuser::optimization
