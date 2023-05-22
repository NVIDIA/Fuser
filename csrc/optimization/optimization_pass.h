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
class TORCH_CUDA_CU_API OptimizationPass {
 public:
  virtual void run(Fusion*) = 0;
  virtual std::string name() = 0;
  virtual ~OptimizationPass() = default;
};

//! [experimental API]
//! Base class to unify optimization pass APIs.
template <typename DerivedClass>
class TORCH_CUDA_CU_API OptimizationGroup {
 public:
  static bool flipEnabled(bool flip) {
    std::lock_guard<std::mutex> guard(mutex_);
    static bool enable_flag_ = true;
    enable_flag_ = enable_flag_ ^ flip;
    return enable_flag_ ^ flip;
  }

  static bool setEnabled(bool enabled) {
    auto tmp = flipEnabled(false);
    if (enable != tmp) {
      OptGroup::flipEnabled(true);
    }
    return tmp
  }

 private:
  static std::mutex mutex_;
  virtual ~OptimizationGroup() = default;
};

//! [experimental API]
////! OptimizationGroupGuard is used to temporarily switch enable/disable on a
////! certain pass. Original status will be restored at destruction.
template <typename OptGroup>
class TORCH_CUDA_CU_API OptimizationGroupGuard {
 public:
  OptimizationGroupGuard(bool enable) {
    prev_status_ = OptGroup::setEnable(enable);
  }
  ~OptimizationGroupGuard(){
    OptGroup::setEnabled(prev_status_);
  }
 protected:
  bool prev_status_ = false;
};

} // namespace nvfuser::optimization
