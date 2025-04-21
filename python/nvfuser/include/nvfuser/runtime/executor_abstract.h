// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/executor_params.h>
#include <runtime/executor_utils.h>
#include <scheduler/scheduler_types.h>

namespace nvfuser {

class ExecutorAbstract : public PolymorphicBase, public NonCopyable {
 public:
  ExecutorAbstract(
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      int64_t group_id = 0)
      : fusion_id_(fusion_id),
        concrete_id_(concrete_id),
        runtime_id_(runtime_id),
        group_id_(group_id) {}
  virtual bool isCompiled() const = 0;

 protected:
  // ID of fusion in python frontend fusion cache, which maps to a single
  // FusionExecutorCache.
  int64_t fusion_id_ = 0;

  // ID of (device, concrete_info) key in FusionExecutorCache
  int64_t concrete_id_ = 0;

  // ID of FusionKernelRuntime given (device, concrete_info) key
  int64_t runtime_id_ = 0;

  // ID of segment in FusionKernelRuntime
  int64_t group_id_ = 0;

 private:
};

} // namespace nvfuser
