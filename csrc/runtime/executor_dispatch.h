// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <runtime/executor.h>
#include <runtime/executor_abstract.h>

namespace nvfuser {

// Simple stateless dispatch system for KernelExecutor, HostIrExecutor, and
// ExprEvalExecutor
class ExecutorDispatch {
 public:
  // If `scheduler_type` is `SchedulerType::None`, this function Iterates
  // through executors in priority order creating the first executor that
  // returns true when checking their "supported" method. Otherwise, create the
  // executor according to `scheduler_type`, which is faster.
  //
  // The slow path (i.e. `SchedulerType::None`) is only used by
  // MultiDeviceExecutor at this moment.
  static std::unique_ptr<ExecutorAbstract> makeExecutor(
      Fusion* fusion,
      int64_t fusion_id = -1,
      int64_t concrete_id = -1,
      int64_t runtime_id = -1,
      int64_t group_id = -1,
      SchedulerType scheduler_type);

  static void compile(ExecutorAbstract* executor, Fusion* fusion);

  static void compile(
      ExecutorAbstract* executor,
      Fusion* fusion,
      const KernelArgumentHolder& args,
      const LaunchParams& launch_constraints,
      CompileParams compile_params,
      SchedulerType scheduler_type = SchedulerType::None);

  static bool isCompiled(const ExecutorAbstract* executor);

  static KernelArgumentHolder run(
      ExecutorAbstract* executor,
      const KernelArgumentHolder& args,
      KernelArgumentHolder outputs = {},
      const LaunchParams& launch_constraints = LaunchParams(),
      const CompileParams& compile_params = CompileParams());
};

} // namespace nvfuser
