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

class FusionExecutorAbstract : public PolymorphicBase, public NonCopyable {
 public:
  //   // NVF_API was added for nvfuser_extension. See examples/sinh_extension.
  //   NVF_API FusionExecutorAbstract();

  //   //! To compile a fusion with the 32-bit index type, CompileParams
  //   //! must be passed in. There used to be an index type associated
  //   //! with KernelArgumentHolder, but it is no longer the case.
  //   NVF_API virtual void compileFusion(
  //       Fusion* fusion,
  //       const KernelArgumentHolder& args,
  //       const LaunchParams& launch_constraints,
  //       CompileParams compile_params,
  //       SchedulerType sceduler_type = SchedulerType::None,
  //       int64_t fusion_id = 0,
  //       int64_t concrete_id = 0,
  //       int64_t runtime_id = 0,
  //       int64_t group_id = 0) = 0;

  //   // TODO: args shouldn't come in a reference here because we will append
  //   the
  //   // outputs to be able to send it to the kernel. For now none of the users
  //   are
  //   // reconsuming the args, so it is okay. It isn't done now because
  //   changing it
  //   // from a reference makes a call as runFusion({}) ambiguous, and that is
  //   used
  //   // in some places in the codebase.
  //   NVF_API virtual std::vector<at::Tensor> runFusion(
  //       KernelArgumentHolder& args,
  //       const LaunchParams& launch_constraints = LaunchParams(),
  //       CompileParams compile_params = CompileParams(),
  //       std::vector<at::Tensor> outputs = {}) = 0;

 private:
};

} // namespace nvfuser
