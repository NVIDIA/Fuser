// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <runtime/compiled_kernel.h>
#include <runtime/cutlass_compiled_kernel.h>
#include <runtime/executor_abstract.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/executor_params.h>
#include <scheduler/cutlass.h>
#include <memory>
#include <string>

namespace nvfuser {

class CutlassCompiledKernel;

class CutlassExecutor : public ExecutorAbstract {
 public:
  CutlassExecutor(
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      int64_t group_id = 0)
      : ExecutorAbstract(fusion_id, concrete_id, runtime_id, group_id) {}

  //! Returns true if fusion can be executed by CUTLASS
  static bool supported(Fusion* fusion);

  //! Compile the fusion into a CUTLASS kernel
  void compile(
      Fusion* fusion,
      const CutlassParams& cutlass_params = CutlassParams());

  bool isCompiled() const override;

  // Execute the compiled CUTLASS kernel
  KernelArgumentHolder run(
      const KernelArgumentHolder& args,
      KernelArgumentHolder outputs = {});

  const std::unique_ptr<Fusion>& fusion() const {
    return fusion_;
  }

 private:
  // Allocate output tensors
  KernelArgumentHolder allocateOutputs(
      Fusion* fusion,
      const KernelArgumentHolder& inputs,
      const c10::Device& device) const;

 private:
  std::unique_ptr<Fusion> fusion_;
  std::unique_ptr<CutlassCompiledKernel> cutlass_kernel_;
  LaunchParams launch_params_;

  // Generated CUTLASS code
  std::string generated_code_;
};

} // namespace nvfuser
