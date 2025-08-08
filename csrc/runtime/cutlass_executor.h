// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <runtime/compiled_kernel.h>
#include <runtime/executor_abstract.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/executor_params.h>
#include <scheduler/scheduler_types.h>
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

  // Returns true if fusion can be executed by CUTLASS
  static bool supported(Fusion* fusion);

  // Compile the fusion into a CUTLASS kernel
  void compile(
      Fusion* fusion,
      const KernelArgumentHolder& args,
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams());

  bool isCompiled() const override;

  // Execute the compiled CUTLASS kernel
  KernelArgumentHolder run(
      const KernelArgumentHolder& args,
      KernelArgumentHolder outputs = {},
      const LaunchParams& launch_constraints = LaunchParams(),
      const CompileParams& compile_params = CompileParams());

  const std::unique_ptr<Fusion>& fusion() const {
    return fusion_;
  }

 private:
  // Generate CUTLASS C++ code from the fusion
  std::string generateCutlassCode(Fusion* fusion);

  // Compile the generated code using NVRTC or nvcc
  void compileGeneratedCode(const std::string& code);

  // Extract launch parameters from compiled kernel
  void extractLaunchParams();

  // Allocate output tensors
  KernelArgumentHolder allocateOutputs(
      Fusion* fusion,
      const KernelArgumentHolder& inputs);

 private:
  std::unique_ptr<Fusion> fusion_;
  std::unique_ptr<CutlassCompiledKernel> cutlass_kernel_;
  LaunchParams launch_params_;
  bool compiled_ = false;

  // Generated CUTLASS code
  std::string generated_code_;

  // Kernel function name
  std::string kernel_name_;

  // Compiled kernel handle - no longer used
  void* kernel_func_ = nullptr;
};

} // namespace nvfuser
