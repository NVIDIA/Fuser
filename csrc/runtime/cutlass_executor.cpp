// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <runtime/cutlass_executor.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <exceptions.h>
#include <instrumentation.h>
#include <type.h>

namespace nvfuser {

bool CutlassExecutor::supported(Fusion* fusion) {
  FUSER_PERF_SCOPE("CutlassExecutor::supported");
  
  // CutlassExecutor is only created when scheduler type is already known to be Cutlass
  // This method should return false to prevent automatic selection in ExecutorDispatch
  return false;
}

void CutlassExecutor::compile(
    Fusion* fusion,
    const KernelArgumentHolder& args,
    const LaunchParams& launch_constraints,
    CompileParams compile_params) {
  FUSER_PERF_SCOPE("CutlassExecutor::compile");
  
  NVF_CHECK(!compiled_, "Cannot re-compile a CutlassExecutor");
  // No need to check scheduler type here as CutlassExecutor is only created
  // when the scheduler type is already known to be Cutlass
  
  // Clone the fusion
  fusion_ = std::make_unique<Fusion>(*fusion);
  
  // Generate CUTLASS code
  generated_code_ = generateCutlassCode(fusion_.get());
  
  // Compile the generated code
  compileGeneratedCode(generated_code_);
  
  // Extract launch parameters
  extractLaunchParams();
  
  compiled_ = true;
}

bool CutlassExecutor::isCompiled() const {
  return compiled_;
}

KernelArgumentHolder CutlassExecutor::run(
    const KernelArgumentHolder& args,
    KernelArgumentHolder outputs,
    const LaunchParams& launch_constraints,
    const CompileParams& compile_params) {
  FUSER_PERF_SCOPE("CutlassExecutor::run");
  
  NVF_CHECK(compiled_, "CutlassExecutor must be compiled before running");
  
  // TODO: Implement kernel execution
  NVF_THROW("CutlassExecutor::run not implemented yet");
  
  return outputs;
}

std::string CutlassExecutor::generateCutlassCode(Fusion* fusion) {
  FUSER_PERF_SCOPE("CutlassExecutor::generateCutlassCode");
  
  // TODO: Implement CUTLASS code generation
  // For now, just return a placeholder
  return R"(
// Generated CUTLASS kernel code
// TODO: Implement actual code generation
)";
}

void CutlassExecutor::compileGeneratedCode(const std::string& code) {
  FUSER_PERF_SCOPE("CutlassExecutor::compileGeneratedCode");
  
  // TODO: Implement compilation using NVRTC or nvcc
  // For now, just set a placeholder kernel name
  kernel_name_ = "cutlass_kernel";
}

void CutlassExecutor::extractLaunchParams() {
  FUSER_PERF_SCOPE("CutlassExecutor::extractLaunchParams");
  
  // TODO: Extract launch parameters from the compiled kernel
  // For now, use default launch params
  launch_params_ = LaunchParams();
}

} // namespace nvfuser