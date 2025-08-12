// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <debug.h>
#include <exceptions.h>
#include <fusion.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <options.h>
#include <runtime/allocations.h>
#include <runtime/cutlass_compiled_kernel.h>
#include <runtime/cutlass_executor.h>
#include <scheduler/cutlass.h>
#include <type.h>

namespace nvfuser {

bool CutlassExecutor::supported(Fusion* fusion) {
  FUSER_PERF_SCOPE("CutlassExecutor::supported");

  // CutlassExecutor is only created when scheduler type is already known to be
  // Cutlass This method should return false to prevent automatic selection in
  // ExecutorDispatch
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

  // For now, create default CutlassParams
  // TODO: Get from scheduler once it's properly integrated
  CutlassParams cutlass_params_instance;

  auto cutlass_params = &cutlass_params_instance;

  // Create compile options
  CutlassCompileOptions compile_options;
  compile_options.optimization_level = 3;

  // Add CUTLASS include path if available
  if (const char* cutlass_path = std::getenv("CUTLASS_PATH")) {
    compile_options.include_paths.push_back(
        std::string(cutlass_path) + "/include");
  }

  // Create and compile the CUTLASS kernel
  cutlass_kernel_ = std::make_unique<CutlassCompiledKernel>(
      fusion_.get(), *cutlass_params, compile_options);
  cutlass_kernel_->compile();

  // Store the generated code for debugging
  generated_code_ = cutlass_kernel_->getCode();

  // Extract kernel name from descriptor
  kernel_name_ = cutlass_kernel_->getDescriptor().kernel_name;

  // Store launch parameters
  const auto& descriptor = cutlass_kernel_->getDescriptor();
  launch_params_ = LaunchParams(
      descriptor.grid_dim.x,
      descriptor.grid_dim.y,
      descriptor.grid_dim.z,
      descriptor.block_dim.x,
      descriptor.block_dim.y,
      descriptor.block_dim.z);

  if (descriptor.shared_memory_size > 0) {
    // TODO: Set shared memory size properly
    // launch_params_ doesn't have a setter for smem
  }

  // Output debug information if requested
  // Debug output disabled for now
  // TODO: Enable when debug flags are properly set up

  compiled_ = true;
}

bool CutlassExecutor::isCompiled() const {
  return compiled_ && cutlass_kernel_ && cutlass_kernel_->isCompiled();
}

KernelArgumentHolder CutlassExecutor::run(
    const KernelArgumentHolder& args,
    KernelArgumentHolder outputs,
    const LaunchParams& launch_constraints,
    const CompileParams& compile_params) {
  FUSER_PERF_SCOPE("CutlassExecutor::run");

  NVF_CHECK(isCompiled(), "CutlassExecutor must be compiled before running");

  auto device = c10::Device(c10::DeviceType::CUDA, args.getDeviceIndex());

  // Allocate outputs if not provided
  if (outputs.empty()) {
    outputs = allocateOutputs(fusion_.get(), args, device);
  }

  // Create kernel arguments including outputs
  KernelArgumentHolder kernel_args = args;
  for (const auto& arg : outputs) {
    kernel_args.push(arg);
  }

  cutlass_kernel_->run(kernel_args);

  return outputs;
}

KernelArgumentHolder CutlassExecutor::allocateOutputs(
    Fusion* fusion,
    const KernelArgumentHolder& inputs,
    const c10::Device& device) const {
  FUSER_PERF_SCOPE("CutlassExecutor::allocateOutputs");

  KernelArgumentHolder outputs;

  ExpressionEvaluator expr_eval = executor_utils::bindInputs(inputs, fusion);

  // For each output tensor in the fusion
  for (auto output : fusion->outputs()) {
    if (auto tv = dynamic_cast<TensorView*>(output)) {
      // Create output tensor with appropriate size and dtype
      at::ScalarType aten_type = data_type_to_aten(tv->dtype());

      // Use Byte for reduced precision types for now
      switch (aten_type) {
        // case at::ScalarType::Float8_e4m3fn:
        // case at::ScalarType::Float8_e5m2:
        // case at::ScalarType::Float8_e8m0fnu:
        case at::ScalarType::Float4_e2m1fn_x2:
          aten_type = at::ScalarType::Byte;
          break;
        default:
          break;
      }

      auto options = at::TensorOptions().dtype(aten_type).device(device);

      auto alias_info = fusion->getOutputAlias(tv);
      NVF_ERROR(
          alias_info.type == AllocationType::New,
          "Aliased inputs are not yet supported in CUTLASS fusions");

      const auto& [size, stride] = inferShapeOfOutput(tv, expr_eval);
      at::Tensor output_tensor = at::empty_strided(size, stride, options);

      outputs.push(output_tensor);
    }
  }

  return outputs;
}

// Keep these legacy methods for compatibility
std::string CutlassExecutor::generateCutlassCode(Fusion* fusion) {
  FUSER_PERF_SCOPE("CutlassExecutor::generateCutlassCode");
  return generated_code_;
}

void CutlassExecutor::compileGeneratedCode(const std::string& code) {
  FUSER_PERF_SCOPE("CutlassExecutor::compileGeneratedCode");
  // No longer used - compilation happens in CutlassCompiledKernel
}

void CutlassExecutor::extractLaunchParams() {
  FUSER_PERF_SCOPE("CutlassExecutor::extractLaunchParams");
  // No longer used - launch params are extracted from CutlassCompiledKernel
}

} // namespace nvfuser
