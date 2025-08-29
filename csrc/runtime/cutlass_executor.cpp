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
    const LaunchParams& launch_params,
    CompileParams compile_params) {
  FUSER_PERF_SCOPE("CutlassExecutor::compile");

  NVF_CHECK(!isCompiled(), "Cannot re-compile a CutlassExecutor");
  // No need to check scheduler type here as CutlassExecutor is only created
  // when the scheduler type is already known to be Cutlass

  // Clone the fusion
  fusion_ = std::make_unique<Fusion>(*fusion);

  // Add CUTLASS include path if available
  if (const char* cutlass_path = std::getenv("CUTLASS_PATH")) {
    compile_params.include_paths.push_back(
        std::string(cutlass_path) + "/include");
  }

  // Create and compile the CUTLASS kernel
  c10::Device device(c10::DeviceType::CUDA, args.getDeviceIndex());
  cutlass_kernel_ = std::make_unique<CutlassCompiledKernel>(
      fusion_.get(), compile_params, device);
  cutlass_kernel_->compile(launch_params);
}

bool CutlassExecutor::isCompiled() const {
  return cutlass_kernel_ && cutlass_kernel_->isCompiled();
}

KernelArgumentHolder CutlassExecutor::run(
    const KernelArgumentHolder& args,
    KernelArgumentHolder outputs) {
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

} // namespace nvfuser
