// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <runtime/executor_dispatch.h>

namespace nvfuser {
namespace {} // namespace

// Iterates through executors in priority order creating the first executor that
// returns true when checking their "supported" method
std::unique_ptr<ExecutorAbstract> ExecutorDispatch::makeExecutor(
    Fusion* fusion) {
  if (HostIRExecutor::supported(fusion)) {
    return std::make_unique<HostIRExecutor>();
  }
  if (ExprEvalExecutor::supported(fusion)) {
    return std::make_unique<ExprEvalExecutor>();
  }
  if (KernelExecutor::supported(fusion)) {
    return std::make_unique<KernelExecutor>();
  }
  NVF_THROW("No executor supports provided fusion.");
}

void ExecutorDispatch::compile(
    std::unique_ptr<ExecutorAbstract>& executor,
    Fusion* fusion) {
  if (auto hire = dynamic_cast<HostIRExecutor*>(executor.get())) {
    hire->compile(fusion);
  }
  if (auto eee = dynamic_cast<ExprEvalExecutor*>(executor.get())) {
    eee->compile(fusion);
  }
  if (executor->isA<KernelExecutor>()) {
    NVF_THROW(
        "KernelExecutor needs more information to be provided for compilation.");
  }
  NVF_THROW("Unsupported Executor detected.");
}

void ExecutorDispatch::compile(
    std::unique_ptr<ExecutorAbstract>& executor,
    Fusion* fusion,
    const KernelArgumentHolder& args,
    const LaunchParams& launch_constraints,
    CompileParams compile_params,
    SchedulerType scheduler_type,
    int64_t fusion_id,
    int64_t concrete_id,
    int64_t runtime_id,
    int64_t group_id) {
  if (auto hire = dynamic_cast<HostIRExecutor*>(executor.get())) {
    hire->compile(fusion);
  }
  if (auto eee = dynamic_cast<ExprEvalExecutor*>(executor.get())) {
    eee->compile(fusion);
  }
  if (auto ke = dynamic_cast<KernelExecutor*>(executor.get())) {
    ke->compileFusion(
        fusion,
        args,
        launch_constraints,
        compile_params,
        scheduler_type,
        fusion_id,
        runtime_id,
        group_id);
  }
  NVF_THROW("Unsupported Executor detected.");
}

bool ExecutorDispatch::isCompiled(
    const std::unique_ptr<ExecutorAbstract>& executor) {
  if (auto hire = dynamic_cast<HostIRExecutor*>(executor.get())) {
    return hire->isCompiled();
  }
  if (auto eee = dynamic_cast<ExprEvalExecutor*>(executor.get())) {
    return eee->isCompiled();
  }
  if (auto ke = dynamic_cast<KernelExecutor*>(executor.get())) {
    return ke->isCompiled();
  }
  NVF_THROW("Unsupported Executor detected.");
}

std::vector<at::Tensor> ExecutorDispatch::run(
    std::unique_ptr<ExecutorAbstract>& executor,
    KernelArgumentHolder& args,
    std::vector<at::Tensor> outputs) {
  if (auto hire = dynamic_cast<HostIRExecutor*>(executor.get())) {
    return hire->run(args, outputs);
  }
  if (auto eee = dynamic_cast<ExprEvalExecutor*>(executor.get())) {
    return eee->run(args, outputs);
  }
  if (auto ke = dynamic_cast<KernelExecutor*>(executor.get())) {
    return ke->runFusion(args, LaunchParams(), CompileParams(), outputs);
  }
  NVF_THROW("Unsupported Executor detected.");
}

std::vector<at::Tensor> ExecutorDispatch::run(
    std::unique_ptr<ExecutorAbstract>& executor,
    KernelArgumentHolder& args,
    const LaunchParams& launch_constraints,
    CompileParams compile_params,
    std::vector<at::Tensor> outputs) {
  if (auto hire = dynamic_cast<HostIRExecutor*>(executor.get())) {
    return hire->run(args, outputs);
  }
  if (auto eee = dynamic_cast<ExprEvalExecutor*>(executor.get())) {
    return eee->run(args, outputs);
  }
  if (auto ke = dynamic_cast<KernelExecutor*>(executor.get())) {
    return ke->runFusion(args, launch_constraints, compile_params, outputs);
  }
  NVF_THROW("Unsupported Executor detected.");
}

} // namespace nvfuser
