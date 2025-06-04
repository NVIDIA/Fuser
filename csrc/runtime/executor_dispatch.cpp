// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <runtime/executor_dispatch.h>

#include <host_ir/executor.h>
#include <instrumentation.h>

#include <typeinfo>

namespace nvfuser {

// Iterates through executors in priority order creating the first executor that
// returns true when checking their "supported" method
std::unique_ptr<ExecutorAbstract> ExecutorDispatch::makeExecutor(
    Fusion* fusion,
    int64_t fusion_id,
    int64_t concrete_id,
    int64_t runtime_id,
    int64_t group_id) {
  FUSER_PERF_SCOPE("ExecutorDispatch::makeExecutor");
  if (HostIrExecutor::supported(fusion)) {
    return std::make_unique<HostIrExecutor>(
        fusion_id, concrete_id, runtime_id, group_id);
  }
  if (ExprEvalExecutor::supported(fusion)) {
    return std::make_unique<ExprEvalExecutor>(
        fusion_id, concrete_id, runtime_id, group_id);
  }
  if (KernelExecutor::supported(fusion)) {
    return std::make_unique<KernelExecutor>(
        fusion_id, concrete_id, runtime_id, group_id);
  }
  NVF_THROW("No executor supports provided fusion.");
}

void ExecutorDispatch::compile(ExecutorAbstract* executor, Fusion* fusion) {
  FUSER_PERF_SCOPE("ExecutorDispatch::compile");
  if (auto hire = dynamic_cast<HostIrExecutor*>(executor)) {
    hire->compile(fusion);
    return;
  }
  if (auto eee = dynamic_cast<ExprEvalExecutor*>(executor)) {
    eee->compile(fusion);
    return;
  }
  if (dynamic_cast<KernelExecutor*>(executor) != nullptr) {
    NVF_THROW(
        "KernelExecutor needs more information to be provided for "
        "compilation.");
  }
  NVF_THROW("Unsupported Executor detected.");
}

void ExecutorDispatch::compile(
    ExecutorAbstract* executor,
    Fusion* fusion,
    const KernelArgumentHolder& args,
    const LaunchParams& launch_constraints,
    CompileParams compile_params,
    SchedulerType scheduler_type) {
  FUSER_PERF_SCOPE("ExecutorDispatch::compile2");

  if (auto hire = dynamic_cast<HostIrExecutor*>(executor)) {
    hire->compile(fusion);
    return;
  }
  if (auto eee = dynamic_cast<ExprEvalExecutor*>(executor)) {
    eee->compile(fusion);
    return;
  }
  if (auto ke = dynamic_cast<KernelExecutor*>(executor)) {
    ke->compile(
        fusion, args, launch_constraints, compile_params, scheduler_type);
    return;
  }
  NVF_THROW("Unsupported Executor detected.");
}

bool ExecutorDispatch::isCompiled(const ExecutorAbstract* executor) {
  if (!executor) {
    return false;
  }
  FUSER_PERF_SCOPE("ExecutorDispatch::isCompiled");
  if (auto hire = dynamic_cast<const HostIrExecutor*>(executor)) {
    return hire->isCompiled();
  }
  if (auto eee = dynamic_cast<const ExprEvalExecutor*>(executor)) {
    return eee->isCompiled();
  }
  if (auto ke = dynamic_cast<const KernelExecutor*>(executor)) {
    return ke->isCompiled();
  }
  NVF_THROW("Unsupported Executor detected.");
}

KernelArgumentHolder ExecutorDispatch::run(
    ExecutorAbstract* executor,
    const KernelArgumentHolder& args,
    KernelArgumentHolder outputs,
    const LaunchParams& launch_constraints,
    const CompileParams& compile_params) {
  FUSER_PERF_SCOPE("ExecutorDispatch::run2");
  if (auto hire = dynamic_cast<HostIrExecutor*>(executor)) {
    return hire->run(args, outputs);
  }
  if (auto eee = dynamic_cast<ExprEvalExecutor*>(executor)) {
    return eee->run(args, outputs);
  }
  if (auto ke = dynamic_cast<KernelExecutor*>(executor)) {
    return ke->run(args, outputs, launch_constraints, compile_params);
  }
  NVF_THROW("Unsupported Executor detected.");
}

} // namespace nvfuser
