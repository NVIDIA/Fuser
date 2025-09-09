// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <runtime/executor_dispatch.h>

#include <instrumentation.h>
#include <runtime/communication_executor.h>
#include <runtime/cutlass_executor.h>
#include <runtime/executor.h>

namespace nvfuser {

std::unique_ptr<ExecutorAbstract> ExecutorDispatch::makeExecutor(
    Fusion* fusion,
    int64_t fusion_id,
    int64_t concrete_id,
    int64_t runtime_id,
    int64_t group_id,
    SchedulerType scheduler_type) {
  FUSER_PERF_SCOPE("ExecutorDispatch::makeExecutor");
  if (scheduler_type == SchedulerType::None) {
    if (CommunicationExecutor::supported(fusion)) {
      return std::make_unique<CommunicationExecutor>(
          fusion_id, concrete_id, runtime_id, group_id);
    }
    if (ExprEvalExecutor::supported(fusion)) {
      return std::make_unique<ExprEvalExecutor>(
          fusion_id, concrete_id, runtime_id, group_id);
    }
    if (CutlassExecutor::supported(fusion)) {
      return std::make_unique<CutlassExecutor>(
          fusion_id, concrete_id, runtime_id, group_id);
    }
    if (KernelExecutor::supported(fusion)) {
      return std::make_unique<KernelExecutor>(
          fusion_id, concrete_id, runtime_id, group_id);
    }
    NVF_THROW("No executor supports provided fusion.");
  }

  switch (scheduler_type) {
    case SchedulerType::Communication:
      return std::make_unique<CommunicationExecutor>(
          fusion_id, concrete_id, runtime_id, group_id);
    case SchedulerType::ExprEval:
      return std::make_unique<ExprEvalExecutor>(
          fusion_id, concrete_id, runtime_id, group_id);
    case SchedulerType::Cutlass:
      return std::make_unique<CutlassExecutor>(
          fusion_id, concrete_id, runtime_id, group_id);
    default:
      return std::make_unique<KernelExecutor>(
          fusion_id, concrete_id, runtime_id, group_id);
  };
}

void ExecutorDispatch::compile(ExecutorAbstract* executor, Fusion* fusion) {
  FUSER_PERF_SCOPE("ExecutorDispatch::compile");
  if (auto ce = dynamic_cast<CommunicationExecutor*>(executor)) {
    ce->compile(fusion);
    return;
  }
  if (auto eee = dynamic_cast<ExprEvalExecutor*>(executor)) {
    eee->compile(fusion);
    return;
  }
  if (dynamic_cast<CutlassExecutor*>(executor) != nullptr) {
    NVF_THROW(
        "CutlassExecutor needs more information to be provided for "
        "compilation.");
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
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("ExecutorDispatch::compile2");

  if (auto ce = dynamic_cast<CommunicationExecutor*>(executor)) {
    ce->compile(fusion);
    return;
  }
  if (auto eee = dynamic_cast<ExprEvalExecutor*>(executor)) {
    eee->compile(fusion);
    return;
  }
  if (auto ce = dynamic_cast<CutlassExecutor*>(executor)) {
    const auto* cutlass_params = dynamic_cast<const CutlassParams*>(params);
    NVF_ERROR(
        cutlass_params != nullptr,
        "Expected CutlassParams for CutlassExecutor");
    ce->compile(fusion, *cutlass_params);
    return;
  }
  if (auto ke = dynamic_cast<KernelExecutor*>(executor)) {
    ke->compile(
        fusion, args, params->lparams, params->cparams, params->scheduler_type);
    return;
  }
  NVF_THROW("Unsupported Executor detected.");
}

bool ExecutorDispatch::isCompiled(const ExecutorAbstract* executor) {
  if (!executor) {
    return false;
  }
  FUSER_PERF_SCOPE("ExecutorDispatch::isCompiled");
  if (auto ce = dynamic_cast<const CommunicationExecutor*>(executor)) {
    return ce->isCompiled();
  }
  if (auto eee = dynamic_cast<const ExprEvalExecutor*>(executor)) {
    return eee->isCompiled();
  }
  if (auto ce = dynamic_cast<const CutlassExecutor*>(executor)) {
    return ce->isCompiled();
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
  FUSER_PERF_SCOPE("ExecutorDispatch::run");
  if (auto ce = dynamic_cast<CommunicationExecutor*>(executor)) {
    return ce->run(args, outputs);
  }
  if (auto eee = dynamic_cast<ExprEvalExecutor*>(executor)) {
    return eee->run(args, outputs);
  }
  if (auto ce = dynamic_cast<CutlassExecutor*>(executor)) {
    return ce->run(args, outputs);
  }
  if (auto ke = dynamic_cast<KernelExecutor*>(executor)) {
    return ke->run(args, outputs, launch_constraints, compile_params);
  }
  NVF_THROW("Unsupported Executor detected.");
}

} // namespace nvfuser
