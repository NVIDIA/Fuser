// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <runtime/expr_eval_exec.h>

#include <fusion_profiler.h>
#include <instrumentation.h>

namespace nvfuser {

bool ExprEvalExecutor::supported(Fusion* fusion) {
  FUSER_PERF_SCOPE("ExprEvalExecutor::supported");
  return std::all_of(
      fusion->outputs().begin(), fusion->outputs().end(), [&fusion](Val* out) {
        return fusion->getOutputAlias(out).type == AllocationType::Evaluate;
      });
}

void ExprEvalExecutor::compile(Fusion* fusion) {
  FUSER_PERF_SCOPE("ExprEvalExecutor::compile");
  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id_).startCompile();
  }
  NVF_ERROR(
      supported(fusion),
      "ExprEvalExecutor does not support the Fusion provided.");
  fusion_ = std::make_unique<Fusion>(*fusion);
  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id_).stopCompile();
  }
}

bool ExprEvalExecutor::isCompiled() const {
  return fusion_ != nullptr;
}

std::vector<at::Tensor> ExprEvalExecutor::run(
    KernelArgumentHolder& args,
    std::vector<at::Tensor> outputs) {
  FUSER_PERF_SCOPE("ExprEvalExecutor::run");

  if (isProfilerEnabled()) {
    NVF_CHECK(
        group_id_ >= 0,
        "An invalid segment id is passed to FusionProfiler!:",
        group_id_);
    SegmentProfiler& sprof = FusionProfiler::segment(group_id_);
    sprof.inputBytesAccessed(computeBytes(args));
    sprof.scheduler(toString(SchedulerType::ExprEval));
    sprof.startKernel();
  }

  NVF_ERROR(fusion_, "Need to compile before you can run.");
  // Bind fusion inputs
  ExpressionEvaluator expr_eval;

  {
    FUSER_PERF_SCOPE("ExprEvalExecutor::bindInputs");
    expr_eval = executor_utils::bindInputs(args, fusion_.get());
  }
  {
    FUSER_PERF_SCOPE("ExprEvalExecutor::Eval");
    NVF_ERROR(
        outputs.empty(),
        "Fusion executor is using expression evaluator,",
        " and expects that the outputs are not populated, which they were.");
    if (outputs.empty()) {
      for (const auto& out_val : fusion_->outputs()) {
        auto out_tensor =
            expr_eval.evaluate(out_val->as<TensorView>()).as<at::Tensor>();
        expr_eval.bind(out_val, out_tensor);
        outputs.emplace_back(out_tensor);
      }
    }
  }
  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id_).stopKernel();
    FusionProfiler::segment(group_id_).setDevice(args.getDeviceIndex());
  }
  return outputs;
}

} // namespace nvfuser
