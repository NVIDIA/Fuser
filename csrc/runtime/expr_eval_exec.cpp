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

#include <cuda_profiler_api.h>

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
  exprs_ = fusion_->exprs();
  for (auto expr : exprs_) {
    if (expr->isA<ViewOp>()) {
      compile(expr->as<ViewOp>());
    } else if (expr->isA<LoadStoreOp>()) {
      compile(expr->as<LoadStoreOp>());
    }
  }
  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id_).stopCompile();
  }
  cudaProfilerStart();
}

bool ExprEvalExecutor::isCompiled() const {
  return fusion_ != nullptr;
}

std::vector<at::Tensor> ExprEvalExecutor::run(
    KernelArgumentHolder& args,
    std::vector<at::Tensor> outputs) {
  FUSER_PERF_SCOPE("ExprEvalExecutor::run");

  NVF_ERROR(
      outputs.empty(),
      "Fusion executor is using expression evaluator,",
      " and expects that the outputs are not populated, which they were.");

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

    for (auto expr : exprs_) {
      if (ViewOp* view = dynamic_cast<ViewOp*>(expr)) {
        auto output_tensor =
            run(view, expr_eval.evaluate(view->in()).as<at::Tensor>());
        expr_eval.bind(view->out(), output_tensor);
        continue;
      } else if (LoadStoreOp* ld_st_op = dynamic_cast<LoadStoreOp*>(expr)) {
        auto output_tensor =
            run(ld_st_op, expr_eval.evaluate(ld_st_op->in()).as<at::Tensor>());
        expr_eval.bind(ld_st_op->out(), output_tensor);
        continue;
      }
      expr_eval.evaluate(expr->outputs()[0]);
    }

    for (const auto& out_val : fusion_->outputs()) {
      auto out_tensor = expr_eval.evaluate(out_val).as<at::Tensor>();
      // expr_eval.bind(out_val, out_tensor);
      outputs.emplace_back(out_tensor);
    }
  }
  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id_).stopKernel();
    FusionProfiler::segment(group_id_).setDevice(args.getDeviceIndex());
  }
  return outputs;
}

namespace {
bool isContiguous(TensorView* tv) {
  auto logical = TensorDomain::noReductions(tv->getLogicalDomain());
  auto alloc = TensorDomain::noReductions(tv->getMaybeAllocationDomain());
  if (logical.size() != alloc.size()) {
    return false;
  }
  for (int64_t id_i : c10::irange(logical.size())) {
    if (logical[id_i]->isBroadcast() && alloc[id_i]->isBroadcast()) {
      if (logical[id_i]->hasExpandedExtent()) {
        return false;
      }
      continue;
    }
    if (logical[id_i] != alloc[id_i]) {
      return false;
    }
    if (!tv->getContiguity()[id_i]) {
      return false;
    }
  }
  return true;
}
} // namespace

void ExprEvalExecutor::compile(ViewOp* view_op) {
  FUSER_PERF_SCOPE("ExprEvalExecutor::compile(ViewOp* view_op");
  std::vector<int64_t> sizes;
  bool neg_1_found = false;
  for (auto id : view_op->out()->getLogicalDomain()) {
    // Ignore sharded dimensions
    if (id->isDeviceDim()) {
      sizes.push_back(1);
      continue;
    }

    // Constant reshape specified dimensions
    auto id_size = id->getMaybeExpandedExtent();
    if (id_size->isConstInt()) {
      sizes.push_back(id_size->evaluate().as<int64_t>());
      continue;
    }

    NVF_ERROR(
        !neg_1_found,
        "Invalid reshape op found, more than one unknown dimensions size specified.");

    // Only one free variable allowed
    sizes.push_back(-1);
    neg_1_found = true;
  }
  output_view_sizes[view_op] = sizes;

  use_view[view_op] = isContiguous(view_op->in());
}

at::Tensor ExprEvalExecutor::run(ViewOp* view_op, at::Tensor input) {
  FUSER_PERF_SCOPE("ExprEvalExecutor::run(ViewOp* view_op");
  if (use_view[view_op]) {
    return input.view(output_view_sizes[view_op]);
  }
  return input.reshape(output_view_sizes[view_op]);
}

void ExprEvalExecutor::compile(LoadStoreOp* ld_st_op) {
  FUSER_PERF_SCOPE("ExprEvalExecutor::compile(LoadStoreOp* ld_st_op");
  if (TensorView* out_tv = dynamic_cast<TensorView*>(ld_st_op->out())) {
    if (out_tv->hasRoot()) {
      std::optional<std::vector<int64_t>> permutation =
          ir_utils::computePermutation(
              out_tv->getRootDomain(), out_tv->getLogicalDomain());
      NVF_ERROR(
          permutation.has_value(),
          "The logical domain of a Set.Permute is supposed to be a permutation of the root domain: ",
          out_tv->toString());
      permutation_orders[ld_st_op] = *permutation;
    }
  }
}

at::Tensor ExprEvalExecutor::run(LoadStoreOp* ld_st_op, at::Tensor input) {
  FUSER_PERF_SCOPE("ExprEvalExecutor::run(LoadStoreOp* ld_st_op");
  auto permute_it = permutation_orders.find(ld_st_op);
  if (permute_it == permutation_orders.end()) {
    return input;
  }
  return input.permute(permute_it->second);
}

} // namespace nvfuser
