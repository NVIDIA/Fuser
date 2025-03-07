// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <alias_analysis.h>
#include <ir/utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/expr_eval_sched.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>

namespace nvfuser {

namespace {
bool allOutputsArePointerArithmetics(Fusion* fusion) {
  const AliasAnalysisResult analysis =
      findAliases(fusion, /*can_override_empty_allocation_domain=*/false);
  auto out_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
  return std::all_of(out_tvs.begin(), out_tvs.end(), [&](TensorView* out) {
    // Check out has an alias and out is not an inplace update target.
    if (fusion->getOutputAlias(out).type == AllocationType::ReuseBuffer) {
      return false;
    }

    TensorView* root = analysis.getRoot(out);
    return root != nullptr && root->isFusionInput();
  });
}

bool isNoOp(Expr* expr) {
  if (expr->isA<LoadStoreOp>() &&
      (expr->as<LoadStoreOp>()->opType() == LoadStoreOpType::Set ||
       expr->as<LoadStoreOp>()->opType() == LoadStoreOpType::SegmenterSet)) {
    return true;
  }
  if (ir_utils::isReductionOp(expr)) {
    for (auto out_tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
      const std::vector<IterDomain*>& logical_dom =
          TensorDomain::noReductions(out_tv->getLogicalDomain());
      const bool non_zero_reduction = std::any_of(
          logical_dom.begin(), logical_dom.end(), [](IterDomain* id) {
            return !(
                id->extent()->isConstScalar() &&
                id->extent()->evaluate().as<int64_t>() == 0);
          });
      if (non_zero_reduction) {
        return false;
      }
    }
    return true;
  }
  if (expr->isOneOf<
          SqueezeOp,
          BroadcastOp,
          SliceOp,
          CatOp,
          ViewOp,
          RepeatOp>()) {
    return true;
  }
  return false;
}
} // namespace

// Check if the fusion has a single MatmulOp/LinearOp node
bool ExprEvalScheduler::canScheduleCompileTime(Fusion* fusion) {
  if (scheduler_utils::isResharding(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Fusion is resharding.");
    return false;
  }

  if (allOutputsArePointerArithmetics(fusion)) {
    return true;
  }

  auto expr_check = [](Expr* expr) {
    return expr->isOneOf<SdpaFwdOp, SdpaBwdOp, EmbeddingFwdOp, GetMetaData>() ||
        (expr->isOneOf<LinearOp, MatmulOp>() &&
         !isOptionDisabled(DisableOption::MatmulExprEval)) ||
        ir_utils::isScalarOp(expr) || isNoOp(expr);
  };

  auto exprs = fusion->exprs();

  for (auto expr : exprs) {
    if (!expr_check(expr)) {
      scheduler_debug_utils::canScheduleRejectReason(
          "Expr not supported in ExprEvalScheduler:", expr->toString());
      return false;
    }
  }

  return true;
}

void ExprEvalScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  NVF_ERROR(
      params->scheduler_type == schedulerType(),
      "Invalid heuristic sent to ExprEval scheduler: ",
      params);

  std::for_each(
      fusion->outputs().begin(), fusion->outputs().end(), [&](Val* out) {
        fusion->aliasOutputToInput(
            out, /*input=*/nullptr, AllocationType::Evaluate);
      });
}

std::unique_ptr<HeuristicParams> ExprEvalScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  auto params = std::make_unique<HeuristicParams>(SchedulerType::ExprEval);
  params->cparams.index_type = runtime_info.getIndexType();
  return params;
}

} // namespace nvfuser
