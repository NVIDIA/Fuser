// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/pass/unroll.h>

#include <device_lower/lower2device.h>
#include <device_lower/pass/misaligned_vectorization.h>
#include <device_lower/utils.h>
#include <expr_evaluator.h>
#include <index_compute.h>
#include <instrumentation.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <ops/arith.h>
#include <predicate_compute.h>

namespace nvfuser {

namespace {

// Provide a new for loop matching the one provided
kir::ForLoop* cloneLoopNest(const kir::ForLoop* for_loop) {
  const auto new_loop = IrBuilder::create<kir::ForLoop>(for_loop);
  for (auto expr : for_loop->body().exprs()) {
    if (auto nested_for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      expr = cloneLoopNest(nested_for_loop);
    }
    new_loop->body().push_back(expr);
  }
  return new_loop;
}

// Returns true if expr is an expression that initializes a reduction
// buffer.
bool isReductionInitExpr(const Expr* expr) {
  // False if its output isn't a TensorView
  if (!ir_utils::isTvOp(expr)) {
    return false;
  }
  // False if it doesn't have any reduction axis
  const auto out_tv = expr->outputs()[0]->as<TensorView>();
  if (!out_tv->domain()->hasReduction()) {
    return false;
  }
  // False if it has have TensorView inputs as initialization should
  // never use TensorViews
  const auto tv_filter_inp_view =
      ir_utils::filterByType<TensorView>(expr->inputs());
  if (tv_filter_inp_view.begin() != tv_filter_inp_view.end()) {
    return false;
  }
  return true;
}

} // namespace

void UnrollPass::registerReplace(Expr* reference, Expr* new_expr) {
  kir::ExprMutator::registerReplace(reference, new_expr);
  GpuLower::current()->propagateExprInfo(reference, new_expr);
}

void UnrollPass::dispatch(Expr* expr) {
  if (ir_utils::isTvOp(expr) && !ir_utils::isCpAsyncBulk(expr)) {
    // If tv op, predicate it
    const auto out_tv = ir_utils::getTvOutput(expr);
    const bool should_predicate = !for_loops_.empty() ||
        out_tv->getMemoryType() == MemoryType::Global ||
        out_tv->getMemoryType() == MemoryType::Shared;
    if (!should_predicate) {
      return;
    }

    auto thread_pred =
        GpuLower::current()->threadPredMap().getPredicate(out_tv);

    // If this expr is for initializing a reduction output tensor, the
    // thread predicate can be ignored if the tensor is not shared by
    // any of the predicated parallel types
    if (isReductionInitExpr(expr)) {
      if (out_tv->getMemoryType() == MemoryType::Local) {
        // Local is always private, so we can always ignore thread predicates
        thread_pred = GpuLower::current()->kernel()->trueVal();
      } else if (out_tv->getMemoryType() == MemoryType::Shared) {
        // In the case of Shared, we can only ignore BIDx predicates
        thread_pred = GpuLower::current()->threadPredMap().getPredicate(
            out_tv, ParallelTypeBitmap().setAllTID());
      } else {
        // In the case of Global, we cannot ignore any predicates at
        // all, so don't modify thread_pred. Just make sure no other
        // memory type shows up here.
        NVF_ERROR(
            out_tv->getMemoryType() == MemoryType::Global,
            "Unexpected memory type: ",
            out_tv->getMemoryType(),
            ", tensor: ",
            out_tv->toString());
      }
    }

    // When this expr is in an unswitched block, only attach the
    // thread predicate to the expr as thread predicates are not
    // grouped to the unswitch predicate.
    kir::Predicate* thread_pred_expr = nullptr;
    if (unswitched_loop_) {
      thread_pred_expr = IrBuilder::create<kir::Predicate>(thread_pred);
    }

    non_trivial_pred_found_ = true;

    Expr* expr_with_predicate = expr;

    // When a predicate needs to account for ShiftOp, it is currently
    // taken care by its own function.
    if (GpuLower::current()->haloInfo()->needsShiftPredicate(expr)) {
      expr_with_predicate = ShiftPredicateInserter::insert(
          expr, for_loops_, thread_pred, unswitched_loop_);
      if (expr_with_predicate != expr) {
        registerReplace(expr, expr_with_predicate);
      }
      return;
    }

    // Reduction may need a separate predicate for writes.
    if (!isReductionInitExpr(expr) && out_tv->domain()->hasReduction()) {
      const auto write_pred = unswitched_loop_
          ? thread_pred_expr
          : IrBuilder::create<kir::Predicate>(
                PredicateType::ReductionWrite, expr, thread_pred);
      expr_with_predicate = expr_with_predicate->withWritePredicate(write_pred);
    }

    // For expr calling a device func with block sync, don't create
    // if-then-else but pass the predicate to the device func
    if (lower_utils::hasBlockSync(expr, GpuLower::current()->threadPredMap())) {
      const auto pred = unswitched_loop_
          ? thread_pred_expr
          : IrBuilder::create<kir::Predicate>(
                PredicateType::Inline, expr, thread_pred);
      expr_with_predicate = expr_with_predicate->withPredicate(pred);
      registerReplace(expr, expr_with_predicate);
      return;
    }

    // Vectorized expressions should never use inline predicates
    kir::Predicate* pred = nullptr;
    if (!unswitched_loop_ &&
        std::any_of(
            for_loops_.begin(), for_loops_.end(), [](const kir::ForLoop* fl) {
              return fl->iter_domain()->getParallelType() ==
                  ParallelType::Vectorize;
            })) {
      pred = IrBuilder::create<kir::Predicate>(PredicateType::Vectorize);
    }

    if (pred == nullptr) {
      pred = unswitched_loop_ ? thread_pred_expr
                              : IrBuilder::create<kir::Predicate>(
                                    PredicateType::Inline, expr, thread_pred);
    }

    if (lower_utils::supportInlinePredicate(expr)) {
      expr_with_predicate = expr_with_predicate->withPredicate(pred);
      registerReplace(expr, expr_with_predicate);
      return;
    }

    // If we need a predicate, put expr inside an if then else
    kir::IfThenElse* inline_ite = IrBuilder::create<kir::IfThenElse>(pred);
    kir::ExprMutator::registerReplace(expr, inline_ite);
    if (expr != expr_with_predicate) {
      GpuLower::current()->propagateExprInfo(expr, expr_with_predicate);
    }
    inline_ite->thenBody().push_back(expr_with_predicate);
  } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
    handle(for_loop);
  } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
    kir::ExprMutator::handle(ite);
  }
}

// We should factor our actual predicate generation from unrolling but insering
// IR nodes "unroll_pred" or "inline_pred", then generate those later.
void UnrollPass::handle(kir::ForLoop* fl) {
  // Setup for loop scoping
  const bool is_unroll =
      fl->iter_domain()->getParallelType() == ParallelType::Unroll ||
      fl->iter_domain()->getParallelType() == ParallelType::Unswitch;

  // If we're not looking for an unroll loop, or didn't find one, process as
  // normal.
  if (!is_unroll || !look_for_unroll_) {
    for_loops_.push_back(fl);
    scope_.push_back(&fl->body());
    scope_exprs_.push_back(fl);

    // Make copy of exprs because we replace them inplace in fl
    const auto exprs_copy = fl->body().exprs();

    // Skip Misaligned Vectorization For-Loops here
    if (!containsAnyDirectChildMisalignedVectorize(fl)) {
      for (auto expr : exprs_copy) {
        dispatch(expr);
      }
    }

    for_loops_.pop_back();
    scope_.pop_back();
    scope_exprs_.pop_back();
    return;
  }

  auto unroll_pred = IrBuilder::create<kir::Predicate>(fl);

  kir::IfThenElse* unroll_ite = IrBuilder::create<kir::IfThenElse>(unroll_pred);

  // Get the loop nest for the unrolled path
  kir::ForLoop* unrolled_loop_nest = cloneLoopNest(fl);
  unroll_ite->thenBody().push_back(unrolled_loop_nest);

  // Thread predicates are not removed from the expressions. Visit
  // each expression to attach kir::Predicate.
  scope_.push_back(&unroll_ite->thenBody());
  scope_exprs_.push_back(unroll_ite);
  unswitched_loop_ = true;
  look_for_unroll_ = false;
  handle(unrolled_loop_nest);
  unswitched_loop_ = false;
  look_for_unroll_ = true;
  scope_.pop_back();
  scope_exprs_.pop_back();

  // Loop nest for inlined path
  kir::ForLoop* inlined_loop = cloneLoopNest(fl);

  // Add inline predicates for inlined loop nest
  scope_.push_back(&unroll_ite->elseBody());
  scope_exprs_.push_back(unroll_ite);
  look_for_unroll_ = false;
  non_trivial_pred_found_ = false;
  handle(inlined_loop);
  look_for_unroll_ = true;
  scope_.pop_back();
  scope_exprs_.pop_back();
  if (!non_trivial_pred_found_) {
    kir::ExprMutator::registerReplace(fl, inlined_loop);
  } else {
    if (!canOmitElseClause(fl)) {
      unroll_ite->elseBody().push_back(inlined_loop);
    }
    kir::ExprMutator::registerReplace(fl, unroll_ite);
  }
}

bool UnrollPass::canOmitElseClause(kir::ForLoop* fl) {
  std::vector<kir::ForLoop*> loops({fl});

  const auto& pred_map = GpuLower::current()->threadPredMap();

  std::unordered_set<Expr*> all_exprs_inside_loop_nest;
  std::unordered_set<Expr*> resize_exprs;

  while (!loops.empty()) {
    auto loop = loops.back();
    loops.pop_back();

    // If there's any expression that requires barrier
    // synchronization, the else part can't be omitted
    for (auto expr : loop->body().exprs()) {
      if (lower_utils::hasBlockSync(expr, pred_map)) {
        return false;
      }
      // Keep track of all expressions for additional check for
      // resizing expressions
      all_exprs_inside_loop_nest.insert(expr);
      if (std::any_of(
              expr->outputs().begin(), expr->outputs().end(), [](Val* output) {
                return output->isA<TensorView>() &&
                    ir_utils::hasResizedRfactor(output->as<TensorView>());
              })) {
        resize_exprs.insert(expr);
      }
    }
    // If the number of visits of the loop body per thread is one, the
    // unswitch predicate is sufficient.
    // When the loop stop is the same as the extent of its IterDomain,
    // the per-thread visit count is guaranteed to be one at most (see
    // CudaKernelGenerator::handle(kir::ForLoop*) as well. Also, when a
    // loop is vectorized (not misaligned), the count must be one at
    // most. Even if not parallelized nor vectoirzed, it is also
    // sufficient if the loop stop is in fact one.
    bool visit_once = false;
    auto id = loop->iter_domain();
    if ((id->isThread() && (loop->stop() == id->extent())) ||
        id->getParallelType() == ParallelType::Vectorize) {
      visit_once = true;
    }
    if (!visit_once) {
      if (loop->stop()->isConstInt() && loop->stop()->evaluateInt() == 1) {
        visit_once = true;
      }
    }

    // The visit count is not guaranteed to be one, so the else part
    // must be created.
    if (!visit_once) {
      return false;
    }

    // The unswitch predicate is sufficient for this loop. Proceed to
    // nested loops.
    for (auto nested_loop :
         ir_utils::filterByType<kir::ForLoop>(loop->body().exprs())) {
      loops.push_back(nested_loop);
    }
  }

  // If an expression generates a resized tensor and any of its
  // dependencies appears in the loop nest, the else clause cannot be
  // omitted. The tensors appearing before the resizing expression has
  // a different shape than the output of the resizing expression and
  // its subsequent consumers, so the unswitch predicates would
  // include the predicates for both sizes, which means the larger
  // tensors would still need the else clause.
  if (!resize_exprs.empty()) {
    std::unordered_set<Val*> resize_expr_inputs;
    std::transform(
        resize_exprs.begin(),
        resize_exprs.end(),
        std::inserter(resize_expr_inputs, resize_expr_inputs.begin()),
        [](Expr* resize_expr) { return resize_expr->input(0); });
    if (std::any_of(
            all_exprs_inside_loop_nest.begin(),
            all_exprs_inside_loop_nest.end(),
            [&](Expr* loop_expr) {
              return std::any_of(
                  loop_expr->outputs().begin(),
                  loop_expr->outputs().end(),
                  [&](Val* expr_output) {
                    return resize_expr_inputs.count(expr_output);
                  });
            })) {
      return false;
    }
  }

  return true;
}

UnrollPass::UnrollPass(const std::vector<Expr*>& exprs) {
  kir::ExprMutator::traverseAndInsert(exprs);
}

std::vector<Expr*> UnrollPass::runPass(
    Fusion* fusion,
    const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::UnrollPass::runPass");

  UnrollPass unroll_pass(exprs);
  return unroll_pass.exprs_;
}

} // namespace nvfuser
