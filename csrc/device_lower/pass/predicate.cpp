// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/pass/predicate.h>

#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <index_compute.h>
#include <instrumentation.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <ops/arith.h>
#include <predicate_compute.h>
#include <transform_iter.h>
#include <transform_replay.h>

namespace nvfuser {

namespace {

class ConditionalFromPredicateModifier : public kir::ExprMutator {
 public:
  ConditionalFromPredicateModifier() = delete;

  static std::vector<Expr*> fillPredicates(const std::vector<Expr*>& exprs) {
    ConditionalFromPredicateModifier cfpm(exprs);
    return cfpm.exprs_;
  }

 private:
  ConditionalFromPredicateModifier(const std::vector<Expr*>& exprs) {
    FUSER_PERF_SCOPE(
        "ConditionalFromPredicateModifier::ConditionalFromPredicateModifier");
    traverseAndInsert(exprs);
  }

  using kir::ExprMutator::handle;

  void dispatch(Expr* expr) final {
    if (expr != nullptr && expr->predicate() != nullptr) {
      // Replace expr predicate with bool conditional
      auto conditional = generateConditional(expr->predicate());
      if (expr->predicate()->predicate_type() == PredicateType::Vectorize) {
        if (expr->isA<kir::IfThenElse>()) {
          // TODO: This logic doesn't seem to fit well here, for unswitch the
          // logic is in the unroll loop to set the thread predicate to the
          // expr. I didn't have a quick way to do that so placing this here for
          // now.
          auto ite = expr->as<kir::IfThenElse>();

          NVF_ERROR(
              ite->thenBody().size() == 1,
              "Expecting predicated body to only have one vectorized expression.");
          auto vec_expr = ite->thenBody()[0];
          NVF_ERROR(
              vec_expr->isA<UnaryOp>() || vec_expr->isA<LoadStoreOp>(),
              "Vectorize predicate exprs only supported on set operations.");
          NVF_ERROR(
              ir_utils::isTvOp(vec_expr),
              "Vectorize predicate exprs only supported on tensor view operations.");
          if (!vec_expr->inputs()[0]->isConstScalar()) {
            conditional = SimplifyingIrBuilder::logicalAndExpr(
                conditional,
                GpuLower::current()->threadPredMap().getPredicate(
                    ir_utils::getTvOutput(vec_expr)));
          }
        } else {
          NVF_ERROR(lower_utils::supportInlinePredicate(expr));
          auto thread_pred = GpuLower::current()->threadPredMap().getPredicate(
              ir_utils::getTvOutput(expr));
          NVF_ERROR(thread_pred->isConst() && thread_pred->value());
          conditional = SimplifyingIrBuilder::logicalAndExpr(
              conditional,
              GpuLower::current()->threadPredMap().getPredicate(
                  ir_utils::getTvOutput(expr)));
        }
      }
      NVF_ERROR(conditional != nullptr);
      conditional = GpuLower::current()->commonScalarMap().hoistScalar(
          conditional, for_loops_);
      expr->predicate()->setValue(conditional);
      NVF_ERROR(expr->predicate()->value() != nullptr);
      setWritePredicate(expr);
    }

    // According to:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async
    // cp.async has a built-in mechanism `ignore-src` to ignore the source and
    // fill zero. We can just invert the predicate and use it as `ignore-src`.
    if (ir_utils::isCpAsyncOp(expr)) {
      invertPredicate(expr);
    }

    kir::ExprMutator::dispatch(expr);
  }

  // Invert the predicate of given expr.
  void invertPredicate(Expr* expr) {
    NVF_ERROR(expr != nullptr);
    auto pred = expr->predicate()->value();
    Val* invert = SimplifyingIrBuilder::logicalNotExpr(pred);
    invert =
        GpuLower::current()->commonScalarMap().hoistScalar(invert, for_loops_);
    expr->predicate()->setValue(invert);
  }

  void setWritePredicate(Expr* expr) {
    if (expr->writePredicate() != nullptr) {
      auto write_cond = generateConditional(expr->writePredicate());
      if (write_cond) {
        write_cond = GpuLower::current()->commonScalarMap().hoistScalar(
            write_cond, for_loops_);
        expr->writePredicate()->setValue(write_cond);
      } else {
        // If generateConditional returns null, it means no specific
        // predicate needs to be used.
        registerReplace(expr, expr->withWritePredicate(nullptr));
      }
    }
  }

  void handle(kir::IfThenElse* ite) final {
    NVF_ERROR(ite->predicate() != nullptr);

    // Loop rotation transform loops like
    //  for i ...
    //    statement1(i)
    //    statement2(i)
    //    statement3(i)
    //    statement4(i)
    // into
    //  statement1(0)
    //  statement2(0)
    //  for i ...
    //    statement3(i)
    //    statement4(i)
    //    if LoopRotation:
    //      statement1(i+1)
    //      statement2(i+1)
    // So when we see an `if LoopRotation` during visiting, the last loop is
    // rotated, and we need to use `i+1` instead of `i` as loop index.
    if (ite->predicate()->predicate_type() == PredicateType::LoopRotation) {
      rotated_loop_.insert(for_loops_.back());
    }

    // If ite already has Bool conditional, handle internal expressions
    // Otherwise, generate conditional and update predicate
    if (!ite->predicate()->hasValue()) {
      auto conditional = generateConditional(ite->predicate());
      NVF_ERROR(conditional != nullptr);
      conditional = GpuLower::current()->commonScalarMap().hoistScalar(
          conditional, for_loops_);

      // Update bool conditional in-place
      ite->predicate()->setValue(conditional);
      NVF_ERROR(ite->predicate()->value() != nullptr);
    }
    kir::ExprMutator::handle(ite);

    if (ite->predicate()->predicate_type() == PredicateType::LoopRotation) {
      rotated_loop_.erase(for_loops_.back());
    }
  }

  // Generate conditional according to PredicateType
  Val* generateConditional(kir::Predicate* pred) {
    switch (pred->predicate_type()) {
      case PredicateType::Inline:
      case PredicateType::ReductionWrite:
      case PredicateType::Misaligned:
      case PredicateType::Shift:
      case PredicateType::Padding: {
        return PredicateCompute::getInlinePredicate(
            pred->expr(),
            for_loops_,
            rotated_loop_,
            pred->thread_pred(),
            pred->predicate_type());
      }
      case PredicateType::Vectorize: {
        std::vector<kir::ForLoop*> outer_loops;
        kir::ForLoop* vectorized_loop = nullptr;
        for (auto loop : for_loops_) {
          if (loop->iter_domain()->getParallelType() ==
              ParallelType::Vectorize) {
            vectorized_loop = loop;
            break;
          } else {
            outer_loops.emplace_back(loop);
          }
        }
        NVF_ERROR(vectorized_loop != nullptr, "Should be unreachable.");
        return UnswitchPredicate::get(outer_loops, vectorized_loop);
      }
      case PredicateType::Unswitch: {
        return UnswitchPredicate::get(for_loops_, pred->unrolled_loop());
      }
      case PredicateType::Manual: {
        return pred->value();
      }
      case PredicateType::LoopRotation: {
        // Currently, all existing predicates should be able to cover the
        // condition of loop_index + step < end, so nothing to do here. In the
        // future, if we decide that we need to predicate this then we can do it
        // here.
        return IrBuilder::create<Val>(true, DataType::Bool);
      }
      default:
        break;
    }
    return nullptr;
  }

  // Keep track of the loop in which the currently visiting expr is a rotated.
  std::unordered_set<kir::ForLoop*> rotated_loop_;
};

} // namespace

std::vector<Expr*> generateConditionalFromPredicate(
    const std::vector<Expr*>& exprs) {
  return ConditionalFromPredicateModifier::fillPredicates(exprs);
}

} // namespace nvfuser
