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

      // When current elect sync predicates a UBLK TMA load, it also need to
      // predicate the corresponding MBarrierArriveExpectTx and
      // MBarrierWaitParity. Manually hoist the inline predicate to merge with
      // elect sync to avoid deadlock.
      if (current_elect_sync_ && reviseUblkPredicate(expr, conditional)) {
        conditional = GpuLower::current()->kernel()->trueVal();
      }

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
              vec_expr->isA<UnaryOp>() || vec_expr->isA<LoadStoreOp>() ||
                  vec_expr->isA<TernaryOp>() || vec_expr->isA<IndexSelectOp>(),
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
    } else if (
        expr && expr->isA<kir::MBarrierWaitParity>() && ublk_predicate_val_) {
      auto fl = for_loops_.front();
      std::unordered_map<Val*, Val*> replace_map;
      replace_map[tma_branch_fl_index_] = fl->index();
      auto local_predicate_val =
          ir_utils::replaceValRecursively(ublk_predicate_val_, replace_map);
      local_predicate_val = GpuLower::current()->commonScalarMap().hoistScalar(
          local_predicate_val, {for_loops_.front()});
      kir::Predicate* pred =
          IrBuilder::create<kir::Predicate>(local_predicate_val);
      kir::IfThenElse* inline_ite = IrBuilder::create<kir::IfThenElse>(pred);
      kir::ExprMutator::registerReplace(expr, inline_ite);
      inline_ite->thenBody().push_back(expr);
    }

    kir::ExprMutator::dispatch(expr);
  }

  // Move Inline predicate for UBLK TMA load to ElectSync
  // (1) UBLK TMA load can't handle out-of-bound access, it has Inline
  //     predicates added in Unroll pass.
  // (2) TMA loads are synced using MBarrierArriveExpectTx and
  //     MBarrierWaitParity, they need the same predicate to avoid deadlock.
  bool reviseUblkPredicate(Expr* expr, Val* conditional) {
    // Looking for the following pattern:
    // IF Inline
    //   Ts = UBLK TMA load
    if (auto ite = expr->as<kir::IfThenElse>()) {
      if (ite->predicate()->predicate_type() == PredicateType::Inline) {
        for (auto lexpr : ite->thenBody().exprs()) {
          if (ir_utils::isCpAsyncUblk(lexpr)) {
            if (for_loops_.size() > 1) {
              std::unordered_map<Val*, Val*> replace_map;
              for (auto it = for_loops_.begin() + 1; it != for_loops_.end();
                   it++) {
                replace_map[(*it)->index()] =
                    GpuLower::current()->kernel()->zeroVal();
              }
              ublk_predicate_val_ =
                  ir_utils::replaceValRecursively(conditional, replace_map);
            } else {
              ublk_predicate_val_ = conditional;
            }
            // Combine the Inline predicate with the ElectSync predicate
            auto combined_conditional = SimplifyingIrBuilder::logicalAndExpr(
                current_elect_sync_->predicate()->value(), ublk_predicate_val_);
            combined_conditional =
                GpuLower::current()->commonScalarMap().hoistScalar(
                    combined_conditional, {for_loops_.front()});
            current_elect_sync_->predicate()->setValue(combined_conditional);
            // Keep the index of the outermost for-loop, will be replaced in the
            // predicate of MBarrierWaitParity.
            tma_branch_fl_index_ = for_loops_.front()->index();
            return true;
          }
        }
      }
    }
    return false;
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
    if (ite->predicate()->predicate_type() == PredicateType::ElectSync) {
      current_elect_sync_ = ite;
    }
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

    if (ite->predicate()->predicate_type() == PredicateType::ElectSync) {
      current_elect_sync_ = nullptr;
    }

    if (ite->predicate()->predicate_type() == PredicateType::LoopRotation) {
      rotated_loop_.erase(for_loops_.back());
    }
  }

  // Generate conditional according to PredicateType
  Val* generateConditional(kir::Predicate* pred) {
    switch (pred->predicate_type()) {
      case PredicateType::Inline:
      case PredicateType::ReductionWrite:
      case PredicateType::Misaligned: {
        return PredicateCompute::getInlinePredicate(
            pred->expr(),
            for_loops_,
            rotated_loop_,
            pred->thread_pred(),
            pred->predicate_type());
      }
      case PredicateType::Vectorize: {
        std::vector<ForLoop*> outer_loops;
        ForLoop* vectorized_loop = nullptr;
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
        // future, if we decide that we need to predicate this then we can do
        // it here.
        return IrBuilder::create<Val>(true, DataType::Bool);
      }
      case PredicateType::ElectSync: {
        return PredicateCompute::getElectSyncPredicate(pred, for_loops_);
      }
      default:
        break;
    }
    return nullptr;
  }

  // Keep track of the loop in which the currently visiting expr is a rotated.
  std::unordered_set<ForLoop*> rotated_loop_;

  kir::IfThenElse* current_elect_sync_ = nullptr;

  Val* ublk_predicate_val_ = nullptr;
  Val* tma_branch_fl_index_ = nullptr;
};

} // namespace

std::vector<Expr*> generateConditionalFromPredicate(
    const std::vector<Expr*>& exprs) {
  if (isDebugDumpEnabled(DebugDumpOption::PredicateElimination)) {
    debug() << GpuLower::current()->predicateElimination().toString()
            << std::endl;
  }
  return ConditionalFromPredicateModifier::fillPredicates(exprs);
}

} // namespace nvfuser
