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

class UblkTmaFinder : kir::ConstIrVisitor {
 public:
  static Expr* get(const Expr* expr) {
    NVF_CHECK(expr->container()->isA<kir::Kernel>());
    UblkTmaFinder finder;
    finder.handle(std::vector<const Expr*>{expr});
    return finder.ublk_tma_load_;
  }

 private:
  using kir::ConstIrVisitor::handle;

  void dispatch(const Expr* expr) final {
    if (expr->isA<kir::MBarrierArriveExpectTx>()) {
      found_arrive_expect_ = true;
    }
    if (found_arrive_expect_ && ir_utils::isCpAsyncUblk(expr)) {
      ublk_tma_load_ = const_cast<Expr*>(expr);
      return;
    }
    kir::ConstIrVisitor::dispatch(expr);
  }

 private:
  bool found_arrive_expect_ = false;
  Expr* ublk_tma_load_ = nullptr;
};

Expr* getUblkTmaLoad(const Expr* expr) {
  return UblkTmaFinder::get(expr);
}

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
      std::cout << "\n================== dispatch:\n"
                << expr->toString() << std::endl;    
    if (expr != nullptr && expr->predicate() != nullptr) {
      // Replace expr predicate with bool conditional
      auto conditional = generateConditional(expr->predicate());
      kir::IfThenElse* ite = nullptr;
      if (ir_utils::isCpAsyncUblk(expr->predicate()->expr())) {
        const auto tma_expr = expr->predicate()->expr();
        std::cout << "-----------tma_expr " << tma_expr->toString()
                  << std::endl;
        auto tma_tv = ir_utils::getTvOutput(tma_expr);
        ite = ublk_load_ite_extra_predicate_.at(tma_tv);
        ublk_load_ite_extra_predicate_.erase(tma_tv);
        auto inline_pred_val = PredicateCompute::getInlinePredicate(
            tma_expr,
            for_loops_,
            rotated_loop_,
            expr->predicate()->thread_pred(),
            expr->predicate()->predicate_type());
        inline_pred_val = GpuLower::current()->commonScalarMap().hoistScalar(
            inline_pred_val, for_loops_);
        kir::TensorIndex* mbarrier =
            GpuLower::current()->tmaCircularBufferInfo().getTensorIndex(
                tma_expr->as<LoadStoreOp>());
        insertPredicate(mbarrier, inline_pred_val);
        std::cout << "------------mbarrier " << mbarrier->toString()
                  << std::endl;
        std::cout << "------------expr inline_pred_val "
                  << inline_pred_val->toInlineString() << std::endl;
        conditional =
            SimplifyingIrBuilder::logicalAndExpr(conditional, inline_pred_val);
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
                  vec_expr->isA<TernaryOp>(),
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

      if (ite) {
        ite->predicate()->setValue(conditional);
        expr->predicate()->setValue(
            IrBuilder::create<Val>(true, DataType::Bool));
      } else {
        expr->predicate()->setValue(conditional);
        NVF_ERROR(expr->predicate()->value() != nullptr);
        setWritePredicate(expr);
      }
    }

    // mbarrier
    if (auto wait_parity = dynamic_cast<kir::MBarrierWaitParity*>(expr)) {
      auto mbarrier = wait_parity->mbarrier();
      kir::TensorIndex* tensor_index = nullptr;
      auto current_def = mbarrier->definition();
      while (current_def && current_def->isA<UnaryOp>()) {
        auto input = current_def->as<UnaryOp>()->in();
        if (input->isA<kir::TensorIndex>()) {
          tensor_index = input->as<kir::TensorIndex>();
          break;
        }
        current_def = input->definition();
      }
      if (tensor_index) {
        std::cout << "------------wait_parity mbarrier " << tensor_index->toString() << std::endl;
        if (auto pred_val = findPredicate(tensor_index)) {
          std::cout << "------------wait_parity mbarrier found "
                    << tensor_index->toString() << std::endl;
          kir::Predicate * pred = IrBuilder::create<kir::Predicate>(pred_val);
          kir::IfThenElse* inline_ite =
              IrBuilder::create<kir::IfThenElse>(pred);
          kir::ExprMutator::registerReplace(expr, inline_ite);
          inline_ite->thenBody().push_back(expr);
        } else {
          std::cout << "------------wait_parity mbarrier not found "
                    << tensor_index->toString() << std::endl;
          for(auto iter : mbarrier_inline_predicate_) {
            std::cout << "------------mbarrier_inline_predicate_ "
                      << iter.first->toString() << std::endl;
          }
        }
      }
    }

    kir::ExprMutator::dispatch(expr);
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

    if (Expr* ublk_tma_load = getUblkTmaLoad(ite)) {
      // ublk_load_ite_extra_predicate_.insert({ublk_tma_load, ite});
      auto output = ir_utils::getTvOutput(ublk_tma_load);
      ublk_load_ite_extra_predicate_.insert({output, ite});
      std::cout << "------------expr isUblkTmaLoad "
                << ublk_tma_load->toString() << std::endl;
      // for (auto fl : for_loops_) {
      //   std::cout << "\nfor_loops_:\n"
      //             << fl->toString() << std::endl
      //             << std::endl;
      // }
      // auto pred = ublk_tma_load->predicate();
      // auto inline_pred_val = PredicateCompute::getInlinePredicate(
      //     ublk_tma_load,
      //     for_loops_,
      //     rotated_loop_,
      //     pred->thread_pred(),
      //     pred->predicate_type());
      // std::cout << "------------expr inline_pred_val " <<
      // inline_pred_val->toInlineString() << std::endl;
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
  Val* generateConditional(
      kir::Predicate* pred,
      bool is_ublk_tma_load = false) {
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
        // future, if we decide that we need to predicate this then we can do it
        // here.
        return IrBuilder::create<Val>(true, DataType::Bool);
      }
      case PredicateType::ElectSync: {
        // std::cout << "\n\nPredicateType::ElectSync "<< std::endl;

        auto elect_pred_val =
            PredicateCompute::getElectSyncPredicate(pred, for_loops_);
        bool is_1d_tma = !ir_utils::isCpAsyncBulkTensorTile(pred->expr()) &&
            ir_utils::isCpAsyncBulk(pred->expr());
        bool is_arrive_expect_tx =
            pred->expr()->isA<kir::MBarrierArriveExpectTx>();
        if (false && (is_1d_tma || is_arrive_expect_tx)) {
          auto inline_pred_val = PredicateCompute::getInlinePredicate(
              pred->expr(),
              for_loops_,
              rotated_loop_,
              pred->thread_pred(),
              pred->predicate_type());
          return SimplifyingIrBuilder::logicalAndExpr(
              elect_pred_val, inline_pred_val);
        } else {
          return elect_pred_val;
        }
      }
      default:
        break;
    }
    return nullptr;
  }

  // Keep track of the loop in which the currently visiting expr is a rotated.
  std::unordered_set<ForLoop*> rotated_loop_;
  std::unordered_map<TensorView*, kir::IfThenElse*>
      ublk_load_ite_extra_predicate_;

  // // Hash function for kir::TensorIndex*
  // struct TensorIndexHash {
  //   size_t operator()(const kir::TensorIndex* ti) const {
  //     if (!ti)
  //       return 0;
  //     return std::hash<const TensorView*>()(ti->view()) ^
  //         std::hash<Val*>()(ti->index());
  //   }
  // };
  // // Equality function for kir::TensorIndex*
  // struct TensorIndexEqual {
  //   bool operator()(const kir::TensorIndex* lhs, const kir::TensorIndex* rhs)
  //       const {
  //     if (lhs == rhs)
  //       return true;
  //     if (!lhs || !rhs)
  //       return false;
  //     std::cout << "lhs->view() " << lhs->view() << " rhs->view() " << rhs->view() << std::endl;
  //     std::cout << "lhs->index() " << lhs->index() << " rhs->index() " << rhs->index() << std::endl;
  //     std::cout << (lhs->view() == rhs->view() && lhs->index() == rhs->index()) << std::endl;
  //     return lhs->view() == rhs->view() && lhs->index() == rhs->index();
  //   }
  // };
  // // Define the unordered_map using custom hash and equality functions
  // std::unordered_map<kir::TensorIndex*, Val*, TensorIndexHash, TensorIndexEqual>
  //     mbarrier_inline_predicate_;

  // Define a vector to store pairs of TensorIndex* and Val*
  std::vector<std::pair<kir::TensorIndex*, Val*>> mbarrier_inline_predicate_;

  // Helper function to find the corresponding Val* for a given TensorIndex*
  Val* findPredicate(const kir::TensorIndex* ti) {
      for (const auto& pair : mbarrier_inline_predicate_) {
          const auto& lhs = pair.first;
          if (lhs == ti) {
              return pair.second;
          }
          int64_t index_self = lhs->index()->value().as<int64_t>();
          int64_t index_other = ti->index()->value().as<int64_t>();
          if (lhs->view()->name() == ti->view()->name() && index_self == index_other) {
              return pair.second;
          }
      }
      return nullptr;
  }

  // Helper function to insert a TensorIndex*-Val* pair
  void insertPredicate(kir::TensorIndex* ti, Val* val) {
      // Check if the TensorIndex* already exists
      for (auto& pair : mbarrier_inline_predicate_) {
          if (pair.first == ti || (pair.first && ti && pair.first->view()->name() == ti->view()->name() && pair.first->index() == ti->index())) {
              pair.second = val;
              return;
          }
      }
      // If not found, add a new pair
      mbarrier_inline_predicate_.emplace_back(ti, val);
  }

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
