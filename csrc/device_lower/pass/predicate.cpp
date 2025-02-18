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

// class UblkTmaFinder : kir::ConstIrVisitor {
//  public:
//   static Expr* get(const Expr* expr) {
//     NVF_CHECK(expr->container()->isA<kir::Kernel>());
//     UblkTmaFinder finder;
//     finder.handle(std::vector<const Expr*>{expr});
//     return finder.ublk_tma_load_;
//   }

//  private:
//   using kir::ConstIrVisitor::handle;

//   void dispatch(const Expr* expr) final {
//     if (expr->isA<kir::MBarrierArriveExpectTx>()) {
//       found_arrive_expect_ = true;
//     }
//     if (found_arrive_expect_ && ir_utils::isCpAsyncUblk(expr)) {
//       ublk_tma_load_ = const_cast<Expr*>(expr);
//       return;
//     }
//     kir::ConstIrVisitor::dispatch(expr);
//   }

//  private:
//   bool found_arrive_expect_ = false;
//   Expr* ublk_tma_load_ = nullptr;
// };
// Expr* getUblkTmaLoad(const Expr* expr) {
//   return UblkTmaFinder::get(expr);
// }
// return the ublk tma load expr if the input expr is a ite with both arrive
// expect and tma load. For example, when the input expr is:
// IF ElectSync():
//   MBarrierArriveExpectTx()
//   IF ElectSync():
//     CpAsyncUblk()
// This function will return the CpAsyncUblk() expr.
// Extra inline predicate may be further added to this ublk tma load to avoid
// out-of-bound access. This ite code is then modified to:
// IF ElectSync() && inline predicate:
//   MBarrierArriveExpectTx()
//   CpAsyncUblk()

Expr* getUblkTmaLoad(Expr* ite_expr) {
  if(auto ite = dynamic_cast<kir::IfThenElse*>(ite_expr)){
    const auto& flattened_exprs = ir_utils::flattenScopedExprs(ite->thenBody().exprs());
    bool found_arrive_expect_ = false;
    for(auto expr : flattened_exprs) {
      if (expr->isA<kir::MBarrierArriveExpectTx>()) {
        found_arrive_expect_ = true;
      }
      if (found_arrive_expect_ && ir_utils::isCpAsyncUblk(expr)) {
        return expr;
      }
    }
  }
  return nullptr;
}

Expr* getUblkTmaLoadFromIte(Expr* ite_expr) {
  if(auto ite = dynamic_cast<kir::IfThenElse*>(ite_expr)){
    const auto& flattened_exprs = ir_utils::flattenScopedExprs(ite->thenBody().exprs());
    for(auto expr : flattened_exprs) {
      if (ir_utils::isCpAsyncUblk(expr)) {
        return expr;
      }
    }
  }
  return nullptr;
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
    std::cout << "\n======================= is_circular_buffer_main_loop_: " << is_circular_buffer_main_loop_ << std::endl;
    std::cout << " dispatch:\n" << expr->toString() << std::endl;
    if(auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      if (Expr* ublk_tma_load = getUblkTmaLoad(ite)) {
        // auto output = ir_utils::getTvOutput(ublk_tma_load);
        auto ldst = dynamic_cast<LoadStoreOp*>(ublk_tma_load);
        ublk_load_to_arrive_expect_ite.insert({ldst, ite});
        std::cout << "Add ublk tma load:\n" << ublk_tma_load->as<LoadStoreOp>()->toString()
                  << std::endl;
      }
    }

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

      if (true && ir_utils::isCpAsyncUblk(expr->predicate()->expr())) {
        predicateCpAsyncUblk(expr, conditional);
      } else {
        NVF_ERROR(conditional != nullptr);
        conditional = GpuLower::current()->commonScalarMap().hoistScalar(
            conditional, for_loops_);
        expr->predicate()->setValue(conditional);
        NVF_ERROR(expr->predicate()->value() != nullptr);
        setWritePredicate(expr);
      }
    }

    // may add extra predicate for wait parity to avoid deadlock
    if (is_circular_buffer_main_loop_ && expr->isA<kir::MBarrierWaitParity>()) {
      predicateUblkWaitParity(expr);
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

  // This function combines the original elect sync predicate with the inline
  // predicate to avoid out-of-bound access for the ublk tma load.
  void predicateCpAsyncUblk(Expr* ite_tma_expr, Val* elect_sync_pred) {
    auto ublk_tma_expr = getUblkTmaLoadFromIte(ite_tma_expr);
    auto ldst = dynamic_cast<LoadStoreOp*>(const_cast<Expr*>(ublk_tma_expr));
    // auto tma_tv = ldst->output(0)->as<TensorView>();
    if(ublk_load_to_arrive_expect_ite.find(ldst) == ublk_load_to_arrive_expect_ite.end()) {
      std::cout << "Cannot find ublk tma load for: " << ublk_tma_expr->toString() << std::endl;
      for(auto[tv, ite] : ublk_load_to_arrive_expect_ite) {
        std::cout << "\nublk tma load: " << tv->toString() << std::endl;
        std::cout << "ublk tma itte: " << ite->toString() << std::endl;
      }
      return;
    }
    std::cout << "Find ublk tma load for: " << ublk_tma_expr->toString() << std::endl;
    kir::IfThenElse* ite = ublk_load_to_arrive_expect_ite.at(ldst);
    // inline predicate to void out-of-bound access
    auto inline_pred_val = PredicateCompute::getInlinePredicate(
        ublk_tma_expr,
        for_loops_,
        rotated_loop_,
        ite_tma_expr->predicate()->thread_pred(),
        ite_tma_expr->predicate()->predicate_type());
    std::cout << "inline pred:" << inline_pred_val->toString() << std::endl;
    inline_pred_val = GpuLower::current()->commonScalarMap().hoistScalar(
        inline_pred_val, for_loops_);
    // combine inline predicate with the original elect sync predicate
    auto combined_pred_val =
        SimplifyingIrBuilder::logicalAndExpr(elect_sync_pred, inline_pred_val);
    combined_pred_val = GpuLower::current()->commonScalarMap().hoistScalar(
        combined_pred_val, for_loops_);
    // map the mbarrier used in this tma load to the extra inline predicate,
    // this is then used to predicate the mbarrier wait parity.
    kir::TensorIndex* mbarrier =
        GpuLower::current()->tmaCircularBufferInfo().getTensorIndex(
            ublk_tma_expr->as<LoadStoreOp>());
    if(mbarrier && is_circular_buffer_main_loop_){
      std::cout << "insert mbarrier:" << mbarrier->toString() << std::endl;
      tma_mbarrier_tv_to_inline_predicate_.insert({mbarrier->view(), inline_pred_val});
    }else{
      std::cout << "Cannot find mbarrier for: " << ublk_tma_expr->toString() << std::endl;
    }
    // Since tma load expr is nested in the ite, we only need to predicate the
    // ite with the combined predicate and remove the tma load predicate by set
    // it to true.
    ite->predicate()->setValue(combined_pred_val);
    ite_tma_expr->predicate()->setValue(
        IrBuilder::create<Val>(true, DataType::Bool));

    // remove this tma load from map
    ublk_load_to_arrive_expect_ite.erase(ldst);
    std::cout << "erase:\n" << ublk_tma_expr->toString()
              << std::endl;
  }

  // This function addes the inline predicate to the mbarrier wait parity to
  // avoid deadlock since its corresponding ublk tma load may be predicated with
  // an inline predicate to avoid out-of-bound access.
  void predicateUblkWaitParity(Expr* expr) {
    // find the tensor index used in the mbarrier
    auto wait_parity = dynamic_cast<kir::MBarrierWaitParity*>(expr);
    // don't need to predicate wait parity for computations
    if(!wait_parity->parity()->isConstScalar()){
      return;
    }
    auto mbarrier = wait_parity->mbarrier();
    kir::TensorIndex* tensor_index = nullptr;
    auto current_def = mbarrier->definition();
    while (current_def && current_def->isA<UnaryOp>()) {
      std::cout << "current def:\n" << current_def->toString() << std::endl;
      auto input = current_def->as<UnaryOp>()->in();
      if (input->isA<kir::TensorIndex>()) {
        tensor_index = input->as<kir::TensorIndex>();
        break;
      }
      current_def = input->definition();
    }
    NVF_CHECK(
        tensor_index,
        "Cannot find tensor index for mbarrier: ",
        mbarrier->toInlineString());

    // predicate this wait parity with the inline predicate used to predicate
    // the corresponding ublk tma load.
    auto mbarrier_tv = tensor_index->view();
    if (tma_mbarrier_tv_to_inline_predicate_.find(mbarrier_tv) !=
        tma_mbarrier_tv_to_inline_predicate_.end()) {
      auto pred_val = tma_mbarrier_tv_to_inline_predicate_.at(mbarrier_tv);
      kir::Predicate* pred = IrBuilder::create<kir::Predicate>(pred_val);
      kir::IfThenElse* inline_ite = IrBuilder::create<kir::IfThenElse>(pred);
      kir::ExprMutator::registerReplace(expr, inline_ite);
      inline_ite->thenBody().push_back(expr);
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

  void handle(ForLoop* for_loop) final {
      if(for_loop->circularBufferLoopStage() == CircularBufferLoopStage::Main) {
        is_circular_buffer_main_loop_ = true;
      }

      kir::ExprMutator::handle(for_loop);
      
      if(for_loop->circularBufferLoopStage() == CircularBufferLoopStage::Main) {
        is_circular_buffer_main_loop_ = false;
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
        return PredicateCompute::getElectSyncPredicate(pred, for_loops_);
      }
      default:
        break;
    }
    return nullptr;
  }

  bool is_circular_buffer_main_loop_ = false;
  bool is_circular_buffer_epilog_loop_ = false;

  // Keep track of the loop in which the currently visiting expr is a rotated.
  std::unordered_set<ForLoop*> rotated_loop_;

  // map from ublk tma load tensor view to the ite that contains arrive expt and
  // ublk tma load
  std::unordered_map<LoadStoreOp*, kir::IfThenElse*>
      ublk_load_to_arrive_expect_ite;

  struct TensorIndexHash {
    size_t operator()(const TensorView* ti) const {
      return std::hash<const TensorView*>()(ti);
      // return std::hash<const TensorView*>()(ti->view()) ^
      //     std::hash<int64_t>()(ti->index()->value().as<int64_t>());
    }
  };
  struct TensorIndexEqual {
    bool operator()(const TensorView* lhs, const TensorView* rhs)
        const {
      if (lhs == rhs) {
        return true;
      }
      return lhs->name() == rhs->name();
      // if(lhs->index()->isConstInt() && rhs->index()->isConstInt()){
      //   return lhs->view() == rhs->view() &&
      //       lhs->index()->value().as<int64_t>() ==
      //       rhs->index()->value().as<int64_t>();
      // }

      // return lhs->view() == rhs->view();     
      // return lhs->view() == rhs->view() &&
      //     lhs->index()->value().as<int64_t>() ==
      //     rhs->index()->value().as<int64_t>();
    }
  };
  // map from mbarrier (tensor index) to inline predicate val
  std::unordered_map<TensorView*, Val*, TensorIndexHash, TensorIndexEqual>
      tma_mbarrier_tv_to_inline_predicate_;
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
