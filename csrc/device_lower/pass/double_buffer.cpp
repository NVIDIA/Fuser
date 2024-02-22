// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <ir/utils.h>
#include <kernel_ir.h>

#include <device_lower/pass/double_buffer.h>

#include <algorithm>
#include <iterator>
#include <vector>

#define EXTRA_LOGS

namespace nvfuser {

int64_t getDoubleBufferAxisPosition(const TensorView* tv) {
  // Double-buffering prefetches the next subregion of the tensor by
  // doubling the allocation. The subregion is defined by the axes
  // at the CA position till the inner-most position. There must be
  // at least one axis that is outside (left) of the CA position,
  // which defines the loop where prefetching is applied. Therefore,
  // the CA position must be larger than 0.

  NVF_ERROR(tv->getComputeAtPosition() > 0);

  // Unroll must not exist outside of double-buffer axis
  auto first_unroll_it = std::find_if(
      tv->getLeafDomain().begin(),
      tv->getLeafDomain().end(),
      [](const auto axis) {
        return axis->getParallelType() == ParallelType::Unroll;
      });

  const int64_t first_unroll_pos =
      (int64_t)std::distance(tv->getLeafDomain().begin(), first_unroll_it);

  const int64_t unroll_or_ca_pos =
      std::min(tv->getComputeAtPosition(), first_unroll_pos);

  NVF_ERROR(
      unroll_or_ca_pos > 0,
      "Invalid tensor to double-buffer. Valid double buffer axis not found due to Unroll. ",
      tv->toString());

  int64_t valid_pos = -1;
  // Skip parallelized or broadcast axes
  for (int64_t i = unroll_or_ca_pos - 1; i >= 0; --i) {
    auto pt = tv->axis(i)->getParallelType();
    if (!isParallelTypeThread(pt) && !tv->axis(i)->isBroadcast()) {
      valid_pos = i;
      break;
    }
  }

  NVF_ERROR(
      valid_pos >= 0,
      "Invalid tensor to double-buffer. Valid double buffer axis not found. ",
      tv->toString());

  return valid_pos;
}

IterDomain* getDoubleBufferAxis(const TensorView* tv) {
  return tv->axis(getDoubleBufferAxisPosition(tv));
}

void validateDoubleBufferedTensor(const TensorView* tv) {
  auto double_buffer_pos = getDoubleBufferAxisPosition(tv);

  // Like vectorization, only LoadStoreOp with another TensorView is
  // considered.
  auto def = tv->definition();
  NVF_ERROR(
      def->isA<LoadStoreOp>(),
      "Invalid tensor to double-buffer. Only tensor defined by LoadStoreOp is supported: ",
      def->toString());

  NVF_ERROR(
      def->input(0)->isA<TensorView>(),
      "Invalid tensor to double-buffer. Only tensor defined by LoadStoreOp with TensorView is supported: ",
      def->toString());

  NVF_ERROR(
      !tv->hasComputeWith(),
      "computeWith is not supported with double buffering: ",
      tv->toString());

  // Require the producer tensor to have been computed entirely for
  // the double-buffering loop. Otherwise, the producer itself would
  // also need to be double-bufferred.
  auto producer = def->input(0)->as<TensorView>();
  NVF_ERROR(
      producer->getComputePosition(tv) <= double_buffer_pos,
      "Invalid tensor to double-buffer. The computeAt position of the producer tensor must be moved left: ",
      producer->toString());

  // Not strictly necessary, but only gmem -> smem or local and smem -> local
  // are allowed.
  const auto p_mem_type = producer->getMemoryType();
  const auto c_mem_type = tv->getMemoryType();
  NVF_ERROR(
      (p_mem_type == MemoryType::Global &&
       (c_mem_type == MemoryType::Shared || c_mem_type == MemoryType::Local)) ||
          (p_mem_type == MemoryType::Shared && c_mem_type == MemoryType::Local),
      "Invalid tensor to double-buffer: ",
      tv->toString(),
      ". Producer memory type: ",
      p_mem_type,
      ". Consumer memory type: ",
      c_mem_type);

  return;
}

namespace {

// Initial inspection of a fusion to find and validate double buffered tensors
class DoubleBufferFusionInspector : private IterVisitor {
 public:
  DoubleBufferFusionInspector(Fusion* fusion, DoubleBufferInfo& db_info)
      : db_info_(db_info) {
    traverse(fusion);
  }

 private:
  using IterVisitor::handle;

  void handle(TensorView* tv) final {
    if (!(tv->isDoubleBuffered() || tv->isCircularBuffered())) {
      return;
    }

    NVF_ERROR(
        tv->definition(), "Fusion input shouldn't be double buffered.", tv);

    validateDoubleBufferedTensor(tv);

    auto db_axis = getDoubleBufferAxis(tv);

    db_info_.setDoubleBufferAxis(tv, db_axis);
  }

 private:
  DoubleBufferInfo& db_info_;
};

// Creates kir::IfThenElse with the following predicate:
//  threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
kir::IfThenElse* createThreadPredicatedIfThenElse() {
  auto zero_val = IrBuilder::create<Val>(0L, PrimDataType::UInt);
  auto if_predicate_expr = IrBuilder::logicalAndExpr(
      IrBuilder::logicalAndExpr(
          IrBuilder::eqExpr(
              NamedScalar::getParallelIndex(ParallelType::TIDx), zero_val),
          IrBuilder::eqExpr(
              NamedScalar::getParallelIndex(ParallelType::TIDy), zero_val)),
      IrBuilder::eqExpr(
          NamedScalar::getParallelIndex(ParallelType::TIDz), zero_val));

  auto if_expr = IrBuilder::create<kir::IfThenElse>(
      IrBuilder::create<kir::Predicate>(if_predicate_expr));

  return if_expr;
}

// Creates kir::Loop with range based on stages' number
kir::ForLoop* createStagesForLoop(kir::ForLoop* double_buffer_loop) {
  auto stage_depth = GpuLower::current()->doubleBufferInfo().getStageDepthFor(
      double_buffer_loop->iter_domain());

  auto loop_start = IrBuilder::create<Val>(0L, PrimDataType::Index);
  auto loop_index = IrBuilder::create<Val>(PrimDataType::Index);
  auto loop_extend = IrBuilder::create<Val>(stage_depth, PrimDataType::Index);
  auto loop_domain_builder = IterDomainBuilder(loop_start, loop_extend);
  auto loop_step = IrBuilder::create<Val>(1L, PrimDataType::Index);

  const auto vectorize = false;
  Val* vectorize_shift = nullptr;
  const auto unroll_required = false;

  auto loop = IrBuilder::create<kir::ForLoop>(
      loop_domain_builder.build(),
      loop_index,
      loop_start,
      loop_extend,
      loop_step,
      vectorize,
      vectorize_shift,
      unroll_required,
      DoubleBufferLoopStage::NotApplicable);

  return loop;
}

// Creates kir::MBarrierArriveExpectTx for given LoadStoreOp and index of
//  loop in scope's which LoadStoreOp is present
kir::MBarrierArriveExpectTx* createMbarrierArriveExpectTx(
    LoadStoreOp* ldst,
    Val* loop_index) {
  auto ldst_out_tv = ldst->out()->as<TensorView>();
  Val* expect_bytes =
      IrBuilder::create<Val>(dataTypeSize(ldst_out_tv->dtype()));
  for (auto id : ldst_out_tv->getLeafDomain()) {
    if (id->getParallelType() == ParallelType::Bulk) {
      expect_bytes = SimplifyingIrBuilder::mulExpr(expect_bytes, id->extent());
    }
  }
  expect_bytes =
      SimplifyingIrBuilder::maybeCastExpr(DataType::UInt32, expect_bytes);

  auto mbarrier_tokens = GpuLower::current()->ldstMBarrierTokenMap().at(ldst);
  auto state = IrBuilder::create<kir::TensorIndex>(mbarrier_tokens, loop_index);

  auto mbarrier_objs = GpuLower::current()->ldstMBarrierMap().at(ldst);
  auto mbarrier =
      IrBuilder::create<kir::TensorIndex>(mbarrier_objs, loop_index);

  auto mbarrier_arrive_tx = IrBuilder::create<kir::MBarrierArriveExpectTx>(
      state, mbarrier, expect_bytes);

  return mbarrier_arrive_tx;
}

// Creates kir::MBarrierWait for given LoadStoreOp and loop index
kir::MBarrierWait* createMbarrierWait(LoadStoreOp* ldst, Val* loop_index) {
  auto mbarrier_objs = GpuLower::current()->ldstMBarrierMap().at(ldst);
  auto mbarrier =
      IrBuilder::create<kir::TensorIndex>(mbarrier_objs, loop_index);

  auto mbarrier_tokens = GpuLower::current()->ldstMBarrierTokenMap().at(ldst);
  auto state = IrBuilder::create<kir::TensorIndex>(mbarrier_tokens, loop_index);

  auto mbarrier_wait = IrBuilder::create<kir::MBarrierWait>(mbarrier, state);
  return mbarrier_wait;
}

// The epilogue loop is only created when the producer of a double
// buffer tensor is on smem, in which case it would otherwise require
// an additional predicate to guard buffer overruns. When it's on
// gmem, that isn't the case, so it does not need to create an
// epilogue loop.
// In case of cpAsyncBulk there is always an epilogue loop
bool requireEpilogue(const std::vector<Expr*>& exprs) {
  return std::any_of(exprs.begin(), exprs.end(), [](const Expr* expr) {
    return (expr->input(0)->as<TensorView>()->getMemoryType() ==
            MemoryType::Shared) ||
        (expr->as<LoadStoreOp>()->opType() ==
         LoadStoreOpType::CpAsyncBulkTensorTile);
  });
}

// Replicates double buffer loops for Prologue, Main, and
// Epilogue. Prologue only copies the load expressions of double
// buffered tensors, whereas Epilogue does any expression other than
// the loads. Main copies everything.
class DoubleBufferLoopCloner : public kir::IrVisitor {
 public:
  static kir::ForLoop* clone(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs,
      DoubleBufferLoopStage loop_type,
      const bool has_cp_async_bulk = false,
      const std::unordered_set<Expr*>& exclude = {}) {
    DoubleBufferLoopCloner cloner(
        double_buffer_loop,
        double_buffer_load_exprs,
        loop_type,
        has_cp_async_bulk,
        exclude);
    cloner.clone();
    return cloner.cloned_top_level_loop_;
  }

 private:
  DoubleBufferLoopCloner(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs,
      DoubleBufferLoopStage loop_type,
      const bool has_cp_async_bulk,
      const std::unordered_set<Expr*>& exclude)
      : double_buffer_loop_(double_buffer_loop),
        double_buffer_load_exprs_(double_buffer_load_exprs),
        loop_type_(loop_type),
        has_cp_async_bulk_(has_cp_async_bulk),
        exclude_(exclude) {}

  using kir::IrVisitor::handle;

  void clone() {
    const auto gpu_lower = GpuLower::current();

#ifdef EXTRA_LOGS
    std::cout << "[DEBUG] double_buffer_loop_ loop:\n"
              << double_buffer_loop_->toString() << std::endl;
#endif //  EXTRA_LOGS

    // Cloning the double buffer loop as follows:
    //
    // Prologue: 0 to 1
    // Main: 0 to (extent-1)
    // Epilogue: (extent-1) to extent

    auto index = GpuLower::current()->caMap()->getIndexVariable(
        double_buffer_loop_->iter_domain(), loop_type_);
    auto start = double_buffer_loop_->start();
    auto stop = double_buffer_loop_->stop();
    auto stage_depth = gpu_lower->doubleBufferInfo().getStageDepthFor(
        double_buffer_loop_->iter_domain());

    if (loop_type_ == DoubleBufferLoopStage::Prolog) {
      NVF_ERROR(start->isZeroInt());
      stop = SimplifyingIrBuilder::create<Val>(
          int64_t(stage_depth - 1), DataType::Index);
    } else if (
        loop_type_ == DoubleBufferLoopStage::Main &&
        requireEpilogue(double_buffer_load_exprs_)) {
      stop = IrBuilder::subExpr(
          double_buffer_loop_->stop(), gpu_lower->kernel()->oneVal());
    } else if (loop_type_ == DoubleBufferLoopStage::Epilog) {
      NVF_ERROR(requireEpilogue(double_buffer_load_exprs_));
      start = IrBuilder::subExpr(
          double_buffer_loop_->stop(),
          SimplifyingIrBuilder::create<Val>(
              int64_t(stage_depth - 1), DataType::Index));
    }

    cloned_top_level_loop_ = IrBuilder::create<kir::ForLoop>(
        double_buffer_loop_->iter_domain(),
        index,
        start,
        stop,
        gpu_lower->kernel()->oneVal(),
        false,
        nullptr,
        double_buffer_loop_->isUnrollRequired(),
        loop_type_);

    handle(double_buffer_loop_);
  }

  void handle(kir::ForLoop* fl) final {
    kir::ForLoop* cloned_loop = fl == double_buffer_loop_
        ? cloned_top_level_loop_
        : IrBuilder::create<kir::ForLoop>(fl);

    cloned_scopes_.push_back(&cloned_loop->body());

    kir::IrVisitor::handle(fl);

    cloned_scopes_.pop_back();

    // Add the cloned loop into the parent loop body only when the
    // cloned loop contains expressions.
    if (!cloned_loop->body().empty() && !cloned_scopes_.empty()) {
      cloned_scopes_.back()->push_back(cloned_loop);
    }
  }

  void handle(kir::IfThenElse* ite) final {
    NVF_ERROR(false, "No IfThenElse should exist yet");
  }

  void dispatch(Expr* expr) final {
    if (exclude_.count(expr) > 0) {
      return;
    }

    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::dispatch(expr);
      return;
    }

    NVF_ERROR(!cloned_scopes_.empty());

    auto out_tv = ir_utils::getTvOutput(expr);
    const auto is_double_buffer_load_expr = std::any_of(
        double_buffer_load_exprs_.begin(),
        double_buffer_load_exprs_.end(),
        [out_tv](const auto load_expr) {
          auto double_buffer_tv = ir_utils::getTvOutput(load_expr);
          NVF_ERROR(double_buffer_tv != nullptr);
          return out_tv == double_buffer_tv;
        });

    if (has_cp_async_bulk_) {
      auto gpu_lower = GpuLower::current();

      // Expr that is a part of cpAsyncBulk synchronization proces, added
      //  earlier to satisfy other passes checks, are handled already won't
      //  be pushed to the new scope.
      // There can be cases where cpAsyncBulk data movement is not a part of
      //  double buffering, such expr will be added to new scope.
      const auto isIgnorableCpAsyncBulkSmemAlloc =
          (gpu_lower->mBarrierTokenSmemAllocSet().count(expr) != 0);
      const auto isIgnorableMbarrierInit = (expr->isA<kir::MBarrierInit>()) &&
          (gpu_lower->ldstMBarrierTokenMap().count(expr) != 0);
      const auto isIgnorableMbarrierInval =
          (expr->isA<kir::MBarrierInvalidate>()) &&
          (gpu_lower->ldstMBarrierTokenMap().count(expr) != 0);

      // Target:
      // pre-prolog:
      // - smem allocations (mbarriers, tokens)
      // - mbarrier init (0..stages)
      //
      // prolog loop:
      // - 0th thread:
      //   - issue 0..stages-1 cp async bulks
      //
      // main loop:
      // - 0th thread:
      //   - issue next cp async bulk
      // - copy body, without
      //   - smem allocations
      //   - mbarrier inits
      //   - mbarrier inval
      //
      // epilogue loop:
      // - copy body, without
      //   - smem allocations
      //   - issuing cp async
      //   - mbarrier inits
      //   - mbarrier inval
      //
      // post-epilogue:
      //  - 0th thread: loop with mbarriers inval

      switch (loop_type_) {
        case DoubleBufferLoopStage::Prolog: {
          if (expr->isA<LoadStoreOp>()) {
            // Handle cpAsyncBulk type LoadStoreOp that is registered with
            //  token smem TVs as it requires synchronization
            if (gpu_lower->ldstMBarrierTokenMap().count(expr) != 0) {
              // See AllocationInserter for details when and how
              //  token map is filled with data

              // Replace cpAsyncBulk type LoadStoreOp with:
              //  if (0th thread in block)
              //    token[loop_idx] =
              //    mbarrier::arriveExpectTx(mbarrier[loop_idx])
              //    cpAsyncBulk(mbarrier[loop_idx],...)
              //
              // Where loop_idx is in range 0...stages-1

              auto if_expr = createThreadPredicatedIfThenElse();
              auto body = if_expr->thenBody();

              auto ldst = expr->as<LoadStoreOp>();

              auto mbarrier_arrive_tx = createMbarrierArriveExpectTx(
                  ldst, double_buffer_loop_->index());
              body.push_back(mbarrier_arrive_tx);

              // Clone LoadStoreOp & map it to mbarrier alloc
              auto new_ldst =
                  IrBuilder::create<LoadStoreOp>(
                      ldst->opType(), ldst->out(), ldst->in(), ldst->cacheOp())
                      ->withPredicate(ldst->predicate());

              body.push_back(new_ldst);

              // Register mbarrier object to be used with new LoadStoreOp
              //  from prolog loop
              gpu_lower->ldstMBarrierIndexMap()[new_ldst] =
                  mbarrier_arrive_tx->mbarrier();

              cloned_scopes_.back()->push_back(if_expr);
#ifdef EXTRA_LOGS
              std::cout << "[DEBUG] new MBarrierArriveExpectTx node: "
                        << mbarrier_arrive_tx->toString();
              std::cout << "[DEBUG] new LoadStoreOp node: "
                        << new_ldst->toString();
#endif //  EXTRA_LOGS
              break;
            } else if (is_double_buffer_load_expr) {
              // NOTE: that there can be multiple exprs defining double buffered
              // TVs (e.g., buffer initialization).
              cloned_scopes_.back()->push_back(expr);
            }
          }
          break;
        }
        case DoubleBufferLoopStage::Main: {
          // Handle cpAsyncBulk type LoadStoreOp that is registered with
          //  token smem TVs as it requires synchronization
          if (expr->isA<LoadStoreOp>() &&
              gpu_lower->ldstMBarrierTokenMap().count(expr) != 0) {
            // cpAsyncBulk for double-buffered tensor has assigned
            //  a placeholder for token objects

            auto stage_depth =
                GpuLower::current()->doubleBufferInfo().getStageDepthFor(
                    double_buffer_loop_->iter_domain());

            if (curr_stage_index_ == nullptr) {
              curr_stage_index_ = IrBuilder::modExpr(
                  double_buffer_loop_->index(),
                  IrBuilder::create<Val>(stage_depth, PrimDataType::Index));
              auto curr_stage_index_alloc = IrBuilder::create<kir::Allocate>(
                  curr_stage_index_,
                  MemoryType::Local,
                  IrBuilder::create<Val>(1L, PrimDataType::Index),
                  false);
              cloned_scopes_.back()->push_back(curr_stage_index_alloc);
              cloned_scopes_.back()->push_back(curr_stage_index_->definition());
            }
            if (next_stage_index_ == nullptr) {
              next_stage_index_ = IrBuilder::modExpr(
                  IrBuilder::addExpr(
                      double_buffer_loop_->index(),
                      IrBuilder::subExpr(
                          IrBuilder::create<Val>(
                              stage_depth, PrimDataType::Index),
                          IrBuilder::create<Val>(1L, PrimDataType::Index))),
                  IrBuilder::create<Val>(stage_depth, PrimDataType::Index));
              auto next_stage_index_alloc = IrBuilder::create<kir::Allocate>(
                  next_stage_index_,
                  MemoryType::Local,
                  IrBuilder::create<Val>(1L, PrimDataType::Index),
                  false);
              cloned_scopes_.back()->push_back(next_stage_index_alloc);
              cloned_scopes_.back()->push_back(next_stage_index_->definition());
            }

            // Replace LoadStoreOp with:
            //  if (0th thread in block)
            //    token[next_stage] =
            //    mbarrier::arriveExpectTx(mbarrier[next_stage])
            //    cpAsyncBulk(mbarrier[next_stage],...)
            //  mbarrier::wait(token[curr_stage])
            //
            // Where mbarrier and token are smem arrays bound to the LoadStoreOp

            auto if_expr = createThreadPredicatedIfThenElse();
            auto body = if_expr->thenBody();

            auto ldst = expr->as<LoadStoreOp>();
            auto mbarrier_arrive_tx =
                createMbarrierArriveExpectTx(ldst, next_stage_index_);
            body.push_back(mbarrier_arrive_tx);
            body.push_back(ldst);

            // Register mbarrier object to be used with LoadStoreOp
            //  from main loop
            gpu_lower->ldstMBarrierIndexMap()[ldst] =
                mbarrier_arrive_tx->mbarrier();

            cloned_scopes_.back()->push_back(if_expr);

            // Construct mBarrier::wait for current stage
            auto mbarrier_wait = createMbarrierWait(ldst, curr_stage_index_);
            cloned_scopes_.back()->push_back(mbarrier_wait);
#ifdef EXTRA_LOGS
            std::cout << "[DEBUG] new MBarrierArriveExpectTx node: "
                      << mbarrier_arrive_tx->toString();
            std::cout << "[DEBUG] new MBarrierWait node: "
                      << mbarrier_wait->toString();
#endif //  EXTRA_LOGS
            break;
          }
          if (!(isIgnorableCpAsyncBulkSmemAlloc || isIgnorableMbarrierInit ||
                isIgnorableMbarrierInval)) {
            cloned_scopes_.back()->push_back(expr);
          }
          break;
        }
        case DoubleBufferLoopStage::Epilog: {
          if (!(isIgnorableCpAsyncBulkSmemAlloc || isIgnorableMbarrierInit ||
                isIgnorableMbarrierInval || is_double_buffer_load_expr)) {
            cloned_scopes_.back()->push_back(expr);
          }
          break;
        }
        case DoubleBufferLoopStage::NotApplicable: {
          NVF_ERROR(false, "Unsupported loop mode, got: ", loop_type_);
        }
      }
    } else {
      if (loop_type_ == DoubleBufferLoopStage::Main) {
        cloned_scopes_.back()->push_back(expr);
        return;
      }

      // In Prologue and Epilogue, either load expressions or anything
      // else are copied. Note that there can be multiple exprs defining
      // double buffered TVs (e.g., buffer initialization).
      if ((loop_type_ == DoubleBufferLoopStage::Prolog &&
           is_double_buffer_load_expr) ||
          (loop_type_ == DoubleBufferLoopStage::Epilog &&
           !is_double_buffer_load_expr)) {
        cloned_scopes_.back()->push_back(expr);
      }
    }
  }

 private:
  kir::ForLoop* double_buffer_loop_ = nullptr;
  const std::vector<Expr*>& double_buffer_load_exprs_;
  const DoubleBufferLoopStage loop_type_;

  kir::ForLoop* cloned_top_level_loop_ = nullptr;
  std::deque<kir::Scope*> cloned_scopes_;
  const bool has_cp_async_bulk_;
  const std::unordered_set<Expr*>& exclude_;

  // Current stage, expectation:
  //  curr_stages_idx = (double_buffer_loop_idx % stages)
  Val* curr_stage_index_ = nullptr;
  // Next stage, expectation:
  //  next_stages_idx = (double_buffer_loop_idx + (stages -1)) % stages
  Val* next_stage_index_ = nullptr;
};

using InsertionInfo = std::unordered_map<kir::ForLoop*, std::vector<Expr*>>;

class IsDoubleBufferLoadLoop : public kir::IrVisitor {
 public:
  static bool check(
      Expr* expr,
      const std::vector<Expr*>& double_buffer_load_exprs) {
    IsDoubleBufferLoadLoop checker(double_buffer_load_exprs);
    return checker.check(expr);
  }

 private:
  IsDoubleBufferLoadLoop(const std::vector<Expr*>& double_buffer_load_exprs)
      : double_buffer_load_exprs_(double_buffer_load_exprs) {}

  using kir::IrVisitor::handle;

  bool check(Expr* expr) {
    dispatch(expr);
    return result_;
  }

  void dispatch(Expr* expr) final {
    if (result_) {
      return;
    }
    if (std::find(
            double_buffer_load_exprs_.begin(),
            double_buffer_load_exprs_.end(),
            expr) != double_buffer_load_exprs_.end()) {
      result_ = true;
      return;
    }
    IrVisitor::dispatch(expr);
  }

 private:
  const std::vector<Expr*>& double_buffer_load_exprs_;
  bool result_ = false;
};

// Traverse lowered loop-nests and find all double buffer loops and
// associated load expressions.
class DoubleBufferLoopNestInspector : private kir::IrVisitor {
 public:
  static InsertionInfo run(const std::vector<Expr*>& exprs) {
    DoubleBufferLoopNestInspector inspector(exprs);
    return inspector.insertion_info_;
  }

 private:
  DoubleBufferLoopNestInspector(const std::vector<Expr*>& exprs) {
    handle(exprs);
  }

  using kir::IrVisitor::handle;

  // Collect double buffer related information on a expr
  //  that is a memory load, i.e. a LoadStore or a Set.
  void handlePossibleLoadExpr(Expr* expr) {
    const auto gpu_lower = GpuLower::current();

    auto out_tv = ir_utils::getTvOutput(expr);

    if (out_tv == nullptr) {
      return;
    }

    // Ignore init loop
    if (!(out_tv->isDoubleBuffered() || out_tv->isCircularBuffered()) ||
        !expr->input(0)->isA<TensorView>()) {
      return;
    }

    auto double_buffer_loop =
        gpu_lower->doubleBufferInfo().getDoubleBufferLoop(out_tv, for_loops_);

    NVF_ERROR(
        double_buffer_loop != nullptr,
        "No double buffer loop found for a double buffered tensor: ",
        out_tv->toString());

    validateDoubleBufferLoop(double_buffer_loop);

    insertion_info_[double_buffer_loop].push_back(expr);
  }

  void handle(UnaryOp* uop) final {
    handlePossibleLoadExpr(uop);
  }

  void handle(LoadStoreOp* ldst) final {
    handlePossibleLoadExpr(ldst);
  }

  static void validateDoubleBufferLoop(kir::ForLoop* loop) {
    NVF_ERROR(
        loop->start()->isZeroInt(), "Unsupported loop: ", loop->toString());
    NVF_ERROR(loop->step()->isOneInt(), "Unsupported loop: ", loop->toString());
    NVF_ERROR(
        !loop->vectorize(),
        "Vectorized loop should not be the allocation loop for double-buffered tensor: ",
        loop->toString());
    NVF_ERROR(
        !loop->vectorize_shift(),
        "Vectorize shift loop should not be the allocation loop for double-buffered tensor: ",
        loop->toString());
  }

  InsertionInfo insertion_info_;
};

// Creates pre-prologue pieces of code needed for proper handling
//  async TMA memory copies - moving allocation outside of main loop
//  and initialization of mbarriers.
//
// Expected result:
//   mbarrier and token smem allocations
//   if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
//     for (unsigned i = 0; i < stages; ++i) {
//       mbarrier::init(...);
//     }
//   }
class CpAsyncBulkPrePrologue : public kir::IrVisitor {
 public:
  static std::vector<Expr*> create(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs) {
    CpAsyncBulkPrePrologue creator(
        double_buffer_loop, double_buffer_load_exprs);
    creator.create();
    return creator.pre_prologue_exprs_;
  }

 private:
  CpAsyncBulkPrePrologue(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs)
      : double_buffer_loop_(double_buffer_loop),
        double_buffer_load_exprs_(double_buffer_load_exprs) {}

  using kir::IrVisitor::handle;

  void create() {
    const auto gpu_lower = GpuLower::current();

    // Find and add smem allocations for tokens and mbarrier objects
    handle(double_buffer_loop_);

    // Define how many threads should arrive at the barrier
    //  we expect 0th thread to handle init/arrive & transaction config
    //  while other threads will wait for it
    auto one_val = IrBuilder::create<Val>(1L, PrimDataType::UInt32);

    // Construct predicate
    // 'threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0'
    auto if_expr = createThreadPredicatedIfThenElse();

    // Construct for loop, a body for if expressiong
    auto loop = createStagesForLoop(double_buffer_loop_);

    // Construct loop body with:
    // - mBarriers' initializations for each element in smem array for
    //   each double buffered tensor
    // - expected arrival: number of threads in the block (for now)
    for (const auto ldst : double_buffer_load_exprs_) {
      if (0 != gpu_lower->ldstMBarrierMap().count(ldst)) {
        auto mbarrier = gpu_lower->ldstMBarrierMap()[ldst];
        auto mbarrier_index =
            IrBuilder::create<kir::TensorIndex>(mbarrier, loop->index());
        auto mbarrier_init =
            IrBuilder::create<kir::MBarrierInit>(mbarrier_index, one_val);
        loop->body().push_back(mbarrier_init);
      }
    }

    if_expr->thenBody().push_back(loop);

    pre_prologue_exprs_.push_back(if_expr);

#ifdef EXTRA_LOGS
    std::cout
        << "=============================================================\n";
    std::cout << "[INFO] Pre-prologue-exprs: \n";
    for (const auto expr : pre_prologue_exprs_) {
      std::cout << expr->toString(0);
    }
    std::cout
        << "=============================================================\n";
#endif //  EXTRA_LOGS
  }

  void handle(kir::ForLoop* fl) final {
    kir::IrVisitor::handle(fl);
  }

  void handle(kir::IfThenElse* ite) final {
    NVF_ERROR(false, "No IfThenElse should exist yet");
  }

  void dispatch(Expr* expr) final {
    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::dispatch(expr);
      return;
    }

    if (auto alloc = dynamic_cast<kir::Allocate*>(expr)) {
      if (alloc->memoryType() == MemoryType::Shared) {
        if (GpuLower::current()->mBarrierTokenSmemAllocSet().count(alloc) !=
            0) {
          pre_prologue_exprs_.push_back(expr);
        }
      }
    }
  }

 private:
  kir::ForLoop* double_buffer_loop_ = nullptr;
  const std::vector<Expr*>& double_buffer_load_exprs_;

  std::vector<Expr*> pre_prologue_exprs_;
};

// Creates post-epilogue pieces of code needed for proper handling async
//  TMA memory copies - mbarriers release
//
// Expected result:
//   if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
//     for (unsigned i = 0; i < stages; ++i) {
//       mbarrier::inval(...);
//     }
//   }
class CpAsyncBulkPostEpilogue {
 public:
  static std::vector<Expr*> create(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs) {
    CpAsyncBulkPostEpilogue creator(
        double_buffer_loop, double_buffer_load_exprs);
    creator.create();
    return creator.post_prologue_exprs_;
  }

 private:
  CpAsyncBulkPostEpilogue(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs)
      : double_buffer_loop_(double_buffer_loop),
        double_buffer_load_exprs_(double_buffer_load_exprs) {}

  void create() {
    const auto gpu_lower = GpuLower::current();

    // Construct predicate
    // 'threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0'
    auto if_expr = createThreadPredicatedIfThenElse();

    // Construct for loop, a body for if expressiong
    auto loop = createStagesForLoop(double_buffer_loop_);

    // Construct loop body with:
    // - mBarriers' invalidation for each element in smem array for
    //   each double buffered tensor
    for (const auto ldst : double_buffer_load_exprs_) {
      if (0 != gpu_lower->ldstMBarrierMap().count(ldst)) {
        auto mbarrier = gpu_lower->ldstMBarrierMap()[ldst];
        auto mbarrier_index =
            IrBuilder::create<kir::TensorIndex>(mbarrier, loop->index());
        auto mbarrier_inval =
            IrBuilder::create<kir::MBarrierInvalidate>(mbarrier_index);
        loop->body().push_back(mbarrier_inval);
      }
    }

    if_expr->thenBody().push_back(loop);

    post_prologue_exprs_.push_back(if_expr);

#ifdef EXTRA_LOGS
    std::cout
        << "=============================================================\n";
    std::cout << "[INFO] Post-epilogue-exprs: \n";
    for (const auto expr : post_prologue_exprs_) {
      std::cout << expr->toString(0);
    }
    std::cout
        << "=============================================================\n";
#endif //  EXTRA_LOGS
  }

 private:
  kir::ForLoop* double_buffer_loop_ = nullptr;
  const std::vector<Expr*>& double_buffer_load_exprs_;

  std::vector<Expr*> post_prologue_exprs_;
};

namespace {

void getAllocInTrivialLoop(
    kir::ForLoop* fl,
    std::unordered_set<Expr*>& output) {
  if (!fl->isTrivial()) {
    return;
  }
  for (auto expr : fl->body().exprs()) {
    if (expr->isA<kir::Allocate>()) {
      output.emplace(expr);
    } else if (auto loop = dynamic_cast<kir::ForLoop*>(expr)) {
      getAllocInTrivialLoop(loop, output);
    }
  }
}

} // namespace

// Apply double buffering transformations
class DoubleBufferInserter : private kir::ExprMutator {
 public:
  // When there exist multiple double buffer loops, apply
  // transformations to inner-most loops first. A single ExprMutator
  // pass can only process one loop.
  static std::vector<Expr*> run(
      const std::vector<Expr*>& exprs,
      InsertionInfo insertion_info) {
    auto inserted_exprs = exprs;
    while (!insertion_info.empty()) {
      DoubleBufferInserter inserter(inserted_exprs, insertion_info);
      inserted_exprs = inserter.exprs_;
    }
    return inserted_exprs;
  }

 private:
  DoubleBufferInserter(
      const std::vector<Expr*>& exprs,
      InsertionInfo& insertion_info)
      : insertion_info_(insertion_info) {
    auto num_double_buffer_loops = insertion_info.size();
    traverseAndInsert(exprs);
    NVF_ERROR(processed_loop_ != nullptr);
    NVF_ERROR(insertion_info.size() == num_double_buffer_loops - 1);
  }

  using kir::ExprMutator::handle;

  void handle(kir::ForLoop* loop) final {
    kir::ExprMutator::handle(loop);

    // If another loop is already taken care of, no more loop should
    // be done in the same pass
    if (processed_loop_ != nullptr) {
      return;
    }

    auto it = insertion_info_.find(loop);
    if (it == insertion_info_.end()) {
      return;
    }

    auto hasCpAsyncBulk = std::any_of(
        it->second.begin(), it->second.end(), ir_utils::isCpAsyncBulk);

    if (hasCpAsyncBulk) {
      insertTma(loop, it->second);
    } else {
      insert(loop, it->second);
    }
    processed_loop_ = loop;
    insertion_info_.erase(loop);
  }

  void insertTma(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& loads) {
    const bool has_cp_async_bulk = true;

    // cpAsyncBulk (with TMA) requires some operations to be done prior to
    //  prologue loop, for example:
    // - smem allocation
    // - initialization of mbarrier objects
    auto pre_prologue_exprs =
        CpAsyncBulkPrePrologue::create(double_buffer_loop, loads);
    if (!pre_prologue_exprs.empty()) {
      for (auto expr : pre_prologue_exprs) {
        registerInsertBefore(double_buffer_loop, expr);
      }
    }

    auto prologue_loop = DoubleBufferLoopCloner::clone(
        double_buffer_loop,
        loads,
        DoubleBufferLoopStage::Prolog,
        has_cp_async_bulk);
    registerInsertBefore(double_buffer_loop, prologue_loop);

    // cpAsyncBulk (with TMA) block sync prior to entering main loop to
    //  make smem with mbarrier objects is initialized.
    if (has_cp_async_bulk) {
      auto sync = IrBuilder::create<kir::BlockSync>(false);
      registerInsertBefore(double_buffer_loop, sync);
    }

    auto main_loop = DoubleBufferLoopCloner::clone(
        double_buffer_loop,
        loads,
        DoubleBufferLoopStage::Main,
        has_cp_async_bulk);

    registerReplace(double_buffer_loop, main_loop);

    if (requireEpilogue(loads)) {
      // In the case where the main loop is trivial (for example, ldmatrix in
      // matmul kernel), we need to be careful when copying epilog loop. For
      // example, if the main loop is:
      //   for (int i = 0; i < 1; ++i) {
      //     ...
      //     float T1[2];
      //     T1 = ...
      //     ...
      //   }
      // Because trivial loop is not generated, the allocation of T1 will be one
      // level above in the generated scope. So when we copy epilog, we need to
      // make sure we don't copy these allocation so that there is no duplicate
      // allocation.
      std::unordered_set<Expr*> alloc_in_main;
      getAllocInTrivialLoop(main_loop, alloc_in_main);
      auto epilogue_loop = DoubleBufferLoopCloner::clone(
          double_buffer_loop,
          loads,
          DoubleBufferLoopStage::Epilog,
          has_cp_async_bulk,
          alloc_in_main);
      registerInsertAfter(double_buffer_loop, epilogue_loop);
    }

    auto post_epilogue_exprs =
        CpAsyncBulkPostEpilogue::create(double_buffer_loop, loads);
    if (!post_epilogue_exprs.empty()) {
      for (auto expr : post_epilogue_exprs) {
        // TODO: insert after epilogue loop, not the main loop!
        registerInsertAfter(double_buffer_loop, expr);
      }
    }

    NVF_ERROR(false, "insertTma - not implemented");
  }

  void insert(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& loads) {
    static const auto has_cp_async_bulk = false;

    auto prologue_loop = DoubleBufferLoopCloner::clone(
        double_buffer_loop,
        loads,
        DoubleBufferLoopStage::Prolog,
        has_cp_async_bulk);
    registerInsertBefore(double_buffer_loop, prologue_loop);

    auto write_to_smem =
        std::any_of(loads.begin(), loads.end(), [](const Expr* expr) {
          return expr->output(0)->as<TensorView>()->getMemoryType() ==
              MemoryType::Shared;
        });

    // RAW sync is not inserted for double buffered tensors. The only
    // exception is the prologue load.
    bool has_cpasync = false;
    if (write_to_smem) {
      // Here the initial sync before entering double buffer loop is
      //  inserted.

      // If any of the double buffered tensor in this double buffer
      //  loop is async copy. We want to wait for the gmem loads to
      //  finish before synchronizing the block.
      if (std::any_of(loads.begin(), loads.end(), ir_utils::isCpAsyncOp)) {
        auto stage_depth =
            GpuLower::current()->doubleBufferInfo().getStageDepthFor(
                double_buffer_loop->iter_domain());
        auto cp_async_wait = IrBuilder::create<kir::AsyncWait>(
            AsyncOpType::CpAsync, stage_depth - 2);
        prologue_loop->body().push_back(
            IrBuilder::create<kir::AsyncCommit>(AsyncOpType::CpAsync));
        registerInsertBefore(double_buffer_loop, cp_async_wait);
        has_cpasync = true;
      }

      // Insert the initial block sync before entering main loop.
      if (std::any_of(loads.begin(), loads.end(), [](Expr* expr) {
            return GpuLower::current()
                ->syncMap()
                ->needsRawSync(ir_utils::getTvOutput(expr))
                .hasTID();
          })) {
        // If any of the double buffered loads require sync, as indicated
        //  by sync info map, insert the sync before entering the double buffer
        //  loop.
        // TODO:
        //  Currently not supporting double buffer in gmem, but short to mid
        //  term not yet a priority to go for this case.
        auto sync = IrBuilder::create<kir::BlockSync>(false);
        registerInsertBefore(double_buffer_loop, sync);
      }
    }

    auto main_loop = DoubleBufferLoopCloner::clone(
        double_buffer_loop,
        loads,
        DoubleBufferLoopStage::Main,
        has_cp_async_bulk);

    registerReplace(double_buffer_loop, main_loop);

    // Insert the wait instruction in this pass instead
    //  of relying on WAR sync pass to do it.
    // The WAR sync pass today would insert the wait function
    //  exactly where we need it but the purpose of this wait
    //  insertion isn't exactly WAR protection.
    //
    // TODO: [Double Buffer Sync]
    //  We might eventually want to move the block sync inserted
    //   by WAR pass here as well since this sync insertion is kind
    //   of both WAR and RAW (or neither RAW nor WAR, depends
    //   on how we look at it).
    // Eg. in the case when a intermediate
    //   tensor is double buffered.
    //
    //  __block_sync();    // This is the initial sync
    //  For i in ...       // Double buffer loop
    //     A[i%2] = ...;
    //     ...  = A[1-i%2];
    //     __block_sync();  // sync within loop
    //     ...
    //  The "sync within loop" can be placed anywhere in the
    //   double buffer loop while in the case of RAW and WAR
    //   there'd be extra insertion point restrictions.
    //  We are currently not actively exploring opportunities
    //   with this property of "double buffer sync" so this
    //   is more conceptual at the moment, aka low priority.
    if (has_cpasync) {
      insertCpAsyncCommitWaitInMainLoop(main_loop, loads);

      // The main loop will generate some async loads from invalid regions.
      // These populate the current cp.async group and they fill the smem with
      // zero. Subsequent code might assume an empty cp.async group (for example
      // an unparallelized batch matmul), or might re-use memory (WAW
      // hazard, see https://github.com/NVIDIA/Fuser/issues/2000). For safety,
      // we drain the group after the loops by waiting on these transfers.
      auto cp_async_wait_all =
          IrBuilder::create<kir::AsyncWait>(AsyncOpType::CpAsync, 0);
      registerInsertAfter(double_buffer_loop, cp_async_wait_all);
    }

    if (requireEpilogue(loads)) {
      // In the case where the main loop is trivial (for example, ldmatrix in
      // matmul kernel), we need to be careful when copying epilog loop. For
      // example, if the main loop is:
      //   for (int i = 0; i < 1; ++i) {
      //     ...
      //     float T1[2];
      //     T1 = ...
      //     ...
      //   }
      // Because trivial loop is not generated, the allocation of T1 will be one
      // level above in the generated scope. So when we copy epilog, we need to
      // make sure we don't copy these allocation so that there is no duplicate
      // allocation.
      std::unordered_set<Expr*> alloc_in_main;
      getAllocInTrivialLoop(main_loop, alloc_in_main);
      auto epilogue_loop = DoubleBufferLoopCloner::clone(
          double_buffer_loop,
          loads,
          DoubleBufferLoopStage::Epilog,
          has_cp_async_bulk,
          alloc_in_main);
      registerInsertAfter(double_buffer_loop, epilogue_loop);
    }
  }

  // Simple conservative rule for inserting async copy wait
  //  primitive in the double buffer loop:
  void insertCpAsyncCommitWaitInMainLoop(
      kir::ForLoop* main_loop,
      const std::vector<Expr*>& loads) {
    NVF_ERROR(
        !main_loop->body().empty(),
        "Double buffer sync insertion: empty main loop.");
    auto& exprs = main_loop->body().exprs();
    // Note: This pass explicitly assumes that WAR sync has been
    //  inserted so would need to be updated if we re-order the
    //  passes. Cleanups suggested in [Double Buffer Sync]
    //  would resolve this dependency on pass ordering.
    auto stage_depth = GpuLower::current()->doubleBufferInfo().getStageDepthFor(
        main_loop->iter_domain());
    auto cp_async_commit =
        IrBuilder::create<kir::AsyncCommit>(AsyncOpType::CpAsync);
    auto cp_async_wait = IrBuilder::create<kir::AsyncWait>(
        AsyncOpType::CpAsync, stage_depth - 2);

    // Find the last double buffer load in the main loop, and insert
    // cp.async.commit after it.
    std::vector<Expr*>::const_iterator last_double_buffer_load = exprs.end();
    for (auto it = exprs.begin(); it != exprs.end(); ++it) {
      if (IsDoubleBufferLoadLoop::check(*it, loads)) {
        last_double_buffer_load = it;
      }
    }
    NVF_ERROR(last_double_buffer_load != exprs.end());
    std::vector<Expr*>::const_iterator commit_it =
        main_loop->body().insert(last_double_buffer_load + 1, cp_async_commit);

    // Check if a sync has been inserted by WAR sync pass.
    auto rend = std::make_reverse_iterator(commit_it);
    auto block_sync_it =
        std::find_if(exprs.rbegin(), rend, [](const Expr* expr) {
          return expr->isA<kir::BlockSync>();
        });
    if (block_sync_it == rend) {
      // If there's no sync, i.e. no tensor needs cross thread communication. We
      // still need a wait but it can just be anywhere after the cp.async.commit
      // in the loop. Chose to place at the end arbitrarily.
      main_loop->body().insert_after(exprs.back(), cp_async_wait);
    } else {
      // If a sync has been inserted, wait needs to be placed before the sync.
      main_loop->body().insert_before(*block_sync_it, cp_async_wait);
    }
  }

 private:
  InsertionInfo& insertion_info_;
  kir::ForLoop* processed_loop_ = nullptr;
};

} // namespace

void DoubleBufferInfo::build(Fusion* fusion) {
  DoubleBufferFusionInspector inspector(fusion, *this);

  // Build double buffered loop id's
  for (auto& info : map_) {
    auto double_buffer_axis = info.second.double_buffer_axis;
    // Keeps track of which loop disjoint set has been
    //  double buffered. In index allocation, one index
    //  variable would need to be allocated in each
    //  double buffer stage.
    concrete_double_buffered_loop_id_.insert(
        GpuLower::current()->caMap()->getConcreteMappedID(
            double_buffer_axis, IdMappingMode::LOOP));
  }
}

bool DoubleBufferInfo::isDoubleBufferedIterDomain(IterDomain* id) {
  auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::LOOP);
  return concrete_double_buffered_loop_id_.count(concrete_loop_id);
}

DoubleBufferInfo::TvInfo& DoubleBufferInfo::getTvInfo(const TensorView* tv) {
  NVF_ERROR(
      tv->isDoubleBuffered() || tv->isCircularBuffered(),
      "Not a double-buffered tensor: ",
      tv->toString());
  return map_[tv];
}

void DoubleBufferInfo::setDoubleBufferAxis(
    const TensorView* tv,
    IterDomain* axis) {
  getTvInfo(tv).double_buffer_axis = axis;

  // Also validate the stage consistency with CA map.
  unsigned int stage_depth = 0;
  if (tv->isCircularBuffered()) {
    stage_depth = tv->circularBufferDepth();
  } else {
    // Double buffer is essentially
    //  circular buffer with depth 2.
    stage_depth = 2;
  }

  // Set and validate the new stage depth.
  setStageDepth(axis, stage_depth);
}

void DoubleBufferInfo::setStageDepth(IterDomain* id, unsigned int stage_depth) {
  auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::LOOP);

  auto maybe_exisiting_depth_it = stage_depth_.find(concrete_loop_id);
  if (maybe_exisiting_depth_it == stage_depth_.end()) {
    stage_depth_[concrete_loop_id] = stage_depth;
  } else {
    NVF_ERROR(
        stage_depth == maybe_exisiting_depth_it->second,
        "Unsupported multiple depth pipelining, was set to ",
        maybe_exisiting_depth_it->second,
        " by ",
        maybe_exisiting_depth_it->first->toString(),
        " and then set to ",
        stage_depth,
        " by ",
        concrete_loop_id->toString());
  }
}

IterDomain* DoubleBufferInfo::getDoubleBufferAxis(const TensorView* tv) {
  if (!(tv->isDoubleBuffered() || tv->isCircularBuffered())) {
    return nullptr;
  }

  return getTvInfo(tv).double_buffer_axis;
}

unsigned int DoubleBufferInfo::getStageDepthFor(
    IterDomain* double_buffer_axis) {
  auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
      double_buffer_axis, IdMappingMode::LOOP);

  auto maybe_depth_it = stage_depth_.find(concrete_id);

  NVF_ERROR(maybe_depth_it != stage_depth_.end(), "Stage depth not found");

  return maybe_depth_it->second;
}

kir::ForLoop* DoubleBufferInfo::getDoubleBufferLoop(
    IterDomain* axis,
    const std::vector<kir::ForLoop*>& loops,
    bool ignore_prologue) {
  auto loop_it = std::find_if(loops.begin(), loops.end(), [&](const auto loop) {
    return GpuLower::current()->caMap()->areMapped(
               loop->iter_domain(), axis, IdMappingMode::EXACT) &&
        (!ignore_prologue ||
         loop->doubleBufferLoopStage() != DoubleBufferLoopStage::Prolog);
  });

  if (loop_it != loops.end()) {
    return *loop_it;
  } else {
    return nullptr;
  }
}

kir::ForLoop* DoubleBufferInfo::getDoubleBufferLoop(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    bool ignore_prologue) {
  auto axis = getDoubleBufferAxis(tv);

  if (axis == nullptr) {
    return nullptr;
  }

  return getDoubleBufferLoop(axis, loops, ignore_prologue);
}

void DoubleBufferInfo::setOriginalAllocSize(
    const TensorView* tv,
    Val* original_alloc_size) {
  getTvInfo(tv).original_alloc_size = original_alloc_size;
}

Val* DoubleBufferInfo::getOriginalAllocSize(const TensorView* tv) {
  if (!(tv->isDoubleBuffered() || tv->isCircularBuffered())) {
    return nullptr;
  }

  return getTvInfo(tv).original_alloc_size;
}

std::vector<Expr*> DoubleBufferPass::run(const std::vector<Expr*>& exprs) {
  auto insertion_info = DoubleBufferLoopNestInspector::run(exprs);
  return DoubleBufferInserter::run(exprs, insertion_info);
}

} // namespace nvfuser
