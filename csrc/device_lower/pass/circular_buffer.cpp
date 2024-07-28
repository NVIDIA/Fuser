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
#include <transform_iter.h>
#include <transform_replay.h>

#include <device_lower/pass/circular_buffer.h>

#include <algorithm>
#include <iterator>
#include <vector>

namespace nvfuser {

namespace {

// Creates kir::IfThenElse with the following predicate:
// threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
// TODO Replace with elect.sync ptx
kir::IfThenElse* createThreadPredicatedIfThenElse() {
  Val* zero_val = IrBuilder::create<Val>(0L, PrimDataType::UInt);
  Val* if_predicate_expr = IrBuilder::logicalAndExpr(
      IrBuilder::logicalAndExpr(
          IrBuilder::eqExpr(
              NamedScalar::getParallelIndex(ParallelType::TIDx), zero_val),
          IrBuilder::eqExpr(
              NamedScalar::getParallelIndex(ParallelType::TIDy), zero_val)),
      IrBuilder::eqExpr(
          NamedScalar::getParallelIndex(ParallelType::TIDz), zero_val));

  kir::IfThenElse* if_expr = IrBuilder::create<kir::IfThenElse>(
      IrBuilder::create<kir::Predicate>(if_predicate_expr));

  return if_expr;
}

// Creates kir::Loop with range based on stage depth
ForLoop* createStagesForLoop(ForLoop* circular_buffer_loop) {
  int64_t stage_depth =
      GpuLower::current()->circularBufferInfo().getStageDepthFor(
          circular_buffer_loop->iter_domain());

  Val* loop_start = IrBuilder::create<Val>(0L, PrimDataType::Index);
  Val* loop_index = IrBuilder::create<Val>(PrimDataType::Index);
  Val* loop_stop = IrBuilder::create<Val>(stage_depth, DataType::Index);
  IterDomainBuilder loop_domain_builder(loop_start, loop_stop);

  ForLoop* loop = IrBuilder::create<ForLoop>(
      loop_domain_builder.build(),
      loop_index,
      loop_start,
      loop_stop,
      /*step=*/GpuLower::current()->kernel()->oneVal(),
      /*vectorize=*/false,
      /*vectorize_shift=*/nullptr,
      /*unroll_required=*/false,
      CircularBufferLoopStage::NotApplicable);

  return loop;
}

// Get expected transaction count for given tma operation.
Val* getExpectedTransactionCount(LoadStoreOp* ldst) {
  TensorView* consumer_tv = ldst->out()->as<TensorView>();
  NVF_ERROR(
      GpuLower::current()->consumerToTMAInfo().count(consumer_tv),
      "Unable to find TMA info for consumer_tv: ",
      consumer_tv->toString());
  const TMAInfo& tma_info =
      GpuLower::current()->consumerToTMAInfo().at(consumer_tv);
  Val* expected_bytes = SimplifyingIrBuilder::maybeCastExpr(
      DataType::UInt32, tma_info.tileSizeBytes());
  auto is_multiple_of_16B = SimplifyingIrBuilder::eqExpr(
      SimplifyingIrBuilder::modExpr(
          expected_bytes, IrBuilder::create<Val>(16, DataType::Index)),
      expected_bytes->fusion()->zeroVal());
  GpuLower::current()->validate(
      is_multiple_of_16B,
      "The expected bytes must be a multiple of 16 bytes, but ",
      expected_bytes->toInlineString(),
      " is not.");
  return expected_bytes;
}

// Creates kir::MBarrierArriveExpectTx for given LoadStoreOp and index of
// loop in scope's which LoadStoreOp is present
//
// Example:
// __shared__ __mbarrier_t barriers[num_stages];
// __shared__ __mbarrier_token_t tokens[num_stages];
// for(nvfuser_index_t stage = 0; stage < num_stages; ++stage) {
//   if (elect_sync()) {
//     tokens[stage] =
//        mbarrier::arriveExpectTX(toSmem((&barriers[stage])), expected_bytes);
//   }
// }
kir::MBarrierArriveExpectTx* createMbarrierArriveExpectTx(
    LoadStoreOp* ldst,
    Val* loop_index) {
  // Get expected bytes for single TMA load operation
  Val* expected_bytes = getExpectedTransactionCount(ldst);

  // The expected_bytes for mbarrier::arriveExpectTX must account for all TMA
  // load operations launched for each circular buffer stage. We take the
  // product of all coordinate TMA iterDomains to the right of the circular
  // buffer axis.
  TensorView* ldst_out_tv = ldst->out()->as<TensorView>();
  const std::vector<IterDomain*>& leaf_domain = ldst_out_tv->getLoopDomain();
  for (size_t idx = ldst_out_tv->getComputeAtPosition();
       idx < leaf_domain.size();
       ++idx) {
    IterDomain* id = leaf_domain.at(idx);
    if (!isParallelTypeThread(id->getParallelType()) &&
        id->getParallelType() != ParallelType::Bulk) {
      expected_bytes =
          SimplifyingIrBuilder::mulExpr(expected_bytes, id->extent());
    }
  }

  expected_bytes =
      SimplifyingIrBuilder::maybeCastExpr(DataType::UInt32, expected_bytes);
  auto is_multiple_of_16B = SimplifyingIrBuilder::eqExpr(
      SimplifyingIrBuilder::modExpr(
          expected_bytes, IrBuilder::create<Val>(16, DataType::Index)),
      expected_bytes->fusion()->zeroVal());
  GpuLower::current()->validate(
      is_multiple_of_16B,
      "The expected bytes must be a multiple of 16 bytes, but ",
      expected_bytes,
      " is not.");

  TensorView* all_mbarrier_tokens =
      GpuLower::current()->ldstMBarrierTokenMap().at(ldst);
  kir::TensorIndex* stage_token =
      IrBuilder::create<kir::TensorIndex>(all_mbarrier_tokens, loop_index);

  TensorView* all_mbarriers = GpuLower::current()->ldstMBarrierMap().at(ldst);
  kir::TensorIndex* stage_mbarrier =
      IrBuilder::create<kir::TensorIndex>(all_mbarriers, loop_index);

  auto mbarrier_arrive_tx = IrBuilder::create<kir::MBarrierArriveExpectTx>(
      stage_token, stage_mbarrier, expected_bytes);

  return mbarrier_arrive_tx;
}

// Creates kir::MBarrierWait for given LoadStoreOp and loop index
kir::MBarrierWait* createMbarrierWait(LoadStoreOp* ldst, Val* loop_index) {
  TensorView* all_mbarriers = GpuLower::current()->ldstMBarrierMap().at(ldst);
  kir::TensorIndex* stage_mbarrier =
      IrBuilder::create<kir::TensorIndex>(all_mbarriers, loop_index);

  TensorView* all_mbarrier_tokens =
      GpuLower::current()->ldstMBarrierTokenMap().at(ldst);
  kir::TensorIndex* stage_token =
      IrBuilder::create<kir::TensorIndex>(all_mbarrier_tokens, loop_index);

  kir::MBarrierWait* mbarrier_wait =
      IrBuilder::create<kir::MBarrierWait>(stage_mbarrier, stage_token);
  return mbarrier_wait;
}

// The epilogue loop is only created when the producer of a circular
// buffer tensor is on smem, in which case it would otherwise require
// an additional predicate to guard buffer overruns. When it's on
// gmem, that isn't the case, so it does not need to create an
// epilogue loop. For TMA cpAsyncBulk, there is always an epilogue loop.
bool requireEpilogue(const std::vector<Expr*>& exprs) {
  return std::any_of(exprs.begin(), exprs.end(), [](const Expr* expr) {
    return (expr->input(0)->as<TensorView>()->getMemoryType() ==
            MemoryType::Shared) ||
        (expr->as<LoadStoreOp>()->opType() ==
         LoadStoreOpType::CpAsyncBulkTensorTile);
  });
}

// Replicates circular buffer loops for Prologue, Main, and
// Epilogue. Prologue only copies the load expressions of circular
// buffered tensors, whereas Epilogue does any expression other than
// the loads. Main copies everything.
class CircularBufferLoopCloner : public kir::IrVisitor {
 public:
  static ForLoop* clone(
      ForLoop* circular_buffer_loop,
      const std::vector<Expr*>& circular_buffer_load_exprs,
      CircularBufferLoopStage loop_type,
      const std::unordered_set<Expr*>& exclude = {}) {
    CircularBufferLoopCloner cloner(
        circular_buffer_loop, circular_buffer_load_exprs, loop_type, exclude);
    cloner.duplicate();
    return cloner.cloned_top_level_loop_;
  }
  virtual ~CircularBufferLoopCloner() = default;

 protected:
  CircularBufferLoopCloner(
      ForLoop* circular_buffer_loop,
      const std::vector<Expr*>& circular_buffer_load_exprs,
      CircularBufferLoopStage loop_type,
      const std::unordered_set<Expr*>& exclude)
      : circular_buffer_loop_(circular_buffer_loop),
        circular_buffer_load_exprs_(circular_buffer_load_exprs),
        loop_type_(loop_type),
        exclude_(exclude) {}

  using kir::IrVisitor::handle;

  void duplicate() {
    // Cloning the circular buffer loop as follows:
    //
    // Prologue: 0 to 1
    // Main: 0 to (extent-1)
    // Epilogue: (extent-1) to extent

    auto index = GpuLower::current()->caMap()->getIndexVariable(
        circular_buffer_loop_->iter_domain(), loop_type_);
    auto start = circular_buffer_loop_->start();
    auto stop = circular_buffer_loop_->stop();
    int64_t stage_depth =
        GpuLower::current()->circularBufferInfo().getStageDepthFor(
            circular_buffer_loop_->iter_domain());

    if (loop_type_ == CircularBufferLoopStage::Prolog) {
      NVF_ERROR(start->isZeroInt());
      stop = SimplifyingIrBuilder::create<Val>(
          int64_t(stage_depth - 1), DataType::Index);
    } else if (
        loop_type_ == CircularBufferLoopStage::Main &&
        requireEpilogue(circular_buffer_load_exprs_)) {
      stop = IrBuilder::subExpr(
          circular_buffer_loop_->stop(),
          SimplifyingIrBuilder::create<Val>(stage_depth - 1, DataType::Index));
    } else if (loop_type_ == CircularBufferLoopStage::Epilog) {
      NVF_ERROR(requireEpilogue(circular_buffer_load_exprs_));
      start = IrBuilder::subExpr(
          circular_buffer_loop_->stop(),
          SimplifyingIrBuilder::create<Val>(stage_depth - 1, DataType::Index));
    }

    cloned_top_level_loop_ = IrBuilder::create<ForLoop>(
        circular_buffer_loop_->iter_domain(),
        index,
        start,
        stop,
        /*step=*/GpuLower::current()->kernel()->oneVal(),
        /*vectorize=*/false,
        /*vectorize_shift=*/nullptr,
        circular_buffer_loop_->isUnrollRequired(),
        loop_type_);

    handle(circular_buffer_loop_);
  }

  void handle(ForLoop* fl) override {
    ForLoop* cloned_loop = fl == circular_buffer_loop_
        ? cloned_top_level_loop_
        : IrBuilder::create<ForLoop>(fl);

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

  void dispatch(Expr* expr) override {
    if (exclude_.count(expr) > 0) {
      return;
    }

    if (expr->isA<ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::dispatch(expr);
      return;
    }

    NVF_ERROR(!cloned_scopes_.empty());

    if (loop_type_ == CircularBufferLoopStage::Main) {
      cloned_scopes_.back()->push_back(expr);
      return;
    }

    // In Prologue and Epilogue, either load expressions or anything
    // else are copied. Note that there can be multiple exprs defining
    // circular buffered TVs (e.g., buffer initialization).

    auto out_tv = ir_utils::getTvOutput(expr);
    const auto is_circular_buffer_load_expr = std::any_of(
        circular_buffer_load_exprs_.begin(),
        circular_buffer_load_exprs_.end(),
        [out_tv](const auto load_expr) {
          auto circular_buffer_tv = ir_utils::getTvOutput(load_expr);
          NVF_ERROR(circular_buffer_tv != nullptr);
          return out_tv == circular_buffer_tv;
        });
    if ((loop_type_ == CircularBufferLoopStage::Prolog &&
         is_circular_buffer_load_expr) ||
        (loop_type_ == CircularBufferLoopStage::Epilog &&
         !is_circular_buffer_load_expr)) {
      cloned_scopes_.back()->push_back(expr);
    }
  }

 protected:
  ForLoop* circular_buffer_loop_ = nullptr;
  const std::vector<Expr*>& circular_buffer_load_exprs_;
  const CircularBufferLoopStage loop_type_;

  ForLoop* cloned_top_level_loop_ = nullptr;
  std::deque<Scope*> cloned_scopes_;
  const std::unordered_set<Expr*>& exclude_;
};

// Replicates circular buffer loops for Prologue, Main, and
// Epilogue. Prologue only copies the load expressions of circular
// buffered tensors, whereas Epilogue does any expression other than
// the loads. Main copies everything.
//
// Loop Structure:
// pre-prolog:
// - smem allocations (mbarriers, tokens)
// - mbarrier init for all stages
//
// prolog loop:
// - 0th thread:
//   - issue cp async bulks for all but last stages
//
// main loop:
// - select a single thread:
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
//  - 0th thread:
//   - loop with mbarriers inval
class TmaCircularBufferLoopCloner : public CircularBufferLoopCloner {
 public:
  static ForLoop* clone(
      ForLoop* circular_buffer_loop,
      const std::vector<Expr*>& circular_buffer_load_exprs,
      CircularBufferLoopStage loop_type,
      const std::unordered_set<Expr*>& exclude = {}) {
    TmaCircularBufferLoopCloner cloner(
        circular_buffer_loop, circular_buffer_load_exprs, loop_type, exclude);
    cloner.duplicate();
    return cloner.cloned_top_level_loop_;
  }
  ~TmaCircularBufferLoopCloner() override = default;

 private:
  TmaCircularBufferLoopCloner(
      ForLoop* circular_buffer_loop,
      const std::vector<Expr*>& circular_buffer_load_exprs,
      CircularBufferLoopStage loop_type,
      const std::unordered_set<Expr*>& exclude)
      : CircularBufferLoopCloner(
            circular_buffer_loop,
            circular_buffer_load_exprs,
            loop_type,
            exclude) {}

  void handle(ForLoop* fl) final {
    ForLoop* cloned_loop = fl == circular_buffer_loop_
        ? cloned_top_level_loop_
        : IrBuilder::create<ForLoop>(fl);

    // Add to stack
    for_loop_id_stack_.push_back(fl->iter_domain());
    cloned_scopes_.push_back(&cloned_loop->body());

    // Process for-loop
    kir::IrVisitor::handle(fl);

    // Pop from stack
    for_loop_id_stack_.pop_back();
    cloned_scopes_.pop_back();

    // if (elect_sync)
    //   // single arrive for all tma operations
    //   arrive_expect_tx
    //   // nested for-loop structure launching multiple tma operations
    //   for (...):
    //     tma_operation
    //   end for
    // end if
    // // all threads wait for transaction to complete
    // mbarrier_wait()
    //
    // Prolog - launch only - (arrive_expect_tx and tma_operations)
    // Main - Launch and wait - (arrive_expect_tx, tma_operations, and
    // mbarrier_wait) Epilog - wait only - (mbarrier_wait)

    // Skip if there is not an active for-loop structure
    if (cloned_scopes_.empty()) {
      return;
    }

    if (!cloned_loop->body().empty()) {
      if (mbarrier_arrive_tx_ == nullptr || cloned_scopes_.size() > 1) {
        // Add cloned for_loop when mbarrier_arrive_tx_ is not active or
        // we are within a nested for-loop structure
        cloned_scopes_.back()->push_back(cloned_loop);
      } else {
        // mbarrier::arriveExpectTx and TMA load operations occur in prologue
        // and main loops.
        //
        // Pseudo-code example:
        // if (elect_sync()) {
        //   mbarrier_tokens[stage] = mbarrier::arriveExpectTx(mbarriers[stage],
        //                                                     expected_tx);
        //   for (...) {
        //     Hopper::cpAsyncBulkTensorTileG2S(
        //       Hopper::CpAsyncBulkTensorTileG2SIndex<num_dims>{
        //         tma_descriptor, global_index, mbarrier[stage] },
        //       shared_index(stage, num_stages));
        //   }
        // }
        NVF_ERROR(cloned_scopes_.front() == &cloned_top_level_loop_->body());
        kir::IfThenElse* if_expr = createThreadPredicatedIfThenElse();
        Scope& body = if_expr->thenBody();
        body.push_back(mbarrier_arrive_tx_);
        body.push_back(cloned_loop);
        cloned_scopes_.back()->push_back(if_expr);
        mbarrier_arrive_tx_ = nullptr;
      }
    }

    // mbarrier::wait occurs in main and epilogue loops.
    //
    // Pseudo-code example:
    //  mbarrier::wait(mbarriers[stage], mbarrier_tokens[stage]);
    if (mbarrier_wait_ != nullptr && cloned_scopes_.size() == 1) {
      NVF_ERROR(cloned_scopes_.back() == &cloned_top_level_loop_->body());
      cloned_top_level_loop_->body().push_back(mbarrier_wait_);
      mbarrier_wait_ = nullptr;
    }
  }

  void dispatch(Expr* expr) final {
    // skip expression if it is in exclude set
    if (exclude_.count(expr) > 0) {
      return;
    }

    // Handle ForLoop and IfThenElse expr separately
    if (expr->isA<ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::dispatch(expr);
      return;
    }

    NVF_ERROR(!cloned_scopes_.empty());

    TensorView* out_tv = ir_utils::getTvOutput(expr);

    bool is_circular_buffer_load_expr = std::any_of(
        circular_buffer_load_exprs_.begin(),
        circular_buffer_load_exprs_.end(),
        [out_tv](Expr* load_expr) {
          TensorView* circular_buffer_tv = ir_utils::getTvOutput(load_expr);
          NVF_ERROR(circular_buffer_tv != nullptr);
          return out_tv == circular_buffer_tv;
        });

    // This expr is a part of cpAsyncBulk synchronization process. It was
    // added earlier to satisfy checks in other passes. It was already handled
    // already, so it will not be pushed to the new scope. cpAsyncBulk exprs
    // that are not a part of circular buffering, will be added to a new scope.
    bool mbarrier_token_exists =
        GpuLower::current()->ldstMBarrierTokenMap().count(expr) != 0;

    bool is_ignorable_tma_smem_alloc =
        (GpuLower::current()->mBarrierTokenSmemAllocSet().count(expr) != 0);

    bool is_ignorable_mbarrier_init =
        (expr->isA<kir::MBarrierInit>() && mbarrier_token_exists);

    bool is_ignorable_mbarrier_inval =
        (expr->isA<kir::MBarrierInvalidate>() && mbarrier_token_exists);

    int64_t stage_depth =
        GpuLower::current()->circularBufferInfo().getStageDepthFor(
            circular_buffer_loop_->iter_domain());

    switch (loop_type_) {
      case CircularBufferLoopStage::Prolog: {
        // Skip if not LoadStoreOp expression
        if (!expr->isA<LoadStoreOp>()) {
          break;
        }

        // Skip expr if it is not circular buffer expression
        if (!is_circular_buffer_load_expr) {
          break;
        }

        // NOTE: There can be circular buffered TVs without TMA load exprs.
        if (!mbarrier_token_exists) {
          cloned_scopes_.back()->push_back(expr);
          break;
        }

        // Handle cpAsyncBulk type LoadStoreOp that is registered with token
        //
        // See AllocationInserter for details when and how token map is filled
        // with data
        //
        // Replace cpAsyncBulk type LoadStoreOp with:
        //  if (elect_sync()) {
        //    for (int64_t loop_idx : irange(stages-1)) {
        //      token[loop_idx] =
        //        mbarrier::arriveExpectTx(mbarrier[loop_idx])
        //      cpAsyncBulk(mbarrier[loop_idx],...)
        //    }
        //  }

        LoadStoreOp* ldst = expr->as<LoadStoreOp>();

        // NOTE What happens with multiple ldst for different tensors?
        // There should be a single mbarrier_arrive_tx_ for all ldst in current
        // stage.
        NVF_ERROR(mbarrier_arrive_tx_ == nullptr);
        mbarrier_arrive_tx_ = createMbarrierArriveExpectTx(
            ldst, cloned_top_level_loop_->indexOrStartIfTrivial());

        // Clone LoadStoreOp and map it to mbarrier alloc
        Expr* new_ldst =
            IrBuilder::create<LoadStoreOp>(
                ldst->opType(), ldst->out(), ldst->in(), ldst->cacheOp())
                ->withPredicate(ldst->predicate());

        // Register mbarrier object to be used with new LoadStoreOp
        // from prolog loop
        NVF_ERROR(mbarrier_arrive_tx_->mbarrier()->isA<kir::TensorIndex>());
        GpuLower::current()->ldstMBarrierIndexMap().emplace(
            new_ldst, mbarrier_arrive_tx_->mbarrier()->as<kir::TensorIndex>());

        // If last cloned scope is the cloned_top_level_loop body, then add
        // mbarrier::arriveExpectTx and new loadStoreOp.
        int64_t active_for_loops = std::count_if(
            for_loop_id_stack_.begin(),
            for_loop_id_stack_.end(),
            [](IterDomain* id) {
              return id->getParallelType() == ParallelType::Serial;
            });
        if (active_for_loops == 1) {
          kir::IfThenElse* if_expr = createThreadPredicatedIfThenElse();
          Scope& body = if_expr->thenBody();
          body.push_back(mbarrier_arrive_tx_);
          body.push_back(new_ldst);
          cloned_scopes_.back()->push_back(if_expr);
          mbarrier_arrive_tx_ = nullptr;
          break;
        }

        // Otherwise, we are in a nested for-loop and should wait until we
        // return to top-level for loop.
        cloned_scopes_.back()->push_back(new_ldst);
        break;
      }
      case CircularBufferLoopStage::Main: {
        // Skip shared memory allocation, mbarrier initialize and mbarrier
        // invalidate
        if (is_ignorable_tma_smem_alloc || is_ignorable_mbarrier_init ||
            is_ignorable_mbarrier_inval) {
          break;
        }

        // Add expression if not circular-buffered load store operation
        if (!expr->isA<LoadStoreOp>() || !mbarrier_token_exists) {
          cloned_scopes_.back()->push_back(expr);
          break;
        }

        // Handle cpAsyncBulk type LoadStoreOp that is registered with token
        //
        // See AllocationInserter for details when and how token map is filled
        // with data
        //
        // Before waiting at the mbarrier for the current stage, we
        // launch the load operation for the next available stage. The
        // last buffer in the pipeline is the first available after the
        // prologue loop launches the initial wave of tma loads.
        //
        // current_compute_stage = loop_index % stage_depth
        if (current_compute_stage_ == nullptr) {
          current_compute_stage_ = IrBuilder::modExpr(
              circular_buffer_loop_->index(),
              IrBuilder::create<Val>(stage_depth, PrimDataType::Index));
          kir::Allocate* current_compute_stage_alloc =
              IrBuilder::create<kir::Allocate>(
                  current_compute_stage_,
                  MemoryType::Local,
                  IrBuilder::create<Val>(1L, PrimDataType::Index),
                  /*zero_init=*/false);
          cloned_top_level_loop_->body().push_back(current_compute_stage_alloc);
          cloned_top_level_loop_->body().push_back(
              current_compute_stage_->definition());
        }

        // current_load_stage = (loop_index + (stage_depth - 1)) % stage_depth)
        if (current_load_stage_ == nullptr) {
          current_load_stage_ = IrBuilder::modExpr(
              IrBuilder::addExpr(
                  circular_buffer_loop_->index(),
                  IrBuilder::subExpr(
                      IrBuilder::create<Val>(stage_depth, PrimDataType::Index),
                      IrBuilder::create<Val>(1L, PrimDataType::Index))),
              IrBuilder::create<Val>(stage_depth, PrimDataType::Index));
          kir::Allocate* current_load_stage_alloc =
              IrBuilder::create<kir::Allocate>(
                  current_load_stage_,
                  MemoryType::Local,
                  IrBuilder::create<Val>(1L, PrimDataType::Index),
                  /*zero_init=*/false);
          cloned_top_level_loop_->body().push_back(current_load_stage_alloc);
          cloned_top_level_loop_->body().push_back(
              current_load_stage_->definition());
        }

        // Replace LoadStoreOp with:
        //  if (elect_sync()) {
        //    token[next_stage] =
        //      mbarrier::arriveExpectTx(mbarrier[next_stage])
        //    cpAsyncBulk(mbarrier[next_stage],...)
        //  }
        //  mbarrier::wait(token[curr_stage])
        //
        // Where mbarrier and token are shared memory arrays bound to the
        // LoadStoreOp

        LoadStoreOp* ldst = expr->as<LoadStoreOp>();

        // NOTE What happens with multiple ldst for different tensors
        // There should be a single mbarrier_arrive_tx_ for all ldst in current
        // stage.
        NVF_ERROR(mbarrier_arrive_tx_ == nullptr);
        mbarrier_arrive_tx_ =
            createMbarrierArriveExpectTx(ldst, current_load_stage_);

        // Register mbarrier object to be used with LoadStoreOp
        //  from main loop
        NVF_ERROR(mbarrier_arrive_tx_->mbarrier()->isA<kir::TensorIndex>());
        GpuLower::current()->ldstMBarrierIndexMap().emplace(
            ldst, mbarrier_arrive_tx_->mbarrier()->as<kir::TensorIndex>());

        // Construct mBarrier::wait for current stage
        NVF_ERROR(
            mbarrier_wait_ == nullptr,
            "Expected mbarrier_wait to inactive for current TMA operation");
        mbarrier_wait_ = createMbarrierWait(ldst, current_compute_stage_);

        // If last cloned scope is the cloned_top_level_loop body, then add
        // mbarrier::arriveExpectTx and new loadStoreOp.
        int64_t active_for_loops = std::count_if(
            for_loop_id_stack_.begin(),
            for_loop_id_stack_.end(),
            [](IterDomain* id) {
              return id->getParallelType() == ParallelType::Serial;
            });
        if (active_for_loops == 1) {
          kir::IfThenElse* if_expr = createThreadPredicatedIfThenElse();
          Scope& body = if_expr->thenBody();
          body.push_back(mbarrier_arrive_tx_);
          body.push_back(ldst);
          cloned_scopes_.back()->push_back(if_expr);
          mbarrier_arrive_tx_ = nullptr;
          break;
        }

        // Otherwise, we are in a nested for-loop and should wait until we
        // return to top-level for loop.
        cloned_scopes_.back()->push_back(ldst);
        break;
      }
      case CircularBufferLoopStage::Epilog: {
        // Skip shared memory allocation, mbarrier initialize and mbarrier
        // invalidate
        if (is_ignorable_tma_smem_alloc || is_ignorable_mbarrier_init ||
            is_ignorable_mbarrier_inval) {
          break;
        }

        // Add expression if not circular-buffered load store operation
        if (!expr->isA<LoadStoreOp>() || !mbarrier_token_exists) {
          cloned_scopes_.back()->push_back(expr);
          break;
        }

        // Construct mBarrier::wait for last stage
        LoadStoreOp* ldst = expr->as<LoadStoreOp>();
        Val* epilogue_compute_stage = IrBuilder::modExpr(
            cloned_top_level_loop_->index(),
            IrBuilder::create<Val>(stage_depth, PrimDataType::Index));

        int64_t active_for_loops = std::count_if(
            for_loop_id_stack_.begin(),
            for_loop_id_stack_.end(),
            [](IterDomain* id) {
              return id->getParallelType() == ParallelType::Serial;
            });
        if (active_for_loops == 1) {
          cloned_scopes_.back()->push_back(
              createMbarrierWait(ldst, epilogue_compute_stage));
          break;
        }

        NVF_ERROR(
            mbarrier_wait_ == nullptr,
            "Expected mbarrier_wait to inactive for current TMA operation");
        mbarrier_wait_ = createMbarrierWait(ldst, epilogue_compute_stage);
        break;
      }
      case CircularBufferLoopStage::NotApplicable: {
        NVF_ERROR(false, "Unsupported loop mode, got: ", loop_type_);
      }
    }
  }

 private:
  // Track iterDomain associated with each for-loop
  std::vector<IterDomain*> for_loop_id_stack_;

  // Mbarrier_Wait to add to cloned_top_level_loop
  kir::MBarrierWait* mbarrier_wait_ = nullptr;

  // Mbarrier_ArriveExpectTx to add to cloned_top_level_loop
  kir::MBarrierArriveExpectTx* mbarrier_arrive_tx_ = nullptr;

  // current_stage_index = (loop_index % stages)
  Val* current_compute_stage_ = nullptr;

  // next_stage_index = (loop_index + (stages-1)) % stages
  Val* current_load_stage_ = nullptr;
};

using InsertionInfo = std::unordered_map<ForLoop*, std::vector<Expr*>>;

class IsCircularBufferLoadLoop : public kir::IrVisitor {
 public:
  static bool check(
      Expr* expr,
      const std::vector<Expr*>& circular_buffer_load_exprs) {
    IsCircularBufferLoadLoop checker(circular_buffer_load_exprs);
    return checker.check(expr);
  }

 private:
  IsCircularBufferLoadLoop(const std::vector<Expr*>& circular_buffer_load_exprs)
      : circular_buffer_load_exprs_(circular_buffer_load_exprs) {}

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
            circular_buffer_load_exprs_.begin(),
            circular_buffer_load_exprs_.end(),
            expr) != circular_buffer_load_exprs_.end()) {
      result_ = true;
      return;
    }
    IrVisitor::dispatch(expr);
  }

 private:
  const std::vector<Expr*>& circular_buffer_load_exprs_;
  bool result_ = false;
};

// Traverse lowered loop-nests and find all circular buffer loops and
// associated load expressions.
class CircularBufferLoopNestInspector : private kir::IrVisitor {
 public:
  static InsertionInfo run(const std::vector<Expr*>& exprs) {
    CircularBufferLoopNestInspector inspector(exprs);
    return inspector.insertion_info_;
  }

 private:
  CircularBufferLoopNestInspector(const std::vector<Expr*>& exprs) {
    handle(exprs);
  }

  using kir::IrVisitor::handle;

  // Collect circular buffer related information on a expr
  //  that is a memory load, i.e. a LoadStore or a Set.
  void handlePossibleLoadExpr(Expr* expr) {
    auto out_tv = ir_utils::getTvOutput(expr);

    if (out_tv == nullptr) {
      return;
    }

    // Ignore init loop
    if (!out_tv->isCircularBuffered() || !expr->input(0)->isA<TensorView>()) {
      return;
    }

    auto circular_buffer_loop =
        GpuLower::current()->circularBufferInfo().getCircularBufferLoop(
            out_tv, for_loops_);

    NVF_ERROR(
        circular_buffer_loop != nullptr,
        "No circular buffer loop found for a circular buffered tensor: ",
        out_tv->toString());

    validateCircularBufferLoop(circular_buffer_loop);

    insertion_info_[circular_buffer_loop].push_back(expr);
  }

  void handle(UnaryOp* uop) final {
    handlePossibleLoadExpr(uop);
  }

  void handle(LoadStoreOp* ldst) final {
    handlePossibleLoadExpr(ldst);
  }

  static void validateCircularBufferLoop(ForLoop* loop) {
    NVF_ERROR(
        loop->start()->isZeroInt(), "Unsupported loop: ", loop->toString());
    NVF_ERROR(loop->step()->isOneInt(), "Unsupported loop: ", loop->toString());
    NVF_ERROR(
        !loop->vectorize(),
        "Vectorized loop should not be the allocation loop for circular-buffered tensor: ",
        loop->toString());
    NVF_ERROR(
        !loop->vectorize_shift(),
        "Vectorize shift loop should not be the allocation loop for circular-buffered tensor: ",
        loop->toString());
  }

  InsertionInfo insertion_info_;
};

// TODO Move to CircularBufferLoopCloner
// Creates pre-prologue section necessary for proper handling async TMA memory
// operations. It moves the allocation of mbarriers and its tokens outside of
// the main loop
//
// Expected result:
//   Allocate mbarriers and tokens in shared memory
//   if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
//     for (unsigned i = 0; i < stages; ++i) {
//       mbarrier::init(...);
//     }
//   }
class CpAsyncBulkPrePrologue : public kir::IrVisitor {
 public:
  static std::vector<Expr*> create(
      ForLoop* circular_buffer_loop,
      const std::vector<Expr*>& circular_buffer_load_exprs) {
    CpAsyncBulkPrePrologue creator(
        circular_buffer_loop, circular_buffer_load_exprs);
    creator.create();
    return creator.pre_prologue_exprs_;
  }

 private:
  CpAsyncBulkPrePrologue(
      ForLoop* circular_buffer_loop,
      const std::vector<Expr*>& circular_buffer_load_exprs)
      : circular_buffer_loop_(circular_buffer_loop),
        circular_buffer_load_exprs_(circular_buffer_load_exprs) {}

  using kir::IrVisitor::handle;

  void create() {
    // Find and add smem allocations for tokens and mbarrier objects
    handle(circular_buffer_loop_);

    // Define how many threads should arrive at the barrier
    //  we expect 0th thread to handle init/arrive & transaction config
    //  while other threads will wait for it
    Val* one_val = IrBuilder::create<Val>(1L, PrimDataType::UInt32);

    // Construct predicate
    // 'threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0'
    kir::IfThenElse* if_expr = createThreadPredicatedIfThenElse();

    // Construct for loop, a body for if expression
    ForLoop* loop = createStagesForLoop(circular_buffer_loop_);
    NVF_ERROR(loop->simplifiedStop() == loop->stop());

    // Construct loop body with:
    // - mBarriers' initializations for each element in smem array for
    //   each circular buffered tensor
    // - expected arrival: number of threads in the block
    for (const Expr* ldst : circular_buffer_load_exprs_) {
      if (GpuLower::current()->ldstMBarrierMap().count(ldst) != 0) {
        TensorView* all_mbarriers =
            GpuLower::current()->ldstMBarrierMap().at(ldst);
        kir::TensorIndex* stage_mbarrier =
            IrBuilder::create<kir::TensorIndex>(all_mbarriers, loop->index());
        kir::MBarrierInit* mbarrier_init =
            IrBuilder::create<kir::MBarrierInit>(stage_mbarrier, one_val);
        loop->body().push_back(mbarrier_init);
      }
    }

    if_expr->thenBody().push_back(loop);

    pre_prologue_exprs_.push_back(if_expr);
  }

  void handle(ForLoop* fl) final {
    kir::IrVisitor::handle(fl);
  }

  void handle(kir::IfThenElse* ite) final {
    NVF_ERROR(false, "No IfThenElse should exist yet");
  }

  void dispatch(Expr* expr) final {
    if (expr->isA<ForLoop>() || expr->isA<kir::IfThenElse>()) {
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
  ForLoop* circular_buffer_loop_ = nullptr;
  const std::vector<Expr*>& circular_buffer_load_exprs_;

  std::vector<Expr*> pre_prologue_exprs_;
};

// TODO Move to CircularBufferLoopCloner
// Creates post-epilogue section needed for releasing mbarriers after TMA memory
// operations.
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
      ForLoop* circular_buffer_loop,
      const std::vector<Expr*>& circular_buffer_load_exprs) {
    CpAsyncBulkPostEpilogue creator(
        circular_buffer_loop, circular_buffer_load_exprs);
    creator.create();
    return creator.post_prologue_exprs_;
  }

 private:
  CpAsyncBulkPostEpilogue(
      ForLoop* circular_buffer_loop,
      const std::vector<Expr*>& circular_buffer_load_exprs)
      : circular_buffer_loop_(circular_buffer_loop),
        circular_buffer_load_exprs_(circular_buffer_load_exprs) {}

  void create() {
    // Construct predicate
    // 'threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0'
    kir::IfThenElse* if_expr = createThreadPredicatedIfThenElse();

    // Construct for loop, a body for if expression
    ForLoop* loop = createStagesForLoop(circular_buffer_loop_);
    NVF_ERROR(loop->simplifiedStop() == loop->stop());

    // Construct loop body with:
    // - mBarriers' invalidation for each element in smem array for
    //   each circular buffered tensor
    for (const Expr* ldst : circular_buffer_load_exprs_) {
      if (GpuLower::current()->ldstMBarrierMap().count(ldst) != 0) {
        TensorView* all_mbarriers =
            GpuLower::current()->ldstMBarrierMap()[ldst];
        kir::TensorIndex* stage_mbarrier =
            IrBuilder::create<kir::TensorIndex>(all_mbarriers, loop->index());
        kir::MBarrierInvalidate* mbarrier_inval =
            IrBuilder::create<kir::MBarrierInvalidate>(stage_mbarrier);
        loop->body().push_back(mbarrier_inval);
      }
    }

    if_expr->thenBody().push_back(loop);

    post_prologue_exprs_.push_back(if_expr);
  }

 private:
  ForLoop* circular_buffer_loop_ = nullptr;
  const std::vector<Expr*>& circular_buffer_load_exprs_;

  std::vector<Expr*> post_prologue_exprs_;
};

namespace {

void getAllocInTrivialLoop(ForLoop* fl, std::unordered_set<Expr*>& output) {
  if (!fl->isTrivial()) {
    return;
  }
  for (auto expr : fl->body().exprs()) {
    if (expr->isA<kir::Allocate>()) {
      output.emplace(expr);
    } else if (auto loop = dynamic_cast<ForLoop*>(expr)) {
      getAllocInTrivialLoop(loop, output);
    }
  }
}

} // namespace

// Apply circular buffering transformations
class CircularBufferInserter : private kir::ExprMutator {
 public:
  // When there exist multiple circular buffer loops, apply
  // transformations to inner-most loops first. A single ExprMutator
  // pass can only process one loop.
  static std::vector<Expr*> run(
      const std::vector<Expr*>& exprs,
      InsertionInfo insertion_info) {
    auto inserted_exprs = exprs;
    while (!insertion_info.empty()) {
      CircularBufferInserter inserter(inserted_exprs, insertion_info);
      inserted_exprs = inserter.exprs_;
    }
    return inserted_exprs;
  }

 private:
  CircularBufferInserter(
      const std::vector<Expr*>& exprs,
      InsertionInfo& insertion_info)
      : insertion_info_(insertion_info) {
    auto num_circular_buffer_loops = insertion_info.size();
    traverseAndInsert(exprs);
    NVF_ERROR(processed_loop_ != nullptr);
    NVF_ERROR(insertion_info.size() == num_circular_buffer_loops - 1);
  }

  using kir::ExprMutator::handle;

  void handle(ForLoop* loop) final {
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
      ForLoop* circular_buffer_loop,
      const std::vector<Expr*>& loads) {
    // cpAsyncBulk (with TMA) requires some operations to be done prior to
    //  prologue loop, for example:
    // - smem allocation
    // - initialization of mbarrier objects
    std::vector<Expr*> pre_prologue_exprs =
        CpAsyncBulkPrePrologue::create(circular_buffer_loop, loads);
    if (!pre_prologue_exprs.empty()) {
      for (Expr* expr : pre_prologue_exprs) {
        registerInsertBefore(circular_buffer_loop, expr);
      }
    }

    // cpAsyncBulk (with TMA) block sync prior to entering main loop to
    //  make smem with mbarrier objects is initialized.
    kir::BlockSync* sync = IrBuilder::create<kir::BlockSync>(false);
    registerInsertBefore(circular_buffer_loop, sync);

    ForLoop* prologue_loop = TmaCircularBufferLoopCloner::clone(
        circular_buffer_loop, loads, CircularBufferLoopStage::Prolog);
    registerInsertBefore(circular_buffer_loop, prologue_loop);

    ForLoop* main_loop = TmaCircularBufferLoopCloner::clone(
        circular_buffer_loop, loads, CircularBufferLoopStage::Main);
    registerReplace(circular_buffer_loop, main_loop);

    ForLoop* last_for_loop = circular_buffer_loop;

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
      ForLoop* epilogue_loop = TmaCircularBufferLoopCloner::clone(
          circular_buffer_loop,
          loads,
          CircularBufferLoopStage::Epilog,
          alloc_in_main);
      registerInsertAfter(circular_buffer_loop, epilogue_loop);
      last_for_loop = epilogue_loop;
    }

    std::vector<Expr*> post_epilogue_exprs =
        CpAsyncBulkPostEpilogue::create(circular_buffer_loop, loads);
    if (!post_epilogue_exprs.empty()) {
      for (Expr* expr : post_epilogue_exprs) {
        registerInsertAfter(last_for_loop, expr);
      }
    }
  }

  void insert(ForLoop* circular_buffer_loop, const std::vector<Expr*>& loads) {
    auto prologue_loop = CircularBufferLoopCloner::clone(
        circular_buffer_loop, loads, CircularBufferLoopStage::Prolog);
    registerInsertBefore(circular_buffer_loop, prologue_loop);

    auto write_to_smem =
        std::any_of(loads.begin(), loads.end(), [](const Expr* expr) {
          return expr->output(0)->as<TensorView>()->getMemoryType() ==
              MemoryType::Shared;
        });

    // RAW sync is not inserted for circular buffered tensors. The only
    // exception is the prologue load.
    bool has_cpasync = false;
    if (write_to_smem) {
      // Here the initial sync before entering circular buffer loop is
      //  inserted.

      // If any of the circular buffered tensor in this circular buffer
      //  loop is async copy. We want to wait for the gmem loads to
      //  finish before synchronizing the block.
      if (std::any_of(loads.begin(), loads.end(), ir_utils::isCpAsyncOp)) {
        int64_t stage_depth =
            GpuLower::current()->circularBufferInfo().getStageDepthFor(
                circular_buffer_loop->iter_domain());
        auto cp_async_wait = IrBuilder::create<kir::AsyncWait>(
            AsyncOpType::CpAsync, stage_depth - 2);
        prologue_loop->body().push_back(
            IrBuilder::create<kir::AsyncCommit>(AsyncOpType::CpAsync));
        registerInsertBefore(circular_buffer_loop, cp_async_wait);
        has_cpasync = true;
      }

      // Insert the initial block sync before entering main loop.
      if (std::any_of(loads.begin(), loads.end(), [](Expr* expr) {
            return GpuLower::current()
                ->syncMap()
                ->needsRawSync(ir_utils::getTvOutput(expr))
                .hasTID();
          })) {
        // If any of the circular buffered loads require sync, as indicated
        //  by sync info map, insert the sync before entering the circular
        //  buffer loop.
        // TODO:
        //  Currently not supporting circular buffer in gmem, but short to mid
        //  term not yet a priority to go for this case.
        auto sync = IrBuilder::create<kir::BlockSync>(false);
        registerInsertBefore(circular_buffer_loop, sync);
      }
    }

    auto main_loop = CircularBufferLoopCloner::clone(
        circular_buffer_loop, loads, CircularBufferLoopStage::Main);

    registerReplace(circular_buffer_loop, main_loop);

    // Insert the wait instruction in this pass instead
    //  of relying on WAR sync pass to do it.
    // The WAR sync pass today would insert the wait function
    //  exactly where we need it but the purpose of this wait
    //  insertion isn't exactly WAR protection.
    //
    // TODO: [Circular Buffer Sync]
    //  We might eventually want to move the block sync inserted
    //   by WAR pass here as well since this sync insertion is kind
    //   of both WAR and RAW (or neither RAW nor WAR, depends
    //   on how we look at it).
    // Eg. in the case when a intermediate
    //   tensor is circular buffered.
    //
    //  __block_sync();    // This is the initial sync
    //  For i in ...       // Circular buffer loop
    //     A[i%2] = ...;
    //     ...  = A[1-i%2];
    //     __block_sync();  // sync within loop
    //     ...
    //  The "sync within loop" can be placed anywhere in the
    //   circular buffer loop while in the case of RAW and WAR
    //   there'd be extra insertion point restrictions.
    //  We are currently not actively exploring opportunities
    //   with this property of "circular buffer sync" so this
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
      registerInsertAfter(circular_buffer_loop, cp_async_wait_all);
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
      auto epilogue_loop = CircularBufferLoopCloner::clone(
          circular_buffer_loop,
          loads,
          CircularBufferLoopStage::Epilog,
          alloc_in_main);
      registerInsertAfter(circular_buffer_loop, epilogue_loop);
    }
  }

  // Simple conservative rule for inserting async copy wait
  //  primitive in the circular buffer loop:
  void insertCpAsyncCommitWaitInMainLoop(
      ForLoop* main_loop,
      const std::vector<Expr*>& loads) {
    NVF_ERROR(
        !main_loop->body().empty(),
        "Circular buffer sync insertion: empty main loop.");
    auto& exprs = main_loop->body().exprs();
    // Note: This pass explicitly assumes that WAR sync has been
    //  inserted so would need to be updated if we re-order the
    //  passes. Cleanups suggested in [Circular Buffer Sync]
    //  would resolve this dependency on pass ordering.
    int64_t stage_depth =
        GpuLower::current()->circularBufferInfo().getStageDepthFor(
            main_loop->iter_domain());
    auto cp_async_commit =
        IrBuilder::create<kir::AsyncCommit>(AsyncOpType::CpAsync);
    auto cp_async_wait = IrBuilder::create<kir::AsyncWait>(
        AsyncOpType::CpAsync, stage_depth - 2);

    // Find the last circular buffer load in the main loop, and insert
    // cp.async.commit after it.
    std::vector<Expr*>::const_iterator last_circular_buffer_load = exprs.end();
    for (auto it = exprs.begin(); it != exprs.end(); ++it) {
      if (IsCircularBufferLoadLoop::check(*it, loads)) {
        last_circular_buffer_load = it;
      }
    }
    NVF_ERROR(last_circular_buffer_load != exprs.end());
    std::vector<Expr*>::const_iterator commit_it = main_loop->body().insert(
        last_circular_buffer_load + 1, cp_async_commit);

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
  ForLoop* processed_loop_ = nullptr;
};

} // namespace

std::vector<Expr*> CircularBufferPass::run(const std::vector<Expr*>& exprs) {
  auto insertion_info = CircularBufferLoopNestInspector::run(exprs);
  return CircularBufferInserter::run(exprs, insertion_info);
}

} // namespace nvfuser
