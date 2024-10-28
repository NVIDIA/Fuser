// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <id_model/utils.h>
#include <ir/utils.h>
#include <kernel_ir.h>

#include <device_lower/pass/circular_buffer.h>

#include <algorithm>
#include <iterator>
#include <vector>

namespace nvfuser {

namespace {

// The epilogue loop is only created when the producer of a circular
// buffer tensor is on smem, in which case it would otherwise require
// an additional predicate to guard buffer overruns. When it's on
// gmem, that isn't the case, so it does not need to create an
// epilogue loop. For TMA cpAsyncBulk, there is always an epilogue loop.
bool requireEpilogue(const std::vector<Expr*>& exprs) {
  return std::any_of(exprs.begin(), exprs.end(), [](const Expr* expr) {
    return (expr->input(0)->as<TensorView>()->getMemoryType() ==
            MemoryType::Shared) ||
        ir_utils::isCpAsyncBulk(expr);
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

 protected:
  CircularBufferLoopCloner(
      ForLoop* circular_buffer_loop,
      const std::vector<Expr*>& circular_buffer_load_exprs,
      CircularBufferLoopStage loop_type,
      const std::unordered_set<Expr*>& exclude)
      : circular_buffer_loop_(circular_buffer_loop),
        circular_buffer_load_exprs_(circular_buffer_load_exprs),
        loop_type_(loop_type),
        exclude_(exclude) {
    std::transform(
        circular_buffer_load_exprs_.begin(),
        circular_buffer_load_exprs_.end(),
        std::inserter(
            circular_buffer_load_tvs_, circular_buffer_load_tvs_.begin()),
        [](Expr* load_expr) { return ir_utils::getTvOutput(load_expr); });
  }

  using kir::IrVisitor::handle;

  void duplicate() {
    // Cloning the circular buffer loop as follows:
    //
    // Prologue: 0 to prefetch_distance
    // Main: 0 to (extent-prefetch_distance)
    // Epilogue: (extent-prefetch_distance) to extent

    Val* index = GpuLower::current()->getLoopIndexVariable(
        circular_buffer_loop_->iter_domain(), loop_type_);
    Val* start = circular_buffer_loop_->start();
    Val* stop = circular_buffer_loop_->stop();
    int64_t stage_depth =
        GpuLower::current()->circularBufferInfo().getStageDepthFor(
            circular_buffer_loop_->iter_domain());
    int64_t prefetch_distance =
        GpuLower::current()->circularBufferInfo().getPrefetchDistanceFor(
            circular_buffer_loop_->iter_domain());

    switch (loop_type_) {
      case CircularBufferLoopStage::Prolog: {
        NVF_ERROR(start->isZeroInt());
        stop = SimplifyingIrBuilder::create<Val>(
            prefetch_distance, DataType::Index);
        break;
      }
      case CircularBufferLoopStage::Main: {
        if (requireEpilogue(circular_buffer_load_exprs_)) {
          stop = IrBuilder::subExpr(
              circular_buffer_loop_->stop(),
              SimplifyingIrBuilder::create<Val>(
                  prefetch_distance, DataType::Index));
        }
        break;
      }
      case CircularBufferLoopStage::Epilog: {
        NVF_ERROR(requireEpilogue(circular_buffer_load_exprs_));
        start = IrBuilder::subExpr(
            circular_buffer_loop_->stop(),
            SimplifyingIrBuilder::create<Val>(
                prefetch_distance, DataType::Index));
        break;
      }
      case CircularBufferLoopStage::NotApplicable: {
        NVF_THROW("Unsupported loop mode, got: ", loop_type_);
      }
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
        loop_type_,
        /*circular_buffer_loop_stage_depth=*/stage_depth);

    handle(circular_buffer_loop_);
  }

  void handle(ForLoop* fl) override {
    ForLoop* cloned_loop = fl == circular_buffer_loop_
        ? cloned_top_level_loop_
        : IrBuilder::create<ForLoop>(fl);

    // Add to stack
    for_loop_stack_.push_back(cloned_loop);

    // Process for-loop
    kir::IrVisitor::handle(fl);

    // Pop from stack
    for_loop_stack_.pop_back();

    // Specific handling of for-loop
    processForLoop(cloned_loop);
  }

  virtual void processForLoop(ForLoop* cloned_loop) {
    // Add the cloned loop into the parent loop body only when the
    // cloned loop contains expressions.
    if (!cloned_loop->body().empty() && !for_loop_stack_.empty()) {
      for_loop_stack_.back()->body().push_back(cloned_loop);
    }
  }

  void handle(kir::IfThenElse* ite) final {
    NVF_THROW("No IfThenElse should exist yet:\n", ite->toString());
  }

  void dispatch(Expr* expr) override {
    // Skip expression if it is in exclude set
    if (exclude_.count(expr) > 0) {
      return;
    }

    // Handle ForLoop and IfThenElse expr separately
    if (expr->isA<ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::dispatch(expr);
      return;
    }

    NVF_ERROR(!for_loop_stack_.empty());

    // Specific expression handling
    processExpr(expr);
  }

  virtual void processExpr(Expr* expr) {
    switch (loop_type_) {
      case CircularBufferLoopStage::Prolog: {
        // In Prologue, only copy the load expressions.
        // NOTE that there can be multiple expressions defining
        // circular buffered TVs (e.g., buffer initialization).
        TensorView* out_tv = ir_utils::getTvOutput(expr);
        if (circular_buffer_load_tvs_.count(out_tv) > 0) {
          for_loop_stack_.back()->body().push_back(expr);
        }
        break;
      }
      case CircularBufferLoopStage::Main: {
        for_loop_stack_.back()->body().push_back(expr);
        break;
      }
      case CircularBufferLoopStage::Epilog: {
        // In Epilogue, copy everything except circular buffer load expressions.
        TensorView* out_tv = ir_utils::getTvOutput(expr);
        if (circular_buffer_load_tvs_.count(out_tv) == 0) {
          for_loop_stack_.back()->body().push_back(expr);
        }
        break;
      }
      case CircularBufferLoopStage::NotApplicable: {
        NVF_THROW("Unsupported loop mode, got: ", loop_type_);
      }
    }
  }

 protected:
  ForLoop* circular_buffer_loop_ = nullptr;
  const std::vector<Expr*>& circular_buffer_load_exprs_;
  const CircularBufferLoopStage loop_type_;

  std::unordered_set<TensorView*> circular_buffer_load_tvs_;
  ForLoop* cloned_top_level_loop_ = nullptr;
  std::vector<ForLoop*> for_loop_stack_;
  const std::unordered_set<Expr*>& exclude_;
};

// Description:
// Replicates circular buffer loops for Prologue, Main, and
// Epilogue. Prologue only copies the load expressions of circular
// buffered tensors, whereas Epilogue does any expression other than
// the loads. Main copies everything. The pre-prologue and post-epilogue loops
// are created separately by createCpAsyncBulkFixtures.
//
// Loop Structure Overview:
// Pre-prologue loop:
// - Allocate shared memory for mbarriers
// - Initialize mbarrier for all stages
//
// Prologue loop:
// - if selected_thread:
//   - Issue cp async bulks for all but last stage
//
// Main loop:
// - if selected_thread:
//   - Issue next cp async bulk for available stage
// - All threads wait until tma operation arrives
// - Copy body without
//   - shared memory allocations
//   - mbarrier_init exprs
//   - mbarrier_inval exprs
//
// Epilogue loop:
// - All threads wait until tma operation arrives
// - Copy body without
//   - shared memory allocations
//   - issuing cp async bulk operations
//   - mbarrier_init exprs
//   - mbarrier_inval exprs
//
// Post-epilogue loop:
//  - if selected_thread:
//   - Invalidated mbarrier for all stages
//
// Detailed Pseudo-Code:
// Pre-Prologue loop:
//
// - number_of_arrival_threads is the number of threads to call
//   mbarrier::arrive or mbarrier::arriveExpectTx and to wait at
//   mbarrier:wait.
//
// __shared__ __mbarrier_t barriers[num_stages];
// if (warp_id == 0 && electSync()()) {
//   for (int64_t loop_index : irange(stages)) {
//     int64_t number_of_arrive_threads = blockDim.x * blockDim.y * blockDim.z;
//     mbarrier_init(mbarrier[loop_index], number_of_arrival_threads);
//   }
// }
//
// Prologue loop:
// for (int64_t loop_index : irange(prefetch_distance)) {
//   if (warp_id == 0 && electSync()()) {
//     mbarrier::arriveExpectTx(mbarrier[loop_index], expected_bytes);
//     for (...) {
//       cpAsyncBulk(mbarriers[loop_index], ...);
//     }
//   } else {
//     mbarrier::arrive(mbarrier[loop_index]);
//   }
// }
//
// Main loop:
// for (int64_t loop_index : irange(N-prefetch_distance)) {
//   current_stage = loop_index % stage_depth
//   load_stage = (loop_index + prefetch_distance) % stage_depth)
//   if (warp_id == 0 && electSync()()) {
//     mbarrier::arriveExpectTx(mbarrier[load_stage], expected_bytes);
//     for (...) {
//       cpAsyncBulk(mbarrier[load_stage], ...);
//     }
//   } else {
//     mbarrier::arrive(mbarrier[load_stage]);
//   }
//   mbarrier::waitParity((loop_index / stage_depth) % 2);
//
//   Clone remaining operations
// }
//
// Epilogue loop:
// for (int64_t loop_index : irange(N-prefetch_distance, N)) {
//   current_stage = loop_index % stage_depth
//   mbarrier::waitParity((loop_index / stage_depth) % 2);
//
//   Clone remaining operations
// }
//
// Post-Epilogue loop:
// if (warp_id == 0 && electSync()()) {
//   for (int64_t loop_index : irange(stages)) {
//     mbarrier_inval(mbarrier[loop_index]);
//   }
// }
//
class CloneTmaCircularBufferLoopAndInsertSync
    : public CircularBufferLoopCloner {
 public:
  static ForLoop* clone(
      ForLoop* circular_buffer_loop,
      const std::vector<Expr*>& circular_buffer_load_exprs,
      CircularBufferLoopStage loop_type,
      const std::unordered_set<Expr*>& exclude = {}) {
    CloneTmaCircularBufferLoopAndInsertSync cloner(
        circular_buffer_loop, circular_buffer_load_exprs, loop_type, exclude);
    cloner.duplicate();
    return cloner.cloned_top_level_loop_;
  }

 private:
  CloneTmaCircularBufferLoopAndInsertSync(
      ForLoop* circular_buffer_loop,
      const std::vector<Expr*>& circular_buffer_load_exprs,
      CircularBufferLoopStage loop_type,
      const std::unordered_set<Expr*>& exclude)
      : CircularBufferLoopCloner(
            circular_buffer_loop,
            circular_buffer_load_exprs,
            loop_type,
            exclude),
        mbarriers_to_wait_(getAllMbarriersToWaitFor()),
        circular_buffer_load_tvs_(
            GpuLower::current()->circularBufferInfo().getCircularBufferTvs(
                circular_buffer_loop_)) {}

  // For TmaCircularBufferLoop, we have an mbarrier for each Tensorview and
  // each circular buffer stage, but not for each individual TMA load
  // operation. If there are serial IterDomains to the right of the computeAt
  // position, nvfuser will generate a for-loop to launch multiple TMA load
  // operations. This for-loop is passed to processForLoop as the cloned_loop
  // argument.
  //
  // When we encounter a CpAsyncBulk load expression, we create a mbarrier_wait
  // for the main and epilogue loops and a arriveExpectTx for prologue and main
  // loops. handleMainLoop and handleEpilogLoop create mbarrier_wait expression.
  // handleMainLoop and handlePrologLoop create mbarrier::arriveExpectTx
  // expression. The expected_tx for arriveExpectTx is the cumulative
  // transaction size for all TMA load operations for the TensorView. Next, we
  // generate the nested for-loops for the serial IterDomains, but do not add
  // them to the cloned circular buffer loop immediately. Once the cloned
  // circular buffer loop is the only loop in the stack, add the arriveExpectTx
  // and arrive expressions, then the nested for-loop structure calling the TMA
  // load operations, and finally the mbarrier_wait.
  void processForLoop(ForLoop* cloned_loop) final {
    // Skip if there is not an active for-loop structure
    if (for_loop_stack_.empty()) {
      return;
    }

    // If any of the mbarrier wait expressions is not nullptr, this indicates
    // that the `cloned_loop` is the loop containing the first read of a
    // circular buffered TensorView. We need to insert that wait expression
    // before `cloned_loop`.
    if (onlyOneSerialForLoopOnStack()) {
      for (auto it = mbarriers_to_wait_.begin();
           it != mbarriers_to_wait_.end();) {
        auto wait = it->second;
        // short-circuit: wait expression does not exist yet for mbarrier.
        // This means: the mbarrier is used by the circular buffer loop for
        // waiting for its loads, but we have not encountered the first read of
        // the circular buffer yet, so no need to wait right now.
        if (wait == nullptr) {
          ++it;
          continue;
        }
        for_loop_stack_.back()->body().push_back(wait);
        it = mbarriers_to_wait_.erase(it);
      }
    }

    if (!cloned_loop->body().empty()) {
      // mbarrier_arrive_tx_ is active when we encounter a cpAsyncBulk load
      // operation on a circular buffer TensorView in IrVisitor. A single
      // mbarrier_arrive_tx is active for each TensorView.
      if (mbarrier_arrive_tx_ == nullptr || for_loop_stack_.size() > 1) {
        // Add cloned for_loop when mbarrier_arrive_tx_ is not active or
        // we are within a nested for-loop structure
        for_loop_stack_.back()->body().push_back(cloned_loop);
      } else {
        // mbarrier::arriveExpectTx and TMA load operations occur in prologue
        // and main loops.
        //
        // cloned_loop is nested for-loop containing cpAsyncBulk expressions.
        // addTmaLoadBlock replaces the cloned_loop with:
        //
        // if(elect) {
        //   arriveExpectTx;
        //   for (...) {
        //     cpAsyncBulk;
        //   }
        // } else {
        //   arrive;
        // }
        NVF_ERROR(for_loop_stack_.front() == cloned_top_level_loop_);
        addTmaLoadBlock(cloned_loop);
      }
    }
  }

  // Check if there is only one serial for-loop in the stack
  bool onlyOneSerialForLoopOnStack() const {
    return std::count_if(
               for_loop_stack_.begin(), for_loop_stack_.end(), [](ForLoop* fl) {
                 return fl->iter_domain()->getParallelType() ==
                     ParallelType::Serial;
               }) == 1;
  }

  // Current compute stage: loop_index % stages
  Val* currentComputeStage() const {
    int64_t stage_depth =
        GpuLower::current()->circularBufferInfo().getStageDepthFor(
            circular_buffer_loop_->iter_domain());
    Val* result = IrBuilder::modExpr(
        cloned_top_level_loop_->indexOrStartIfTrivial(),
        IrBuilder::create<Val>(stage_depth, PrimDataType::Index));
    return GpuLower::current()->commonScalarMap().hoistScalar(
        result, for_loop_stack_);
  }

  // Current load stage (for main loop): (loop_index + (stages-1)) % stages
  Val* currentLoadStage() const {
    NVF_ERROR(loop_type_ == CircularBufferLoopStage::Main);
    int64_t stage_depth =
        GpuLower::current()->circularBufferInfo().getStageDepthFor(
            circular_buffer_loop_->iter_domain());
    int64_t prefetch_distance =
        GpuLower::current()->circularBufferInfo().getPrefetchDistanceFor(
            circular_buffer_loop_->iter_domain());

    auto current_load_stage = IrBuilder::modExpr(
        IrBuilder::addExpr(
            cloned_top_level_loop_->indexOrStartIfTrivial(),
            IrBuilder::create<Val>(prefetch_distance, PrimDataType::Index)),
        IrBuilder::create<Val>(stage_depth, PrimDataType::Index));
    return GpuLower::current()->commonScalarMap().hoistScalar(
        current_load_stage, for_loop_stack_);
  }

  // The mbarrier_parity for the current circular buffer stage is:
  //   (loop_index / stage_depth) % 2
  // We have an mbarrier for each circular buffer stage, so loop_index /
  // stage_depth is loop_index_per_stage. The valid values of phaseParity
  // operand are 0 and 1, so we take the modulo of loop_index_per_stage with a
  // divisor of 2. See:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait
  // for reference.
  Val* currentParity() const {
    int64_t stage_depth =
        GpuLower::current()->circularBufferInfo().getStageDepthFor(
            circular_buffer_loop_->iter_domain());

    auto depth = IrBuilder::create<Val>(stage_depth, DataType::UInt32);
    auto two = IrBuilder::create<Val>(2, DataType::UInt32);
    Val* stage_parity = SimplifyingIrBuilder::modExpr(
        SimplifyingIrBuilder::divExpr(
            IrBuilder::maybeCastExpr(
                DataType::UInt32,
                cloned_top_level_loop_->indexOrStartIfTrivial()),
            depth),
        two);
    return GpuLower::current()->commonScalarMap().hoistScalar(
        stage_parity, for_loop_stack_);
  }

  // Check if the given expr is the first read of a circular buffered
  // TensorView. If so, create the mbarrier::wait expression for the
  // corresponding buffer. And if the given expr is on the top-level of the
  // cloned loop, insert the newly created mbarrier::wait expression before the
  // given expr.
  void insertMBarrierWaitBeforeFirstRead(Expr* expr) {
    if (loop_type_ == CircularBufferLoopStage::Prolog) {
      // If we are in the prologue loop, we won't clone expr, so we don't need
      // to insert mbarrier::wait.
      return;
    }

    const auto& ldst_mbarrier_map = GpuLower::current()->ldstMBarrierMap();

    for (auto tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
      // short-circuit: The TensorView input for current expression is not
      // defined by a circular buffered TMA load. So it is unrelated here.
      // Here, we are only interested in inserting mbarrier::wait for the
      // circular buffered TMA loads.
      if (circular_buffer_load_tvs_.count(tv) == 0) {
        continue;
      }
      auto ldst = dynamic_cast<LoadStoreOp*>(tv->definition());
      auto mbarrier_it = ldst_mbarrier_map.find(ldst);
      // short-circuit: Failed to find mbarrier for given TMA load. This could
      // happen when a TV is circular buffered, but not using TMA to load.
      if (mbarrier_it == ldst_mbarrier_map.end()) {
        continue;
      }
      auto mbarrier = mbarrier_it->second;
      auto wait_it = mbarriers_to_wait_.find(mbarrier);
      // short-circuit: mbarrier does not exist in mbarriers_to_wait_, so its
      // corresponding wait expression was already inserted.
      if (wait_it == mbarriers_to_wait_.end()) {
        continue;
      }
      auto& wait = wait_it->second;
      if (wait == nullptr) {
        wait = createMbarrierWait(ldst);
      }
      if (onlyOneSerialForLoopOnStack()) {
        for_loop_stack_.back()->body().push_back(wait_it->second);
        mbarriers_to_wait_.erase(wait_it);
      }
    }
  }

  void processExpr(Expr* expr) final {
    TensorView* out_tv = ir_utils::getTvOutput(expr);
    bool is_circular_buffer_load_expr = std::any_of(
        circular_buffer_load_exprs_.begin(),
        circular_buffer_load_exprs_.end(),
        [out_tv](Expr* load_expr) {
          TensorView* circular_buffer_tv = ir_utils::getTvOutput(load_expr);
          NVF_ERROR(circular_buffer_tv != nullptr);
          return out_tv == circular_buffer_tv;
        });

    insertMBarrierWaitBeforeFirstRead(expr);

    // Handle Short-Circuit conditions
    switch (loop_type_) {
      case CircularBufferLoopStage::Prolog: {
        // Short-circuit: skip expression if it is not circular buffer load
        // expression.
        if (!is_circular_buffer_load_expr) {
          return;
        }

        // Short-circuit: There can be circular buffered loads without
        // cpAsyncBulk load expressions.
        if (!ir_utils::isCpAsyncBulkLoad(expr)) {
          for_loop_stack_.back()->body().push_back(expr);
          return;
        }
        break;
      }
      case CircularBufferLoopStage::Main:
      case CircularBufferLoopStage::Epilog: {
        // Short-circuit: Add expression if not circular-buffered load store
        // operation.
        if (!is_circular_buffer_load_expr || !ir_utils::isCpAsyncBulk(expr)) {
          for_loop_stack_.back()->body().push_back(expr);
          return;
        }
        break;
      }
      case CircularBufferLoopStage::NotApplicable: {
        NVF_ERROR(false, "Unsupported loop mode, got: ", loop_type_);
      }
    }

    // Handle cpAsyncBulk expression with circular buffered TensorView output.
    switch (loop_type_) {
      case CircularBufferLoopStage::Prolog: {
        return handlePrologueLoop(expr);
      }
      case CircularBufferLoopStage::Main: {
        return handleMainLoop(expr);
      }
      case CircularBufferLoopStage::Epilog: {
        return;
      }
      case CircularBufferLoopStage::NotApplicable: {
        NVF_ERROR(false, "Unsupported loop mode, got: ", loop_type_);
      }
    }
  }

  // Replace cpAsyncBulk type LoadStoreOp with:
  //   if (warp_id == 0 && electSync()()) {
  //     mbarrier::arriveExpectTx(mbarrier[loop_index], expected_bytes);
  //     for (...) {
  //       cpAsyncBulk(mbarriers[loop_index], ...);
  //     }
  //   } else {
  //     mbarrier::arrive(mbarrier[loop_index]);
  //   }
  // }
  void handlePrologueLoop(Expr* expr) {
    NVF_ERROR(expr != nullptr);

    // Skip if not LoadStoreOp expression
    if (!expr->isA<LoadStoreOp>()) {
      return;
    }

    LoadStoreOp* ldst = expr->as<LoadStoreOp>();

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
    GpuLower::current()->tmaCircularBufferInfo().recordTensorIndex(
        new_ldst, mbarrier_arrive_tx_->mbarrier()->as<kir::TensorIndex>());

    // If last cloned scope is the cloned_top_level_loop body, then add
    // mbarrier::arriveExpectTx and new loadStoreOp.
    if (onlyOneSerialForLoopOnStack()) {
      return addTmaLoadBlock(new_ldst);
    }

    // Otherwise, we are in a nested for-loop and should wait until we
    // return to top-level for loop.
    for_loop_stack_.back()->body().push_back(new_ldst);
  }

  // Handle cpAsyncBulk type LoadStoreOp that is a circular buffer load
  //
  // compute_stage = loop_index % stage_depth
  // load_stage = (loop_index + prefetch_distance) % stage_depth)
  //
  // Replace LoadStoreOp with:
  //   if (warp_id == 0 && electSync()()) {
  //     mbarrier::arriveExpectTx(mbarrier[load_stage], expected_bytes);
  //     for (...) {
  //       cpAsyncBulk(mbarrier[load_stage], ...);
  //     }
  //   } else {
  //     mbarrier::arrive(mbarrier[load_stage]);
  //   }
  //   mbarrier::wait((loop_index / stage_depth) % 2);
  //
  // Where mbarrier are shared memory arrays bound to the LoadStoreOp
  void handleMainLoop(Expr* expr) {
    NVF_ERROR(expr != nullptr && expr->isA<LoadStoreOp>());

    LoadStoreOp* ldst = expr->as<LoadStoreOp>();

    // There is a single mbarrier_arrive_tx_ for each cpAsyncBulk load
    // expression. A mbarrier_arrive_tx_ for another cpAsyncBulk load expression
    // should not be active.
    NVF_ERROR(mbarrier_arrive_tx_ == nullptr);
    mbarrier_arrive_tx_ =
        createMbarrierArriveExpectTx(ldst, currentLoadStage());

    // Register mbarrier object to be used with LoadStoreOp
    //  from main loop
    NVF_ERROR(mbarrier_arrive_tx_->mbarrier()->isA<kir::TensorIndex>());
    GpuLower::current()->tmaCircularBufferInfo().recordTensorIndex(
        ldst, mbarrier_arrive_tx_->mbarrier()->as<kir::TensorIndex>());

    // If last cloned scope is the cloned_top_level_loop body, then add
    // mbarrier::arriveExpectTx and new loadStoreOp
    if (onlyOneSerialForLoopOnStack()) {
      return addTmaLoadBlock(ldst);
    }

    // Otherwise, we are in a nested for-loop and should wait until we
    // return to top-level for loop.
    for_loop_stack_.back()->body().push_back(ldst);
  }

  // For each mbarrier that is used to wait for the loading of circular buffers
  // in the given loop, create a placeholder (nullptr) for mbarrier_wait
  // expressions
  std::unordered_map<TensorView*, kir::MBarrierWaitParity*>
  getAllMbarriersToWaitFor() {
    const auto& ldst_mbarrier_map = GpuLower::current()->ldstMBarrierMap();
    auto circular_buffer_tvs =
        GpuLower::current()->circularBufferInfo().getCircularBufferTvs(
            circular_buffer_loop_);
    std::unordered_map<TensorView*, kir::MBarrierWaitParity*> wait_exprs;
    for (auto tv : circular_buffer_tvs) {
      LoadStoreOp* ldst = dynamic_cast<LoadStoreOp*>(tv->definition());
      auto mbarrier_it = ldst_mbarrier_map.find(ldst);
      if (mbarrier_it == ldst_mbarrier_map.end()) {
        // This circular buffer tensor does not use mbarrier to synchronize.
        // That is, its definition is not a TMA load operation.
        continue;
      }
      auto mbarrier = mbarrier_it->second;
      wait_exprs[mbarrier] = nullptr;
    }
    return wait_exprs;
  }

  // This function selects a single thread to launch tma load and mbarrier
  // arrive_expected_tx operations. The remaining threads will simply arrive
  // at the mbarrier.
  //
  // Pseudo-code example:
  //  if (warp_id == 0 && electSync()()) {
  //    mbarrier::arriveExpectTx(mbarrier[next_stage], expected_bytes);
  //    for (...) {
  //      cpAsyncBulk(mbarrier[next_stage], ...);
  //    }
  //  } else {
  //    mbarrier::arrive(mbarrier[next_stage]);
  //  }
  //
  //  The expr input argument can be a single cpAsyncBulk expression or a nested
  //  for-loop structure of cpAsyncBulk expressions if there are serial
  //  iterDomains to the right of the computeAt position.
  void addTmaLoadBlock(Expr* expr) {
    NVF_ERROR(mbarrier_arrive_tx_ != nullptr);
    NVF_ERROR(expr != nullptr);

    // Create the if-then-else with electSync() predicate for the arrive expect
    // transaction.
    kir::IfThenElse* if_expr =
        IrBuilder::create<kir::IfThenElse>(expr->predicate());

    // A single thread issues arriveExpectTx with expected transactions and
    // launches the TMA load.
    if_expr->thenBody().push_back(mbarrier_arrive_tx_);
    if_expr->thenBody().push_back(expr);

    // The other threads issue arriveExpectTx without any expected transactions.
    kir::MBarrierArrive* thread_arrive = IrBuilder::create<kir::MBarrierArrive>(
        /*state=*/nullptr, mbarrier_arrive_tx_->mbarrier());
    if_expr->elseBody().push_back(thread_arrive);
    for_loop_stack_.back()->body().push_back(if_expr);

    mbarrier_arrive_tx_ = nullptr;
  }

  // Get size of tma load in bytes. It is used for expected transaction count in
  // kir::MBarrierArriveExpectTx.
  Val* getSizeOfTmaLoad(LoadStoreOp* ldst) {
    NVF_ERROR(ldst != nullptr);

    TensorView* consumer_tv = ldst->out()->as<TensorView>();
    NVF_ERROR(
        GpuLower::current()->consumerToTMAInfo().count(consumer_tv),
        "Unable to find TMA info for consumer_tv: ",
        consumer_tv->toString());

    // Get expected bytes for given TMA load operation.
    const TMAInfo& tma_info =
        GpuLower::current()->consumerToTMAInfo().at(consumer_tv);
    Val* expected_bytes = tma_info.tileSizeBytes();

    // The expected_bytes for mbarrier::arriveExpectTX must account for all TMA
    // load operations launched for each circular buffer stage. We take the
    // product of all coordinate TMA iterDomains to the right of the circular
    // buffer axis.
    const std::vector<IterDomain*>& loop_domain = consumer_tv->getLoopDomain();
    for (size_t idx = consumer_tv->getComputeAtPosition();
         idx < loop_domain.size();
         ++idx) {
      IterDomain* id = loop_domain.at(idx);
      if (!id->isBroadcast() && !isParallelTypeThread(id->getParallelType()) &&
          id->getParallelType() != ParallelType::Bulk) {
        expected_bytes =
            SimplifyingIrBuilder::mulExpr(expected_bytes, id->extent());
      }
    }
    expected_bytes =
        SimplifyingIrBuilder::maybeCastExpr(DataType::UInt32, expected_bytes);
    return expected_bytes;
  }

  // This function creates kir::MBarrierArriveExpectTx for given LoadStoreOp and
  // circular buffer stage.
  //
  // Example:
  //   mbarrier::arriveExpectTX(toSmem((&barriers[stage])),
  //   getSizeOfTmaLoad(ldst));
  kir::MBarrierArriveExpectTx* createMbarrierArriveExpectTx(
      LoadStoreOp* ldst,
      Val* loop_index) {
    NVF_ERROR(ldst != nullptr);
    NVF_ERROR(loop_index != nullptr);

    loop_index = GpuLower::current()->commonScalarMap().hoistScalar(
        loop_index, for_loop_stack_);

    // Get mbarrier for this circular buffer stage.
    TensorView* all_mbarriers = GpuLower::current()->ldstMBarrierMap().at(ldst);
    kir::TensorIndex* stage_mbarrier =
        IrBuilder::create<kir::TensorIndex>(all_mbarriers, loop_index);

    Val* tx_count = GpuLower::current()->commonScalarMap().hoistScalar(
        getSizeOfTmaLoad(ldst), for_loop_stack_);
    kir::MBarrierArriveExpectTx* mbarrier_arrive_tx =
        IrBuilder::create<kir::MBarrierArriveExpectTx>(
            /*state=*/nullptr, stage_mbarrier, tx_count);

    return mbarrier_arrive_tx;
  }

  // This function creates kir::MBarrierWaitParity for given LoadStoreOp and
  // circular buffer stage.
  kir::MBarrierWaitParity* createMbarrierWait(LoadStoreOp* ldst) {
    NVF_ERROR(ldst != nullptr);

    // Get mbarrier for this circular buffer stage.
    TensorView* all_mbarriers = GpuLower::current()->ldstMBarrierMap().at(ldst);
    kir::TensorIndex* stage_mbarrier = IrBuilder::create<kir::TensorIndex>(
        all_mbarriers, currentComputeStage());

    kir::MBarrierWaitParity* mbarrier_wait =
        IrBuilder::create<kir::MBarrierWaitParity>(
            stage_mbarrier, currentParity());
    return mbarrier_wait;
  }

 private:
  // Mbarriers whose wait is not inserted to the loop yet, and its corresponding
  // wait expression. This map is initialized as:
  //   mbarrier1 -> nullptr
  //   mbarrier2 -> nullptr
  //   ...
  //   mbarrierN -> nullptr
  // Indicating that: In the cloned loop, we need to wait for "mbarrier1",
  // "mbarrier2", ..., "mbarrierN"; however, the wait expressions are not
  // created yet.
  //
  // As we run the traversal, when we encounter a first read of a circular
  // buffered tensor, we create mbarrier::wait by replacing nullptr with the
  // actual wait expression. These wait expressions will be inserted to the
  // cloned loop before the first read, and after insertion, the entry will be
  // removed from this map indicating that this mbarrier is already waited, and
  // we don't need to wait for it again.
  //
  // Note that we intentionally design this map as a mbarrier -> wait, instead
  // of ldst -> wait or tv -> wait, because multiple buffers and TMA load
  // operations can share the same mbarrier. In this case, we only want to
  // create a single wait expression to wait for all of them.
  std::unordered_map<TensorView*, kir::MBarrierWaitParity*> mbarriers_to_wait_;

  // Mbarrier_ArriveExpectTx to add to cloned_top_level_loop
  kir::MBarrierArriveExpectTx* mbarrier_arrive_tx_ = nullptr;

  // The circular buffered TVs for the loop being cloned
  std::unordered_set<const TensorView*> circular_buffer_load_tvs_;
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
    TensorView* out_tv = ir_utils::getTvOutput(expr);

    // Short-Circuit
    if (out_tv == nullptr) {
      return;
    }

    // Ignore initialization loop
    if (!out_tv->isCircularBuffered() || !expr->input(0)->isA<TensorView>()) {
      return;
    }

    ForLoop* circular_buffer_loop =
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

namespace {

void getAllocInTrivialLoop(ForLoop* fl, std::unordered_set<Expr*>& output) {
  if (!fl->isTrivial()) {
    return;
  }
  for (Expr* expr : fl->body().exprs()) {
    if (expr->isA<kir::Allocate>()) {
      output.emplace(expr);
    } else if (ForLoop* loop = dynamic_cast<ForLoop*>(expr)) {
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
    std::vector<Expr*> inserted_exprs = exprs;
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
    size_t num_circular_buffer_loops = insertion_info.size();
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

  bool hasPrefetch(ForLoop* circular_buffer_loop) {
    int64_t prefetch_distance =
        GpuLower::current()->circularBufferInfo().getPrefetchDistanceFor(
            circular_buffer_loop->iter_domain());
    return prefetch_distance > 0;
  }

  void insertTma(
      ForLoop* circular_buffer_loop,
      const std::vector<Expr*>& loads) {
    // Prologue loop:
    //  - launch only
    //  - arrive_expect_tx and tma load operations
    if (hasPrefetch(circular_buffer_loop)) {
      // If there is no prefetch, then we don't need a prologue loop.
      ForLoop* prologue_loop = CloneTmaCircularBufferLoopAndInsertSync::clone(
          circular_buffer_loop, loads, CircularBufferLoopStage::Prolog);
      registerInsertBefore(circular_buffer_loop, prologue_loop);
    }

    // Main loop:
    //  - Launch and wait
    //  - arrive_expect_tx, tma load operations, and mbarrier_wait)
    ForLoop* main_loop = CloneTmaCircularBufferLoopAndInsertSync::clone(
        circular_buffer_loop, loads, CircularBufferLoopStage::Main);
    registerReplace(circular_buffer_loop, main_loop);

    if (!hasPrefetch(circular_buffer_loop)) {
      // If there is no prefetch, then we don't need a epilogue loop.
      return;
    }

    // We can use exclude argument in CloneTmaCircularBufferLoopAndInsertSync
    // clone to avoid duplicating allocations if main loop is trivial.
    std::unordered_set<Expr*> expressions_allocated_in_main_loop;
    getAllocInTrivialLoop(main_loop, expressions_allocated_in_main_loop);

    // Epilogue loop:
    //  - wait only
    //  - mbarrier_wait
    ForLoop* epilogue_loop = CloneTmaCircularBufferLoopAndInsertSync::clone(
        circular_buffer_loop,
        loads,
        CircularBufferLoopStage::Epilog,
        expressions_allocated_in_main_loop);
    registerInsertAfter(circular_buffer_loop, epilogue_loop);
  }

  void insert(ForLoop* circular_buffer_loop, const std::vector<Expr*>& loads) {
    ForLoop* prologue_loop = nullptr;
    if (hasPrefetch(circular_buffer_loop)) {
      // If there is no prefetch, then we don't need a prologue loop.
      prologue_loop = CircularBufferLoopCloner::clone(
          circular_buffer_loop, loads, CircularBufferLoopStage::Prolog);
      registerInsertBefore(circular_buffer_loop, prologue_loop);
    }

    bool write_to_smem =
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
      has_cpasync =
          std::any_of(loads.begin(), loads.end(), ir_utils::isCpAsyncOp);
      if (prologue_loop != nullptr && has_cpasync) {
        int64_t prefetch_distance =
            GpuLower::current()->circularBufferInfo().getPrefetchDistanceFor(
                circular_buffer_loop->iter_domain());
        kir::AsyncWait* cp_async_wait = IrBuilder::create<kir::AsyncWait>(
            AsyncOpType::CpAsync, prefetch_distance - 1);
        prologue_loop->body().push_back(
            IrBuilder::create<kir::AsyncCommit>(AsyncOpType::CpAsync));
        registerInsertBefore(circular_buffer_loop, cp_async_wait);
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
        kir::BlockSync* sync = IrBuilder::create<kir::BlockSync>(false);
        registerInsertBefore(circular_buffer_loop, sync);
      }
    }

    ForLoop* main_loop = CircularBufferLoopCloner::clone(
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
      kir::AsyncWait* cp_async_wait_all =
          IrBuilder::create<kir::AsyncWait>(AsyncOpType::CpAsync, 0);
      registerInsertAfter(circular_buffer_loop, cp_async_wait_all);
    }

    if (!hasPrefetch(circular_buffer_loop)) {
      // If there is no prefetch, then we don't need a epilogue loop.
      return;
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
      ForLoop* epilogue_loop = CircularBufferLoopCloner::clone(
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
    const std::vector<Expr*>& exprs = main_loop->body().exprs();
    // Note: This pass explicitly assumes that WAR sync has been
    //  inserted so would need to be updated if we re-order the
    //  passes. Cleanups suggested in [Circular Buffer Sync]
    //  would resolve this dependency on pass ordering.
    int64_t prefetch_distance =
        GpuLower::current()->circularBufferInfo().getPrefetchDistanceFor(
            main_loop->iter_domain());
    kir::AsyncCommit* cp_async_commit =
        IrBuilder::create<kir::AsyncCommit>(AsyncOpType::CpAsync);
    kir::AsyncWait* cp_async_wait = IrBuilder::create<kir::AsyncWait>(
        AsyncOpType::CpAsync, std::max(0L, prefetch_distance - 1));

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

    if (prefetch_distance == 0) {
      // If there is no prefetch, we must wait immediately after the commit
      // because the consumption of the data is immediate.
      main_loop->body().insert_after(cp_async_commit, cp_async_wait);
    } else {
      // Check if a sync has been inserted by WAR sync pass.
      auto rend = std::make_reverse_iterator(commit_it);
      auto block_sync_it =
          std::find_if(exprs.rbegin(), rend, [](const Expr* expr) {
            return expr->isA<kir::BlockSync>();
          });
      if (block_sync_it == rend) {
        // If there's no sync, i.e. no tensor needs cross thread communication.
        // We still need a wait but it can just be anywhere after the
        // cp.async.commit in the loop. Chose to place at the end arbitrarily.
        main_loop->body().insert_after(exprs.back(), cp_async_wait);
      } else {
        // If a sync has been inserted, wait needs to be placed before the sync.
        main_loop->body().insert_before(*block_sync_it, cp_async_wait);
      }
    }
  }

 private:
  InsertionInfo& insertion_info_;
  ForLoop* processed_loop_ = nullptr;
};

} // namespace

void TmaCircularBufferInfo::recordTensorIndex(
    const Expr* expr,
    kir::TensorIndex* index) {
  NVF_ERROR(ir_utils::isCpAsyncBulkLoad(expr));
  NVF_ERROR(ldst_mbarrier_index_map_.count(expr) == 0);
  ldst_mbarrier_index_map_.emplace(expr, index);
}

bool TmaCircularBufferInfo::existsTensorIndex(const Expr* expr) const {
  return ldst_mbarrier_index_map_.count(expr) != 0;
}

kir::TensorIndex* TmaCircularBufferInfo::getTensorIndex(const Expr* expr) {
  NVF_ERROR(ir_utils::isCpAsyncBulkLoad(expr));
  // short-circuit: expr does not have tensor index.
  if (ldst_mbarrier_index_map_.count(expr) == 0) {
    return nullptr;
  }
  return ldst_mbarrier_index_map_.at(expr);
}

std::vector<Expr*> CircularBufferPass::run(const std::vector<Expr*>& exprs) {
  InsertionInfo insertion_info = CircularBufferLoopNestInspector::run(exprs);
  return CircularBufferInserter::run(exprs, insertion_info);
}

} // namespace nvfuser
