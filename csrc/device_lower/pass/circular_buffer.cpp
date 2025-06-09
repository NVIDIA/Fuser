// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/cuda/CUDAContext.h>
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
    const auto& opt =
        GpuLower::current()->circularBufferInfo().getCircularBufferOptionsFor(
            circular_buffer_loop_->iter_domain());

    switch (loop_type_) {
      case CircularBufferLoopStage::Prolog: {
        NVF_ERROR(start->isZeroInt());
        stop = SimplifyingIrBuilder::create<Val>(opt.prefetch, DataType::Index);
        break;
      }
      case CircularBufferLoopStage::Main: {
        if (requireEpilogue(circular_buffer_load_exprs_)) {
          stop = SimplifyingIrBuilder::subExpr(
              circular_buffer_loop_->stop(),
              SimplifyingIrBuilder::create<Val>(opt.prefetch, DataType::Index));
        }
        break;
      }
      case CircularBufferLoopStage::Epilog: {
        NVF_ERROR(requireEpilogue(circular_buffer_load_exprs_));
        start = SimplifyingIrBuilder::subExpr(
            circular_buffer_loop_->stop(),
            SimplifyingIrBuilder::create<Val>(opt.prefetch, DataType::Index));
        break;
      }
      case CircularBufferLoopStage::AsyncWarp:
      case CircularBufferLoopStage::ComputeWarp: {
        break;
      }
      default: {
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
        /*circular_buffer_loop_stage_depth=*/opt.stage);

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
      default: {
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
// are created separately by the allocation insertion pass.
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
// __shared__ __mbarrier_t barriers[num_stages];
// if (warp_id == 0 && electSync()()) {
//   for (int64_t loop_index : irange(stages)) {
//     mbarrier_init(mbarrier[loop_index], number_of_tma_load_exprs);
//   }
// }
//
// Prologue loop:
// for (int64_t loop_index : irange(prefetch_distance)) {
//   if (warp_id == 0 && electSync()) {
//     mbarrier::arriveExpectTx(mbarrier[loop_index], expected_bytes);
//     for (...) {
//       cpAsyncBulk(mbarriers[loop_index], ...);
//     }
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
      int64_t insertion_position,
      const std::unordered_set<Expr*>& exclude = {}) {
    CloneTmaCircularBufferLoopAndInsertSync cloner(
        circular_buffer_loop,
        circular_buffer_load_exprs,
        loop_type,
        insertion_position,
        exclude);
    cloner.duplicate();
    return cloner.cloned_top_level_loop_;
  }

 private:
  CloneTmaCircularBufferLoopAndInsertSync(
      ForLoop* circular_buffer_loop,
      const std::vector<Expr*>& circular_buffer_load_exprs,
      CircularBufferLoopStage loop_type,
      int64_t insertion_position,
      const std::unordered_set<Expr*>& exclude)
      : CircularBufferLoopCloner(
            circular_buffer_loop,
            circular_buffer_load_exprs,
            loop_type,
            exclude),
        circular_buffer_load_tvs_(
            GpuLower::current()->circularBufferInfo().getCircularBufferTvs(
                circular_buffer_loop_)),
        raw_mbarriers_to_wait_(getAllMbarriersToWait()),
        war_mbarriers_to_uses_(getAllWarMbarriersToUses()),
        war_mbarriers_to_wait_(getAllMbarriersToWait()),
        insertion_position_(insertion_position) {}

  bool hasCircularBufferLoad() const {
    return nvfuser::hasCircularBufferLoad(loop_type_);
  }

  bool hasCircularBufferConsume() const {
    return nvfuser::hasCircularBufferConsume(loop_type_);
  }

  bool mayHaveWarHazard() const {
    return nvfuser::mayHaveWarHazard(loop_type_);
  }

  bool usesMBarrierForWAR() const {
    return GpuLower::current()
        ->circularBufferInfo()
        .getCircularBufferOptionsFor(circular_buffer_loop_->iter_domain())
        .usesMBarrierForWAR();
  }

  // If any of the mbarrier wait expressions in raw_mbarriers_to_wait_ is not
  // nullptr, this indicates that we are about to insert the reading of the
  // circular buffer tensor into the cloned loop. Before doing that, we need to
  // insert mbarrier::waitParity expressions to wait for the completion of the
  // load of the circular buffer tensor.
  void insertMBarrierWaitBeforeFirstRead() {
    // short-circuit: only insert mbarrier::waitParity when for_loop_stack_
    // is at the insertion position.
    if ((int64_t)for_loop_stack_.size() != insertion_position_) {
      return;
    }

    for (auto it = raw_mbarriers_to_wait_.begin();
         it != raw_mbarriers_to_wait_.end();) {
      auto wait = it->second;
      // short-circuit: wait expression does not exist yet for mbarrier.
      // This means: the mbarrier is used by the circular buffer for loop
      // to wait for its loads. However, we have not encountered the first
      // read of the circular buffer yet, so no need to wait right now.
      if (wait == nullptr) {
        ++it;
        continue;
      }
      for_loop_stack_.back()->body().push_back(wait);
      it = raw_mbarriers_to_wait_.erase(it);
    }
  }

  // If we have visited the last use of a circular buffer tensor, then we
  // insert a mbarrier::arrive to signal that we have done with the reading
  // of the buffer and it is ready to be loaded with new data.
  void insertWarMBarrierArriveAfterLastRead() {
    if (!usesMBarrierForWAR() || !mayHaveWarHazard() ||
        !hasCircularBufferConsume()) {
      return;
    }
    // Only insert arrive on the top-level loop
    if ((int64_t)for_loop_stack_.size() != insertion_position_) {
      return;
    }
    NVF_ERROR(for_loop_stack_.front() == cloned_top_level_loop_);
    for (auto it = war_mbarriers_to_uses_.begin();
         it != war_mbarriers_to_uses_.end();) {
      auto& uses = it->second;
      if (uses.empty()) {
        auto arrive = createWarMbarrierArrive(it->first);
        for_loop_stack_.back()->body().push_back(arrive);
        it = war_mbarriers_to_uses_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // If there is a persistent outer-loop, this IfThenElse short-circuits
  // computation for quantized waves.
  kir::IfThenElse* createPersistentShortCircuit(
      ForLoop* outer_loop,
      ForLoop* inner_loop) {
    NVF_ERROR(outer_loop != nullptr);
    NVF_ERROR(inner_loop != nullptr);
    NVF_ERROR(outer_loop->index() != nullptr);
    NVF_ERROR(outer_loop->stop() != nullptr);
    NVF_ERROR(inner_loop->index() != nullptr);
    NVF_ERROR(inner_loop->stop() != nullptr);
    NVF_ERROR(outer_loop->iter_domain() != nullptr);
    NVF_ERROR(inner_loop->iter_domain() != nullptr);

    IterDomain* outer_loop_id = outer_loop->iter_domain();
    IterDomain* inner_loop_id = inner_loop->iter_domain();

    // Check that the outer and inner loop iterDomains have the same definition.
    bool has_same_definition = outer_loop_id->definition() &&
        inner_loop_id->definition() &&
        outer_loop_id->definition() == inner_loop_id->definition();
    if (!has_same_definition) {
      return nullptr;
    }

    // Check that outer_fl, inner_fl = split(x,  inner_fl->stop()) where
    // inner_fl->stop() == number of SMs.
    Expr* id_def = outer_loop_id->definition();
    if (!id_def->isA<Split>()) {
      return nullptr;
    }
    Split* id_def_split = id_def->as<Split>();
    if (id_def_split->factor() != inner_loop->stop()) {
      return nullptr;
    }
    const int64_t num_sms =
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    if (!id_def_split->factor()->isConstScalar() ||
        id_def_split->factor()->evaluate().as<int64_t>() != num_sms) {
      return nullptr;
    }

    // Check that the outer loop is a serial for-loop.
    if (outer_loop_id->isParallelized()) {
      return nullptr;
    }

    // Check that the inner loop is grid parallelized.
    if (!inner_loop_id->isBlockDim()) {
      return nullptr;
    }

    // lhs := (outer_fl->index() * inner_fl->stop() + inner_fl->index())
    Val* lhs =
        SimplifyingIrBuilder::mulExpr(outer_loop->index(), inner_loop->stop());
    lhs = SimplifyingIrBuilder::addExpr(lhs, inner_loop->index());

    // Check that outer_loop matches known invariants for persistent kernel.
    IterDomain* outer_id =
        lower_utils::getConcreteLoopID(outer_loop->iterDomain());
    Split* persistent_split = dynamic_cast<Split*>(outer_id->definition());
    NVF_ERROR(
        persistent_split != nullptr,
        "Expected ",
        outer_id->toString(),
        " to be a persistent split");
    Val* presplit_extent = persistent_split->in()->extent();

    // predicate := (lhs >= outer_fl->stop()->definition()->lhs())
    Val* predicate_val = SimplifyingIrBuilder::geExpr(lhs, presplit_extent);
    kir::Predicate* predicate =
        IrBuilder::create<kir::Predicate>(predicate_val);
    kir::IfThenElse* ite = IrBuilder::create<kir::IfThenElse>(predicate);
    kir::Continue* cont = IrBuilder::create<kir::Continue>();
    ite->thenBody().push_back(cont);
    return ite;
  }

  // This function inserts arrive and wait expressions for RAW and WAR
  // hazards. The argument `cloned_loop` can be:
  // 1. A loop containing TMA expressions loading circular buffer tensors.
  // 2. A loop containing an expression that is the first use a circular buffer
  //    tensor.
  // 3. A loop containing an expression that is the last use of a circular
  //    buffer tensor.
  // 4. None of the above.
  //
  // For 4, there is nothing interesting, we just naively add the cloned loop
  // to the parent loop body.
  //
  // For 1, we delegate to addTmaLoadBlock, who is responsible for: i) adding
  // mbarrier::waitParity to avoid WAR hazard, ii) adding the cloned loop
  // containing TMA to the parent loop body, and iii) add the necessary
  // mbarrier::arriveExpectTx expressions to signal that TMA has been issued,
  // and loading in progress.
  //
  // For 2, besides inserting the cloned loop to the parent loop body, we also
  // insert mbarrier::waitParity expressions before the cloned loop to wait for
  // the completion of the load of the circular buffer tensor.
  //
  // For 3, besides inserting the cloned loop to the parent loop body, we also
  // insert mbarrier::arrive expressions after the cloned loop to signal that
  // the reading of the circular buffer tensor is complete, and it is ready to
  // load the buffer with new data.
  void processForLoop(ForLoop* cloned_loop) final {
    // Skip if there is not an active for-loop structure
    if (for_loop_stack_.empty()) {
      return;
    }

    // Create outer for-loop short-circuit to minimize wave quantization.
    // Apply to cloned top-level for-loop and persistent kernels.
    if (for_loop_stack_.size() == 1 && insertion_position_ != 1 &&
        !for_loop_stack_.front()->isTrivial()) {
      kir::IfThenElse* ite =
          createPersistentShortCircuit(for_loop_stack_.front(), cloned_loop);
      if (ite != nullptr) {
        for_loop_stack_.back()->body().push_back(ite);
      }
    }

    insertMBarrierWaitBeforeFirstRead();

    if (!cloned_loop->body().empty()) {
      // mbarrier_arrive_tx_ is active when we encounter a cpAsyncBulk load
      // operation on a circular buffer TensorView in IrVisitor. A single
      // mbarrier_arrive_tx is active for each TensorView.
      if (mbarrier_arrive_tx_ == nullptr ||
          (int64_t)for_loop_stack_.size() > insertion_position_) {
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
        // }
        NVF_ERROR(for_loop_stack_.front() == cloned_top_level_loop_);
        addTmaLoadBlock(cloned_loop);
      }
    }

    insertWarMBarrierArriveAfterLastRead();
  }

  // Get the linear index of all nested serial for-loops. It is used in
  // currentComputeIndex, currentCompletionStage, and currentLoadIndex.
  Val* linearIndex() const {
    return GpuLower::current()
        ->circularBufferInfo()
        .getLinearIndexRelativeForLoopStack(
            for_loop_stack_, insertion_position_);
  }

  // Current compute index: loop_index
  Val* currentComputeIndex() const {
    return linearIndex();
  }

  // Current compute stage: loop_index % stages
  Val* currentComputeStage() const {
    NVF_ERROR(hasCircularBufferConsume());
    int64_t stage_depth =
        GpuLower::current()
            ->circularBufferInfo()
            .getCircularBufferOptionsFor(circular_buffer_loop_->iter_domain())
            .stage;
    Val* result = SimplifyingIrBuilder::modExpr(
        currentComputeIndex(),
        IrBuilder::create<Val>(stage_depth, PrimDataType::Index));
    return GpuLower::current()->commonScalarMap().hoistScalar(
        result, for_loop_stack_);
  }

  // The stage of the completion that we are waiting for in the current
  // iteration of the circular buffer loop.
  // Recall that both the load and compute are pipelined. At each iteration,
  // we load `prefetch` stages ahead of the current compute stage. In order
  // to pipeline the compute as deep as possible, we only wait for the buffer
  // that the next iteration will write to become empty. That is, there are
  // `stages - prefetch - 1` pending computations, and we wait for stage
  //   (loop_index + prefetch + 1) % stages
  Val* currentCompletionStage() const {
    NVF_ERROR(mayHaveWarHazard() && hasCircularBufferConsume());
    const auto& opt =
        GpuLower::current()->circularBufferInfo().getCircularBufferOptionsFor(
            circular_buffer_loop_->iter_domain());
    auto current_load_stage = SimplifyingIrBuilder::modExpr(
        SimplifyingIrBuilder::addExpr(linearIndex(), opt.prefetch + 1),
        IrBuilder::create<Val>(opt.stage, PrimDataType::Index));
    return GpuLower::current()->commonScalarMap().hoistScalar(
        current_load_stage, for_loop_stack_);
  }

  // Current load index:
  // - loop_index + prefetch for main loop
  // - loop_index for prologue
  // - N/A for epilogue
  Val* currentLoadIndex() const {
    int64_t prefetch =
        GpuLower::current()
            ->circularBufferInfo()
            .getCircularBufferOptionsFor(circular_buffer_loop_->iter_domain())
            .prefetch;
    Val* linearized_index = linearIndex();
    if (loop_type_ == CircularBufferLoopStage::Main) {
      auto current_load_index =
          SimplifyingIrBuilder::addExpr(linearized_index, prefetch);
      return GpuLower::current()->commonScalarMap().hoistScalar(
          current_load_index, for_loop_stack_);
    }
    return linearized_index;
  }

  // Current load stage: currentLoadIndex() % stages
  Val* currentLoadStage() const {
    NVF_ERROR(hasCircularBufferLoad());
    int64_t stage =
        GpuLower::current()
            ->circularBufferInfo()
            .getCircularBufferOptionsFor(circular_buffer_loop_->iter_domain())
            .stage;
    auto current_load_stage = SimplifyingIrBuilder::modExpr(
        currentLoadIndex(), IrBuilder::create<Val>(stage, PrimDataType::Index));
    return GpuLower::current()->commonScalarMap().hoistScalar(
        current_load_stage, for_loop_stack_);
  }

  // The mbarrier_parity for the current circular buffer stage is:
  //   (currentComputeIndex() / stage_depth) % 2
  // We have an mbarrier for each circular buffer stage, so
  // currentComputeIndex() / stage_depth is compute_index_per_stage. The valid
  // values of phaseParity operand are 0 and 1, so we take the modulo of
  // compute_index_per_stage with a divisor of 2. See:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait
  // for reference.
  Val* currentRawMbarrierParity() const {
    NVF_ERROR(hasCircularBufferConsume());
    int64_t stage_depth =
        GpuLower::current()
            ->circularBufferInfo()
            .getCircularBufferOptionsFor(circular_buffer_loop_->iter_domain())
            .stage;

    auto depth = IrBuilder::create<Val>(stage_depth, DataType::Index);
    auto two = IrBuilder::create<Val>(2, DataType::Index);
    Val* stage_parity = IrBuilder::maybeCastExpr(
        DataType::UInt32,
        SimplifyingIrBuilder::modExpr(
            SimplifyingIrBuilder::divExpr(currentComputeIndex(), depth), two));
    return GpuLower::current()->commonScalarMap().hoistScalar(
        stage_parity, for_loop_stack_);
  }

  // The parity used for waiting for the WAR mbarrier:
  //   (currentLoadIndex() / stage_depth) % 2
  Val* currentWarMbarrierParity() const {
    NVF_ERROR(mayHaveWarHazard() && hasCircularBufferLoad());
    const auto& opt =
        GpuLower::current()->circularBufferInfo().getCircularBufferOptionsFor(
            circular_buffer_loop_->iter_domain());

    auto depth = IrBuilder::create<Val>(opt.stage, DataType::Index);
    auto two = IrBuilder::create<Val>(2, DataType::Index);
    Val* stage_parity = IrBuilder::maybeCastExpr(
        DataType::UInt32,
        SimplifyingIrBuilder::modExpr(
            SimplifyingIrBuilder::divExpr(currentLoadIndex(), depth), two));
    return GpuLower::current()->commonScalarMap().hoistScalar(
        stage_parity, for_loop_stack_);
  }

  // Check if the given expr is the first read of a circular buffered
  // TensorView. If so, create the mbarrier::wait expression for the
  // corresponding buffer and update raw_mbarriers_to_wait_.
  void updateRawMbarrierToWaitMap(Expr* expr) {
    if (!hasCircularBufferConsume()) {
      // expr won't be cloned, so nothing to worry about RAW hazards
      return;
    }

    const auto& ldst_mbarrier_map = GpuLower::current()->mbarrierMap();

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
      auto wait_it = raw_mbarriers_to_wait_.find(mbarrier);
      // short-circuit: mbarrier does not exist in raw_mbarriers_to_wait_, so
      // its corresponding wait expression was already inserted.
      if (wait_it == raw_mbarriers_to_wait_.end()) {
        continue;
      }
      auto& wait = wait_it->second;
      if (wait == nullptr) {
        wait = createMbarrierWaitForRaw(ldst);
      }
    }
  }

  // Check if the given expr is the load of a circular buffered TensorView. If
  // so, create the mbarrier::wait expression for the corresponding buffer and
  // update war_mbarriers_to_wait_.
  void updateWarMbarrierToWaitMap(Expr* expr) {
    if (!usesMBarrierForWAR() || !mayHaveWarHazard() ||
        !hasCircularBufferLoad()) {
      return;
    }
    const auto& ldst_mbarrier_map = GpuLower::current()->mbarrierMap();

    for (auto tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
      // short-circuit: The current expression is not a circular buffer load, so
      // it is unrelated here. Here, we are only interested in inserting
      // mbarrier::wait for the circular buffered TMA loads.
      if (circular_buffer_load_tvs_.count(tv) == 0) {
        continue;
      }
      auto ldst = dynamic_cast<LoadStoreOp*>(expr);
      auto mbarrier_it = ldst_mbarrier_map.find(ldst);
      // short-circuit: Failed to find mbarrier for given TMA load. This could
      // happen when a TV is circular buffered, but not using TMA to load.
      if (mbarrier_it == ldst_mbarrier_map.end()) {
        continue;
      }
      auto mbarrier = mbarrier_it->second;
      auto wait_it = war_mbarriers_to_wait_.find(mbarrier);
      // short-circuit: mbarrier does not exist in war_mbarriers_to_wait_, so
      // its corresponding wait expression was already inserted.
      if (wait_it == war_mbarriers_to_wait_.end()) {
        continue;
      }
      auto& wait = wait_it->second;
      if (wait == nullptr) {
        wait = createWarMbarrierWait(ldst);
      }
    }
  }

  // Check if the given expr is a read of a circular buffered TensorView. If so,
  // update war_mbarriers_to_uses_.
  void updateWarMbarrierUseMap(Expr* expr) {
    if (!usesMBarrierForWAR()) {
      return;
    }

    const auto& ldst_mbarrier_map = GpuLower::current()->mbarrierMap();
    // remove expr from war_mbarriers_to_uses_
    auto input_tvs = ir_utils::filterByType<TensorView>(expr->inputs());
    for (auto tv : input_tvs) {
      if (circular_buffer_load_tvs_.count(tv) == 0) {
        continue;
      }
      auto ldst = dynamic_cast<LoadStoreOp*>(tv->definition());
      auto mbarrier_it = ldst_mbarrier_map.find(ldst);
      if (mbarrier_it == ldst_mbarrier_map.end()) {
        continue;
      }
      auto mbarrier = mbarrier_it->second;
      auto use_it = war_mbarriers_to_uses_.find(mbarrier);
      if (use_it == war_mbarriers_to_uses_.end()) {
        continue;
      }
      auto& uses = use_it->second;
      uses.erase(expr);
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
    bool is_cp_async_bulk_expr = ir_utils::isCpAsyncBulk(expr);

    updateRawMbarrierToWaitMap(expr);
    updateWarMbarrierToWaitMap(expr);
    insertMBarrierWaitBeforeFirstRead();

    // If expr is a TMA circular buffer load, then special handling it.
    // Otherwise just add it to the cloned loop body if needed.
    if (hasCircularBufferLoad() && is_circular_buffer_load_expr &&
        is_cp_async_bulk_expr) {
      // TMA circular buffer load expression
      auto ldst = dynamic_cast<LoadStoreOp*>(expr);
      NVF_ERROR(ldst != nullptr);

      // We always clone expr. The reason for cloning is because, one loop
      // before this pass will be cloned as multiple loops in this pass. Cloning
      // makes sure that different loop stage has different LoadStoreOp* for
      // the same operation, so that they can be handled differently in the
      // later passes. Depending on the setup of circular buffer options, and
      // the current loop stage, cloning may or may not be strictly necessary,
      // but it is not harmful to just clone it.
      Expr* new_ldst =
          IrBuilder::create<LoadStoreOp>(
              ldst->opType(), ldst->out(), ldst->in(), ldst->cacheOp())
              ->withPredicate(ldst->predicate());

      // Create mbarrier_arrive_tx_. Note that mbarrier_arrive_tx_ is created
      // here when we are processing a cpAsyncBulk load expression, but added to
      // the cloned loop body by addTmaLoadBlock either here or later when we
      // are exiting the last cloned scope containing the cpAsyncBulk load
      // expression.
      NVF_ERROR(
          mbarrier_arrive_tx_ == nullptr,
          "There is a single mbarrier_arrive_tx_ for each cpAsyncBulk load "
          "expression. ",
          "A mbarrier_arrive_tx_ for another cpAsyncBulk load expression "
          "should not be active.");
      mbarrier_arrive_tx_ = createRawMbarrierArriveExpectTx(ldst);
      // Register mbarrier object to be used with the cloned LoadStoreOp
      NVF_ERROR(mbarrier_arrive_tx_->mbarrier()->isA<kir::TensorIndex>());
      GpuLower::current()->tmaCircularBufferInfo().recordTensorIndex(
          new_ldst, mbarrier_arrive_tx_->mbarrier()->as<kir::TensorIndex>());

      // If last cloned scope is the cloned_top_level_loop body, this means that
      // we only have a single TMA instruction in the top level loop for each
      // iteration, then add mbarrier::arriveExpectTx and new loadStoreOp.
      if (for_loop_stack_.size() == 1) {
        NVF_ERROR(for_loop_stack_.front() == cloned_top_level_loop_);
        addTmaLoadBlock(new_ldst);
      } else {
        // Otherwise, in the top-level loop, there is a nested for-loop that
        // issues multiple TMA instructions, and right now, we are in that
        // nested for-loop and handling the cpAsyncBulk in it. In such case, we
        // should wait until we return to top-level for loop and add the
        // mbarrier::arriveExpectTx and new loadStoreOp there.
        for_loop_stack_.back()->body().push_back(new_ldst);
      }
    } else if (
        (is_circular_buffer_load_expr && hasCircularBufferLoad() &&
         !is_cp_async_bulk_expr) ||
        (!is_circular_buffer_load_expr && hasCircularBufferConsume())) {
      // For non-TMA circular buffer loads, and for computes, we just add it to
      // the cloned loop body.
      for_loop_stack_.back()->body().push_back(expr);
    }

    updateWarMbarrierUseMap(expr);
    insertWarMBarrierArriveAfterLastRead();
  }

  // For each mbarrier that is used to wait for the loading of circular buffers
  // in the given loop, create a placeholder (nullptr) for mbarrier_wait
  // expressions
  std::unordered_map<TensorView*, kir::MBarrierWaitParity*>
  getAllMbarriersToWait() {
    const auto& ldst_mbarrier_map = GpuLower::current()->mbarrierMap();
    std::unordered_map<TensorView*, kir::MBarrierWaitParity*> wait_exprs;
    for (auto tv : circular_buffer_load_tvs_) {
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

  // For each mbarrier that is used to wait for the finish reading of circular
  // buffers in the given loop, find all the expressions that use the circular
  // buffer tensor tracked by this mbarrier.
  std::unordered_map<TensorView*, std::unordered_set<Expr*>>
  getAllWarMbarriersToUses() {
    const auto& ldst_mbarrier_map = GpuLower::current()->mbarrierMap();
    std::unordered_map<TensorView*, std::unordered_set<Expr*>> mbarrier_to_uses;
    auto exprs =
        ir_utils::flattenScopedExprs(circular_buffer_loop_->body().exprs());
    for (auto expr : exprs) {
      auto tvs = ir_utils::filterByType<TensorView>(expr->inputs());
      for (auto tv : tvs) {
        if (circular_buffer_load_tvs_.count(tv) == 0) {
          continue;
        }
        LoadStoreOp* ldst = dynamic_cast<LoadStoreOp*>(tv->definition());
        if (ldst == nullptr) {
          continue;
        }
        auto mbarrier_it = ldst_mbarrier_map.find(ldst);
        if (mbarrier_it == ldst_mbarrier_map.end()) {
          // This circular buffer tensor does not use mbarrier to synchronize.
          // That is, its definition is not a TMA load operation.
          continue;
        }
        auto mbarrier = mbarrier_it->second;
        mbarrier_to_uses[mbarrier].insert(expr);
      }
    }
    return mbarrier_to_uses;
  }

  // If there is already an if-then-else with electSync() predicate, use it.
  // Otherwise, create a new one.
  kir::IfThenElse* getElectSyncIfThenElse() {
    if (elect_sync_if_then_else_ == nullptr) {
      elect_sync_if_then_else_ = IrBuilder::create<kir::IfThenElse>(
          IrBuilder::create<kir::Predicate>(PredicateType::ElectSync));
      for_loop_stack_.back()->body().push_back(elect_sync_if_then_else_);
    }
    return elect_sync_if_then_else_;
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
  //  }
  //
  //  The expr input argument can be a single cpAsyncBulk expression or a nested
  //  for-loop structure of cpAsyncBulk expressions if there are serial
  //  iterDomains to the right of the computeAt position.
  void addTmaLoadBlock(Expr* expr) {
    NVF_ERROR(mbarrier_arrive_tx_ != nullptr);
    NVF_ERROR(expr != nullptr);

    // Use the if-then-else with electSync() predicate for the arrive expect
    // and cpAsyncBulk operations.
    kir::IfThenElse* if_expr = getElectSyncIfThenElse();

    // Wait for WAR
    if (usesMBarrierForWAR()) {
      for (auto it = war_mbarriers_to_wait_.begin();
           it != war_mbarriers_to_wait_.end();) {
        auto wait = it->second;
        if (wait != nullptr) {
          if_expr->thenBody().push_back(wait);
          it = war_mbarriers_to_wait_.erase(it);
        } else {
          ++it;
        }
      }
    }

    // Arrive expect tx for RAW
    if_expr->thenBody().push_back(mbarrier_arrive_tx_);
    if_expr->thenBody().push_back(expr);

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

    size_t start_idx = consumer_tv->getComputeAtPosition();
    bool is_warp_specialized = std::holds_alternative<WarpSpecialized>(
        consumer_tv->circularBufferOptions().type);
    if (is_warp_specialized) {
      const auto& warp_specialized =
          std::get<WarpSpecialized>(consumer_tv->circularBufferOptions().type);
      if (warp_specialized.stage_slice_position.has_value()) {
        start_idx = warp_specialized.stage_slice_position.value();
      }
    }

    // The expected_bytes for mbarrier::arriveExpectTX must account for all TMA
    // load operations launched for each circular buffer stage. We take the
    // product of all coordinate TMA iterDomains to the right of the circular
    // buffer axis.
    const std::vector<IterDomain*>& loop_domain = consumer_tv->getLoopDomain();
    for (size_t idx = start_idx; idx < loop_domain.size(); ++idx) {
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
  kir::MBarrierArriveExpectTx* createRawMbarrierArriveExpectTx(
      LoadStoreOp* ldst) {
    NVF_ERROR(ldst != nullptr);

    // Get mbarrier for this circular buffer stage.
    TensorView* all_mbarriers = GpuLower::current()->mbarrierMap().at(ldst);
    kir::TensorIndex* stage_mbarrier =
        IrBuilder::create<kir::TensorIndex>(all_mbarriers, currentLoadStage());

    Val* tx_count = GpuLower::current()->commonScalarMap().hoistScalar(
        getSizeOfTmaLoad(ldst), for_loop_stack_);
    kir::MBarrierArriveExpectTx* mbarrier_arrive_tx =
        IrBuilder::create<kir::MBarrierArriveExpectTx>(
            /*state=*/nullptr, stage_mbarrier, tx_count);

    return mbarrier_arrive_tx;
  }

  kir::MBarrierArrive* createWarMbarrierArrive(TensorView* all_mbarriers) {
    // Get mbarrier for this circular buffer stage.
    auto stage_depth =
        GpuLower::current()
            ->circularBufferInfo()
            .getCircularBufferOptionsFor(circular_buffer_loop_->iter_domain())
            .stage;

    kir::TensorIndex* stage_mbarrier = IrBuilder::create<kir::TensorIndex>(
        all_mbarriers,
        SimplifyingIrBuilder::addExpr(currentCompletionStage(), stage_depth));
    kir::MBarrierArrive* mbarrier_arrive =
        IrBuilder::create<kir::MBarrierArrive>(
            /*state=*/nullptr, stage_mbarrier);
    return mbarrier_arrive;
  }

  // This function creates kir::MBarrierWaitParity for given LoadStoreOp and
  // circular buffer stage for waiting RAW.
  kir::MBarrierWaitParity* createMbarrierWaitForRaw(LoadStoreOp* ldst) {
    NVF_ERROR(ldst != nullptr);

    // Get mbarrier for this circular buffer stage.
    TensorView* all_mbarriers = GpuLower::current()->mbarrierMap().at(ldst);
    kir::TensorIndex* stage_mbarrier = IrBuilder::create<kir::TensorIndex>(
        all_mbarriers, currentComputeStage());

    kir::MBarrierWaitParity* mbarrier_wait =
        IrBuilder::create<kir::MBarrierWaitParity>(
            stage_mbarrier, currentRawMbarrierParity());
    return mbarrier_wait;
  }

  // This function creates kir::MBarrierWaitParity for given LoadStoreOp and
  // circular buffer stage for waiting WAR.
  kir::MBarrierWaitParity* createWarMbarrierWait(LoadStoreOp* ldst) {
    NVF_ERROR(ldst != nullptr);

    auto stage_depth =
        GpuLower::current()
            ->circularBufferInfo()
            .getCircularBufferOptionsFor(circular_buffer_loop_->iter_domain())
            .stage;

    // Get mbarrier for this circular buffer stage.
    TensorView* all_mbarriers = GpuLower::current()->mbarrierMap().at(ldst);
    kir::TensorIndex* stage_mbarrier = IrBuilder::create<kir::TensorIndex>(
        all_mbarriers,
        SimplifyingIrBuilder::addExpr(currentLoadStage(), stage_depth));

    kir::MBarrierWaitParity* mbarrier_wait =
        IrBuilder::create<kir::MBarrierWaitParity>(
            stage_mbarrier, currentWarMbarrierParity());
    return mbarrier_wait;
  }

 private:
  // The circular buffered TVs for the loop being cloned
  std::unordered_set<const TensorView*> circular_buffer_load_tvs_;

  // Mbarriers whose wait is not inserted to the loop yet, and its corresponding
  // wait expression. This map is initialized as:
  //   mbarrier1 -> nullptr
  //   mbarrier2 -> nullptr
  //   ...
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
  std::unordered_map<TensorView*, kir::MBarrierWaitParity*>
      raw_mbarriers_to_wait_;

  // Mbarriers used for WAR synchronization and the expressions that use their
  // corresponding circular buffer tensors. This map is initialized as:
  //   mbarrier1 -> {expr1, expr2, ...}
  //   mbarrier2 -> {expr3, expr4, ...}
  //   ...
  // indicating that at least one of the inputs of expr1, expr2, ... use a TMA
  // loaded circular buffer tensor, and we use mbarrier1 for WAR
  // synchronization.
  //
  // As we run the traversal, when we encounter an expression in this map, we
  // remove the expression from the set, and if the set becomes empty, we insert
  // the mbarrier::arrive expression to the top level cloned loop after the
  // last read of the circular buffer tensor.
  //
  // Note that we intentionally design this map as a mbarrier -> exprs, instead
  // of ldst -> exprs or tv -> exprs, because multiple buffers and TMA load
  // operations can share the same mbarrier. In this case, we only want to
  // create a single mbarrier::arrive expression to synchronize all of them.
  std::unordered_map<TensorView*, std::unordered_set<Expr*>>
      war_mbarriers_to_uses_;

  // Mbarriers used for WAR synchronization and their corresponding wait parity
  // expressions. This map is initialized as:
  //   mbarrier1 -> nullptr
  //   mbarrier2 -> nullptr
  //   ...
  // indicating that we need to wait for "mbarrier1", "mbarrier2", ...,
  // "mbarrierN" to finish reading the circular buffer tensors. However, the
  // wait expressions are not created yet.
  //
  // As we run the traversal, when we encounter the load of a circular
  // buffered tensor, we create mbarrier::arrive expressions by replacing
  // nullptr with the actual arrive expression. These arrive expressions will be
  // inserted to the elect-sync if-then-else block before the load, and after
  // insertion, the entry will be removed from this map indicating that this
  // mbarrier is already arrived, and we don't need to arrive it again.
  //
  // Note that we intentionally design this map as a mbarrier -> wait, instead
  // of ldst -> wait or tv -> wait, because multiple buffers and TMA load
  // operations can share the same mbarrier. In this case, we only want to
  // create a single wait expression to wait for all of them.
  std::unordered_map<TensorView*, kir::MBarrierWaitParity*>
      war_mbarriers_to_wait_;

  // Mbarrier_ArriveExpectTx to add to cloned_top_level_loop
  kir::MBarrierArriveExpectTx* mbarrier_arrive_tx_ = nullptr;

  // ElectSync if-then-else for the cloned loop. We put all the circular buffer
  // load TMA operations under this if-then-else.
  kir::IfThenElse* elect_sync_if_then_else_ = nullptr;

  // The insertion_point is the number of nested for-loops relative to the
  // top-level cloned for-loop where the mbarrier synchronization is inserted.
  // By default, the insertion_point is 1, which is the top-level cloned
  // for-loop. However, for warp specialization, it can be different.
  int64_t insertion_position_;
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
  static std::pair<InsertionInfo, InsertionInfo> run(
      const std::vector<Expr*>& exprs) {
    CircularBufferLoopNestInspector inspector(exprs);

    // InsertionInfo holds all circular buffer for-loops. Split it into warp
    // specialized and pipeline circular buffers. Enforce that we can only nest
    // pipeline circular buffering inside of warp-specialization.

    // Get WarpSpecialized InsertionInfo
    InsertionInfo ws_info;
    int64_t inner_most_ws_position = -1;
    for (auto&& [cb_loop, cb_exprs] : inspector.insertion_info_) {
      if (!lower_utils::isWarpSpecializedLoop(cb_loop)) {
        continue;
      }
      ws_info[cb_loop] = cb_exprs;
      inner_most_ws_position = std::max(
          inner_most_ws_position, inspector.loop_position_.at(cb_loop));
    }

    // WarpSpecialized circular buffering pads the thread block size by 128
    // threads. This is to support register sharing, which shares registers from
    // four warps to another four warps. Thus, we can have four warps running
    // concurrently in AsyncWarp. Each warp can launch an asynchronous operation
    // with mbarrier completion mechanism such as TMA Load and Blackwell UTCMMA.
    //
    // if (Select AsyncWarp) {
    //   if (Select Warp 0 AND elect-sync()) {
    //     do-something
    //   } else if (Select Warp 1 AND elect-sync()) {
    //     do-something
    //   } else if (Select Warp 2 AND elect-sync()) {
    //      do-something
    //   } else if (Select Warp 3 AND elect-sync()) {
    //      do-something
    //   }
    // }
    NVF_ERROR(
        ws_info.size() <= 4,
        "At most four for-loops can run concurrently inside the AsyncWarp.\n",
        "Detected ",
        ws_info.size(),
        " WarpSpecialized for-loops.");

    // Get Pipeline InsertionInfo
    InsertionInfo pipeline_info;
    for (auto&& [cb_loop, cb_exprs] : inspector.insertion_info_) {
      if (lower_utils::isWarpSpecializedLoop(cb_loop)) {
        continue;
      }

      // An example of WarpSpecialized circular buffer nested in Pipeline
      // circular buffer.
      //  * Register sharing would fail because of the return in the AsyncLoop.
      //  * This scenario is not actively tested, so prohibit it until a valid
      //    use-case occurs.
      //
      // warp-specialized mbarrier init
      // for (prologue) {
      //   load something for Prologue
      // }
      //
      // for (main) {
      //   load something for Main
      //   if (AsyncWarp) {
      //     launch async
      //     maybe return for register sharing
      //   } else {
      //     compute something for ComputeWarp
      //   }
      //   compute something for Main
      // }
      //
      // for (epilogue) {
      //   if (AsyncWarp) {
      //     launch async
      //     maybe return for register sharing
      //   } else {
      //     compute something
      //   }
      //   compute something for Epilogue
      //  }
      // warp-specialized mbarrier inval
      NVF_ERROR(
          inspector.loop_position_.at(cb_loop) > inner_most_ws_position,
          "Warp Specialization cannot be nested in Pipeline circular "
          "buffering!");
      pipeline_info[cb_loop] = cb_exprs;
    }

    return {ws_info, pipeline_info};
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

    auto cb_loop_it =
        std::find(for_loops_.begin(), for_loops_.end(), circular_buffer_loop);
    loop_position_[circular_buffer_loop] =
        std::distance(for_loops_.begin(), cb_loop_it);
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
        "Vectorized loop should not be the allocation loop for "
        "circular-buffered tensor: ",
        loop->toString());
    NVF_ERROR(
        !loop->vectorize_shift(),
        "Vectorize shift loop should not be the allocation loop for "
        "circular-buffered tensor: ",
        loop->toString());
  }

  // Map circular buffer loop to its position in the for_loop_ stack.
  std::unordered_map<ForLoop*, int64_t> loop_position_;
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

// Create something like below:
//   for (int i = 0; i < prefetch + 1; ++i) {
//     mbarrier::arrive(mbarrier0[stage + i]]);
//     mbarrier::arrive(mbarrier1[stage + i]);
//     ...
//   }
// where mbarrierX[stage + i] is the X-th WAR mbarrier for stage i.
//
// This is needed because we prefetch data in circular buffering, and we
// need to make sure the initial prefetches are not blocked by the
// non-existing WAR hazards.
ForLoop* createArrivesForWar(ForLoop* circular_buffer_loop) {
  const auto& opt =
      GpuLower::current()->circularBufferInfo().getCircularBufferOptionsFor(
          circular_buffer_loop->iter_domain());
  auto circular_buffer_tvs =
      GpuLower::current()->circularBufferInfo().getCircularBufferTvs(
          circular_buffer_loop->iter_domain());
  VectorOfUniqueEntries<TensorView*> mbarriers;
  for (auto tv : circular_buffer_tvs) {
    auto ldst = dynamic_cast<LoadStoreOp*>(tv->definition());
    NVF_ERROR(ldst != nullptr);
    auto it = GpuLower::current()->mbarrierMap().find(ldst);
    if (it == GpuLower::current()->mbarrierMap().end()) {
      continue;
    }
    mbarriers.pushBack(it->second);
  }
  auto prefetch_loop = ir_utils::createRangeLoop(opt.prefetch + 1);

  // If compute warp groups are independent, then only the first compute warp
  // group needs to set the arrive of the prefetch loop
  const auto& cb_info = GpuLower::current()->circularBufferInfo();
  bool independent_compute_warp_groups =
      cb_info.hasIndependentComputeWarpGroups();
  kir::IfThenElse* ite = nullptr;
  if (independent_compute_warp_groups) {
    NVF_ERROR(lower_utils::isWarpSpecializedLoop(circular_buffer_loop));
    ParallelType warp_specialize_on = std::get<WarpSpecialized>(opt.type).on;
    Val* predicate_val = SimplifyingIrBuilder::eqExpr(
        NamedScalar::getParallelIndex(warp_specialize_on),
        GpuLower::current()->kernel()->zeroVal());
    kir::Predicate* predicate =
        IrBuilder::create<kir::Predicate>(predicate_val);
    ite = IrBuilder::create<kir::IfThenElse>(predicate);
    prefetch_loop->body().push_back(ite);
  }

  for (auto mbarrier : mbarriers) {
    auto mbarrier_to_arrive = IrBuilder::create<kir::TensorIndex>(
        mbarrier,
        SimplifyingIrBuilder::addExpr(
            prefetch_loop->indexOrStartIfTrivial(), opt.stage));
    auto prefetch = IrBuilder::create<kir::MBarrierArrive>(
        /*state=*/nullptr, mbarrier_to_arrive);
    if (ite != nullptr) {
      ite->thenBody().push_back(prefetch);
    } else {
      prefetch_loop->body().push_back(prefetch);
    }
  }
  return prefetch_loop;
}

} // namespace

// Apply warp specialized circular buffering transformations
class WarpSpecializedCircularBufferInserter : private kir::ExprMutator {
 public:
  // When there exist multiple circular buffer loops, apply
  // transformations to inner-most loops first. A single ExprMutator
  // pass can only process one loop.
  static std::vector<Expr*> run(
      const std::vector<Expr*>& exprs,
      InsertionInfo insertion_info) {
    std::vector<Expr*> inserted_exprs = exprs;
    while (!insertion_info.empty()) {
      WarpSpecializedCircularBufferInserter inserter(
          inserted_exprs, insertion_info);
      inserted_exprs = inserter.exprs_;
    }
    return inserted_exprs;
  }

 private:
  WarpSpecializedCircularBufferInserter(
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

    NVF_ERROR(lower_utils::isWarpSpecializedLoop(loop));
    NVF_ERROR(
        std::all_of(
            it->second.begin(), it->second.end(), ir_utils::isCpAsyncBulk),
        "In order to use warp specialization, all buffers must be loaded by "
        "TMA");
    int64_t insertion_position =
        GpuLower::current()
            ->circularBufferInfo()
            .getCircularBufferInsertionPosition(loop->iter_domain());
    insertTmaWarpSpecialized(loop, it->second, insertion_position);

    processed_loop_ = loop;
    insertion_info_.erase(loop);
  }

  // Create predicate for warp-specialized IfThenElse:
  // kir::Predicate is thread_axis >= block_dim_axis - padded_value
  kir::Predicate* getAsyncWarpPredicate(const CircularBufferOptions& options) {
    ParallelType warp_specialize_on =
        std::get<WarpSpecialized>(options.type).on;
    int64_t warp_specialization_pad =
        GpuLower::current()
            ->parallelDimensionMap()
            .getWarpSpecializationPaddedVal(warp_specialize_on);
    Val* raw =
        GpuLower::current()->parallelDimensionMap().get(warp_specialize_on);
    Val* raw_minus_pad = SimplifyingIrBuilder::subExpr(
        raw, IrBuilder::create<Val>(warp_specialization_pad, DataType::Index));
    return IrBuilder::create<kir::Predicate>(IrBuilder::geExpr(
        NamedScalar::getParallelIndex(warp_specialize_on), raw_minus_pad));
  }

  void insertTmaWarpSpecialized(
      ForLoop* circular_buffer_loop,
      const std::vector<Expr*>& loads,
      int64_t insertion_position) {
    const CircularBufferOptions& options =
        GpuLower::current()->circularBufferInfo().getCircularBufferOptionsFor(
            circular_buffer_loop->iter_domain());

    kir::IfThenElse* warp_dispatch_ite =
        IrBuilder::create<kir::IfThenElse>(getAsyncWarpPredicate(options));

    // Set default value
    bool enable_register_sharing =
        std::get<WarpSpecialized>(options.type).num_registers.has_value();
    GpuLower::current()->kernel()->manage(
        "enable_register_sharing", enable_register_sharing);

    if (enable_register_sharing) {
      auto&& [decrease_num_registers, increase_num_registers] =
          std::get<WarpSpecialized>(options.type).num_registers.value();
      GpuLower::current()->decIncRegisterUsage() =
          std::make_pair(decrease_num_registers, increase_num_registers);
      // Decrease registers in async warp group
      kir::SetMaxNReg* dec_reg_async_warp = IrBuilder::create<kir::SetMaxNReg>(
          IrBuilder::create<Val>(decrease_num_registers, DataType::Index),
          /*increase_registers=*/false);
      warp_dispatch_ite->thenBody().push_back(dec_reg_async_warp);

      // Increase registers in compute warp group
      kir::SetMaxNReg* inc_reg_async_warp = IrBuilder::create<kir::SetMaxNReg>(
          IrBuilder::create<Val>(increase_num_registers, DataType::Index),
          /*increase_registers*/ true);
      warp_dispatch_ite->elseBody().push_back(inc_reg_async_warp);
    }

    // Load loop:
    ForLoop* load_loop = CloneTmaCircularBufferLoopAndInsertSync::clone(
        circular_buffer_loop,
        loads,
        CircularBufferLoopStage::AsyncWarp,
        insertion_position);
    warp_dispatch_ite->thenBody().push_back(load_loop);

    // Terminate the warp group handling Load loop immediately after
    // finishing its work.
    kir::Return* ret = IrBuilder::create<kir::Return>();
    warp_dispatch_ite->thenBody().push_back(ret);

    // Prefetch:
    auto prefetch_loop = createArrivesForWar(circular_buffer_loop);
    warp_dispatch_ite->elseBody().push_back(prefetch_loop);

    // Compute loop:
    ForLoop* compute_loop = CloneTmaCircularBufferLoopAndInsertSync::clone(
        circular_buffer_loop,
        loads,
        CircularBufferLoopStage::ComputeWarp,
        insertion_position);
    warp_dispatch_ite->elseBody().push_back(compute_loop);

    registerReplace(circular_buffer_loop, warp_dispatch_ite);
  }

 private:
  InsertionInfo& insertion_info_;
  ForLoop* processed_loop_ = nullptr;
};

// Apply pipeline circular buffering transformations
class PipelineCircularBufferInserter : private kir::ExprMutator {
 public:
  // When there exist multiple circular buffer loops, apply
  // transformations to inner-most loops first. A single ExprMutator
  // pass can only process one loop.
  static std::vector<Expr*> run(
      const std::vector<Expr*>& exprs,
      InsertionInfo insertion_info) {
    std::vector<Expr*> inserted_exprs = exprs;
    while (!insertion_info.empty()) {
      PipelineCircularBufferInserter inserter(inserted_exprs, insertion_info);
      inserted_exprs = inserter.exprs_;
    }
    return inserted_exprs;
  }

 private:
  PipelineCircularBufferInserter(
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

    NVF_ERROR(!lower_utils::isWarpSpecializedLoop(loop));

    auto has_cp_async_bulk = std::any_of(
        it->second.begin(), it->second.end(), ir_utils::isCpAsyncBulk);
    if (has_cp_async_bulk) {
      insertTmaPipelined(loop, it->second);
    } else {
      insert(loop, it->second);
    }

    processed_loop_ = loop;
    insertion_info_.erase(loop);
  }

  bool hasPrefetch(ForLoop* circular_buffer_loop) {
    int64_t prefetch_distance =
        GpuLower::current()
            ->circularBufferInfo()
            .getCircularBufferOptionsFor(circular_buffer_loop->iter_domain())
            .prefetch;
    return prefetch_distance > 0;
  }

  static bool usesMBarrierForWAR(ForLoop* circular_buffer_loop) {
    return GpuLower::current()
        ->circularBufferInfo()
        .getCircularBufferOptionsFor(circular_buffer_loop->iter_domain())
        .usesMBarrierForWAR();
  }

  void insertTmaPipelined(
      ForLoop* circular_buffer_loop,
      const std::vector<Expr*>& loads) {
    // Arrive on the WAR mbarriers to let the prefetching start.
    if (usesMBarrierForWAR(circular_buffer_loop)) {
      auto prefetch_loop = createArrivesForWar(circular_buffer_loop);
      registerInsertBefore(circular_buffer_loop, prefetch_loop);
    }

    // Prologue loop:
    //  - launch only
    //  - arrive_expect_tx and tma load operations
    if (hasPrefetch(circular_buffer_loop)) {
      // If there is no prefetch, then we don't need a prologue loop.
      ForLoop* prologue_loop = CloneTmaCircularBufferLoopAndInsertSync::clone(
          circular_buffer_loop,
          loads,
          CircularBufferLoopStage::Prolog,
          /*insertion_position=*/1);
      registerInsertBefore(circular_buffer_loop, prologue_loop);
    }

    // Main loop:
    //  - Launch and wait
    //  - arrive_expect_tx, tma load operations, and mbarrier_wait
    ForLoop* main_loop = CloneTmaCircularBufferLoopAndInsertSync::clone(
        circular_buffer_loop,
        loads,
        CircularBufferLoopStage::Main,
        /*insertion_position=*/1);
    registerReplace(circular_buffer_loop, main_loop);

    if (!hasPrefetch(circular_buffer_loop)) {
      // If there is no prefetch, then we don't need a epilogue loop.
      return;
    }

    // We can use exclude argument in
    // CloneTmaCircularBufferLoopAndInsertSync clone to avoid
    // duplicating allocations if main loop is trivial.
    std::unordered_set<Expr*> expressions_allocated_in_main_loop;
    getAllocInTrivialLoop(main_loop, expressions_allocated_in_main_loop);

    // Epilogue loop:
    //  - wait only
    //  - mbarrier_wait
    ForLoop* epilogue_loop = CloneTmaCircularBufferLoopAndInsertSync::clone(
        circular_buffer_loop,
        loads,
        CircularBufferLoopStage::Epilog,
        /*insertion_position=*/1,
        expressions_allocated_in_main_loop);
    registerInsertAfter(circular_buffer_loop, epilogue_loop);
  }

  void insert(ForLoop* circular_buffer_loop, const std::vector<Expr*>& loads) {
    NVF_ERROR(
        !usesMBarrierForWAR(circular_buffer_loop),
        "Circular buffer loop with WAR mbarrier is only supported for TMA");
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
        int64_t prefetch_distance = GpuLower::current()
                                        ->circularBufferInfo()
                                        .getCircularBufferOptionsFor(
                                            circular_buffer_loop->iter_domain())
                                        .prefetch;
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
        GpuLower::current()
            ->circularBufferInfo()
            .getCircularBufferOptionsFor(main_loop->iter_domain())
            .prefetch;
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

ForLoop* HopperPingPongMbarriers::initializePingPongMbarrier() {
  ForLoop* loop = ir_utils::createRangeLoop(num_warp_groups_ * 2);
  Val* num_of_arrives = SimplifyingIrBuilder::maybeCastExpr(
      DataType::UInt32,
      GpuLower::current()
          ->parallelDimensionMap()
          .getNumComputeThreadsEachBlock());
  kir::TensorIndex* ping_pong_mbarrier_index =
      IrBuilder::create<kir::TensorIndex>(mbarriers_, loop->index());
  kir::MBarrierInit* ping_pong_mbarrier_init =
      IrBuilder::create<kir::MBarrierInit>(
          ping_pong_mbarrier_index, num_of_arrives);
  Expr* pred_ping_pong_mbarrier_init = ping_pong_mbarrier_init->withPredicate(
      IrBuilder::create<kir::Predicate>(PredicateType::ElectSync));
  loop->body().push_back(pred_ping_pong_mbarrier_init);
  return loop;
}

ForLoop* HopperPingPongMbarriers::invalidatePingPongMbarrier() {
  ForLoop* loop = ir_utils::createRangeLoop(num_warp_groups_ * 2);
  kir::TensorIndex* ping_pong_mbarrier_index =
      IrBuilder::create<kir::TensorIndex>(mbarriers_, loop->index());
  kir::MBarrierInvalidate* ping_pong_mbarrier_inval =
      IrBuilder::create<kir::MBarrierInvalidate>(ping_pong_mbarrier_index);
  Expr* pred_ping_pong_mbarrier_inval = ping_pong_mbarrier_inval->withPredicate(
      IrBuilder::create<kir::Predicate>(PredicateType::ElectSync));
  loop->body().push_back(pred_ping_pong_mbarrier_inval);
  return loop;
}

// This helper function allocates, initializes and invalidates ping-pong
// mbarriers.
std::tuple<kir::Allocate*, ForLoop*, ForLoop*> HopperPingPongMbarriers::
    createPingPongMbarrier() {
  // For each warp group, we have two mbarriers: one for the TensorCore
  // phase and one for the CUDA Epilogue phase.
  mbarriers_ = TensorViewBuilder()
                   .shape(std::vector<int64_t>{num_warp_groups_ * 2})
                   .dtype(DataType::UInt64)
                   .contiguity(true)
                   .build();
  mbarriers_->setMemoryType(MemoryType::Shared);

  // Allocate memory for ping-pong mbarriers.
  kir::Allocate* ping_pong_mbarrier_alloc =
      IrBuilder::create<kir::Allocate>(mbarriers_, MemoryType::Shared);
  ForLoop* ping_pong_mbarrier_init_raw = initializePingPongMbarrier();
  ForLoop* ping_pong_mbarrier_inval_raw = invalidatePingPongMbarrier();
  return {
      ping_pong_mbarrier_alloc,
      ping_pong_mbarrier_init_raw,
      ping_pong_mbarrier_inval_raw};
}

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
  auto&& [ws_insertion_info, pipeline_insertion_info] =
      CircularBufferLoopNestInspector::run(exprs);
  // Process circular buffer for-loops from inner to outer-most.
  // Pipeline must come before WarpSpecialized. We cannot nest WarpSpecialized
  // inside of Pipeline circular buffering.
  std::vector<Expr*> result_exprs =
      PipelineCircularBufferInserter::run(exprs, pipeline_insertion_info);
  return WarpSpecializedCircularBufferInserter::run(
      result_exprs, ws_insertion_info);
}

} // namespace nvfuser
