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
// epilogue loop.
bool requireEpilogue(const std::vector<Expr*>& exprs) {
  return std::any_of(exprs.begin(), exprs.end(), [](const Expr* expr) {
    return expr->input(0)->as<TensorView>()->getMemoryType() ==
        MemoryType::Shared;
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
    // Prologue: 0 to 1
    // Main: 0 to (extent-1)
    // Epilogue: (extent-1) to extent

    Val* index = GpuLower::current()->caMap()->getIndexVariable(
        circular_buffer_loop_->iter_domain(), loop_type_);
    Val* start = circular_buffer_loop_->start();
    Val* stop = circular_buffer_loop_->stop();
    int64_t stage_depth =
        GpuLower::current()->circularBufferInfo().getStageDepthFor(
            circular_buffer_loop_->iter_domain());

    switch (loop_type_) {
      case CircularBufferLoopStage::Prolog: {
        NVF_ERROR(start->isZeroInt());
        stop = SimplifyingIrBuilder::create<Val>(
            int64_t(stage_depth - 1), DataType::Index);
        break;
      }
      case CircularBufferLoopStage::Main: {
        if (requireEpilogue(circular_buffer_load_exprs_)) {
          stop = IrBuilder::subExpr(
              circular_buffer_loop_->stop(),
              SimplifyingIrBuilder::create<Val>(
                  stage_depth - 1, DataType::Index));
        }
        break;
      }
      case CircularBufferLoopStage::Epilog: {
        NVF_ERROR(requireEpilogue(circular_buffer_load_exprs_));
        start = IrBuilder::subExpr(
            circular_buffer_loop_->stop(),
            SimplifyingIrBuilder::create<Val>(
                stage_depth - 1, DataType::Index));
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
        loop_type_);

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
    NVF_THROW("No IfThenElse should exist yet");
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

    insert(loop, it->second);
    processed_loop_ = loop;
    insertion_info_.erase(loop);
  }

  void insert(ForLoop* circular_buffer_loop, const std::vector<Expr*>& loads) {
    ForLoop* prologue_loop = CircularBufferLoopCloner::clone(
        circular_buffer_loop, loads, CircularBufferLoopStage::Prolog);
    registerInsertBefore(circular_buffer_loop, prologue_loop);

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
      if (std::any_of(loads.begin(), loads.end(), ir_utils::isCpAsyncOp)) {
        int64_t stage_depth =
            GpuLower::current()->circularBufferInfo().getStageDepthFor(
                circular_buffer_loop->iter_domain());
        kir::AsyncWait* cp_async_wait = IrBuilder::create<kir::AsyncWait>(
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
    int64_t stage_depth =
        GpuLower::current()->circularBufferInfo().getStageDepthFor(
            main_loop->iter_domain());
    kir::AsyncCommit* cp_async_commit =
        IrBuilder::create<kir::AsyncCommit>(AsyncOpType::CpAsync);
    kir::AsyncWait* cp_async_wait = IrBuilder::create<kir::AsyncWait>(
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

void TmaCircularBufferInfo::recordMBarrierToken(
    const Expr* expr,
    TensorView* mbarrier_tokens) {
  NVF_ERROR(
      ir_utils::isCpAsyncBulkLoad(expr) || expr->isA<kir::MBarrierInit>() ||
      expr->isA<kir::MBarrierInvalidate>());
  NVF_ERROR(ldst_mbarrier_token_map_.count(expr) == 0);
  ldst_mbarrier_token_map_.emplace(expr, mbarrier_tokens);
}

bool TmaCircularBufferInfo::existsMBarrierToken(const Expr* expr) const {
  return ldst_mbarrier_token_map_.count(expr) != 0;
}

TensorView* TmaCircularBufferInfo::getMBarrierToken(const Expr* expr) {
  NVF_ERROR(
      ir_utils::isCpAsyncBulkLoad(expr) || expr->isA<kir::MBarrierInit>() ||
      expr->isA<kir::MBarrierInvalidate>());
  // short-circuit: expr does not have mbarrier token.
  if (ldst_mbarrier_token_map_.count(expr) == 0) {
    return nullptr;
  }
  return ldst_mbarrier_token_map_.at(expr);
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
  InsertionInfo insertion_info = CircularBufferLoopNestInspector::run(exprs);
  return CircularBufferInserter::run(exprs, insertion_info);
}

} // namespace nvfuser
