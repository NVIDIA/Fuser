// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <device_lower/pass/insert_syncs.h>
#include <device_lower/utils.h>
#include <dispatch.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <ops/arith.h>

#include <unordered_set>

namespace nvfuser {

namespace {

//! Insert needed syncs in order to serialize blocks for serial grid reduction.
//!
//! We inspect the loop nest up to the point after which all outer loops are
//! trivial. This corresponds to the outer-most loop in the generated
//! kernel code.
//!
//! Conditions for using serial grid reduction:
//!  - All reduction dimensions are either unparallelized or parallelized as
//!    BID, not TID. Block and warp reductions could be allowed in the future,
//!    but the current focus is on cases where all threads are doing separate
//!    reductions simultaneously.
//!  - rop is not an allreduce. Note that we could implement serial allreduce
//!    but it would require inserting a separate grid sync after this outer
//!    loop.
//!  - There are no other reductions in this loop nest that are TID or BID
//!    parallelized, unless they also satisfy the conditions above and their
//!    reduction pattern matches this one. Otherwise our syncs will be
//!    mismatched, and there is no good way to handle that yet.
class GridSerializationSyncInserter : kir::ExprMutator {
 public:
  GridSerializationSyncInserter(const std::vector<Expr*>& exprs) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    GridSerializationSyncInserter inserter(exprs);
    return inserter.exprs_;
  }

 private:
  using kir::ExprMutator::dispatch;
  using kir::ExprMutator::handle;

  //! If this is a serial grid reduction, replace it with a sequence of IR
  //! nodes to effect the reduction step.
  void handle(ReductionOp* rop) override {
    if (rop->serialGridReductionRequested()) {
      recordSyncPattern(rop);

      replaceReductionOp(rop);
    }
  }

  //! Record the grid sync pattern so that we can check compatibility in case
  //! multiple serial grid reductions occur within the same non-trivial loop.
  void recordSyncPattern(ReductionOp* rop) {
    ParallelTypeBitmap sync_pattern;
    auto out = rop->out()->as<TensorView>();
    NVF_ERROR(out != nullptr);
    for (int i : c10::irange((int)out->nDims())) {
      IterDomain* ax = out->axis(i);
      if (!ax->isReduction()) {
        continue;
      }
      NVF_ERROR(
          !ax->isThreadDim(),
          "Serial grid reduction cannot be applied with block reductions: ",
          rop->toString());
      if (ax->isBlockDim()) {
        sync_pattern.set(ax->getParallelType());
      }
    }

    if (!sync_pattern.hasBID()) {
      // Don't set cur_expr_sync_pattern_ since this is not actually a grid
      // reduction
      return;
    }

    if (cur_expr_sync_pattern_.has_value()) {
      NVF_ERROR(
          cur_expr_sync_pattern_.value() == sync_pattern,
          "Reduction op ",
          rop->toString(),
          " has requested serial grid reduction, but pattern ",
          sync_pattern.toString(),
          " conflicts with previous pattern: ",
          cur_expr_sync_pattern_.value().toString());
    } else {
      cur_expr_sync_pattern_ = sync_pattern;
    }
  }

  //! Replace a serial reduction op with an IR representation of the update
  //! step. This mimics the helper function
  //!
  //!   template <typename T, typename Func>
  //!   __device__ void serialReductionStep(
  //!       T& out,
  //!       T in,
  //!       T init,
  //!       volatile T& work,
  //!       Func reduction_op,
  //!       bool first_step,
  //!       bool last_step,
  //!       bool read_pred,
  //!       bool write_pred) {
  //!     if (!write_pred) {
  //!       return;
  //!     }
  //!     out = read_pred ? in : init;
  //!     if (!first_step) {
  //!       reduction_op(out, work);
  //!     }
  //!     if (!last_step) {
  //!       work = out;
  //!     }
  //!   }
  void replaceReductionOp(ReductionOp* rop) {
    NVF_ERROR(rop->serialGridReductionRequested());
    NVF_ERROR(cur_expr_sync_pattern_.has_value());

    Val* is_first_step = nullptr;
    Val* is_last_step = nullptr;
    for (auto pt : kParallelTypeBIDs) {
      if (cur_expr_sync_pattern_.value().get(pt)) {
        // && BID == 0
        is_first_step = SimplifyingIrBuilder::logicalAndExpr(
            is_first_step,
            IrBuilder::eqExpr(
                NamedScalar::getParallelIndex(pt),
                FusionGuard::getCurFusion()->zeroVal(DataType::Index)));
        // && BID == BDIM - 1
        is_last_step = SimplifyingIrBuilder::logicalAndExpr(
            is_last_step,
            IrBuilder::eqExpr(
                NamedScalar::getParallelIndex(pt),
                IrBuilder::subExpr(
                    NamedScalar::getParallelDim(pt),
                    FusionGuard::getCurFusion()->oneVal(DataType::Index))));
      }
    }
    is_first_step = GpuLower::current()->commonScalarMap().hoistScalar(
        is_first_step, for_loops_);
    is_last_step = GpuLower::current()->commonScalarMap().hoistScalar(
        is_last_step, for_loops_);
    NVF_ERROR(is_first_step != nullptr);
    NVF_ERROR(is_last_step != nullptr);

    auto in = rop->in()->as<TensorView>();
    auto out = rop->out()->as<TensorView>();

    // Create a work buffer
    TensorView* work_buffer_in = IrBuilder::create<TensorView>(
        out->domain(), out->dtype(), MemoryType::Global);

    TensorView* loaded_work_buffer =
        work_buffer_in->cacheAfter(LoadStoreOpType::Set, CacheOp::Global);
    // INTERNAL ASSERT FAILED at
    // "/opt/pytorch/nvfuser/csrc/tensor_view.cpp":1278 Function invalid for
    // kernel container.
    // TODO: caching is not meant to work on kernel IR, so we'll have to work
    // around that

    TensorView* in_plus_work =
        binaryOp(rop->getReductionOpType(), in, loaded_work_buffer);
    TensorView* result = where(is_first_step, in, in_plus_work);

    // work_buffer_out is dead code (it has no uses). This means we won't
    // automatically insert new Allocate nodes for it in subsequent lowering
    // passes, so it is safe for us to create one here.
    TensorView* work_buffer_out =
        result->cacheAfter(LoadStoreOpType::Set, CacheOp::Global);
    work_buffer_out->setMemoryType(MemoryType::Global);
    // TODO: we cannot force aliasing work_buffer_out => work_buffer_in yet,
    // since that is done with Allocate. Since we _do_ want automatic insertion
    // of work_buffer_in's Allocate, it is unclear how we can effectively set
    // this alias.

    // Insert expressions
    // use with{Write}Predicate for global load and store expressions
    registerInsertBefore(
        rop,
        loaded_work_buffer->definition()->withPredicate(
            IrBuilder::create<kir::Predicate>(
                IrBuilder::logicalNotExpr(is_first_step))));
    registerInsertBefore(rop, in_plus_work->definition());
    registerInsertBefore(
        rop,
        work_buffer_out->definition()->withWritePredicate(
            IrBuilder::create<kir::Predicate>(
                IrBuilder::logicalNotExpr(is_last_step))));
    registerInsertBefore(rop, result->definition());
  }

  void dispatch(Expr* expr) override {
    // We will detect top-level exprs here that require serialization and
    // insert the required syncs before and after those exprs.
    if (auto loop = dynamic_cast<kir::ForLoop*>(expr);
        cur_top_level_expr_ != nullptr || (loop && loop->isTrivial())) {
      // Never sync around trivial loops since they do not appear in the
      // generated CUDA code. Also avoid redefining cur_top_level_expr_ if it
      // is already set, which indicates that this expression is contained in
      // an outer non-trivial loop.
      kir::ExprMutator::dispatch(expr);
      return;
    }
    // Any other expr, i.e. non-trivial loops or regular Exprs, can be synced if
    // it is top-level and either is or contains a serial grid reduction
    cur_top_level_expr_ = expr;
    // If a serial grid reduction was found when traversing expr, then
    // cur_expr_sync_pattern_ will be set
    cur_expr_sync_pattern_ = std::nullopt;
    kir::ExprMutator::dispatch(expr);
    if (cur_expr_sync_pattern_.has_value()) {
      insertSyncs();
    }
    // reset state variables
    cur_top_level_expr_ = nullptr;
    cur_expr_sync_pattern_ = std::nullopt;
  }

  void insertSyncs() {
    NVF_ERROR(cur_top_level_expr_ != nullptr);
    NVF_ERROR(cur_expr_sync_pattern_.has_value());
    kir::Allocate* alloc = lower_utils::allocGlobalBufferForGridComm(
        lower_utils::getGridSyncBufferSize(cur_expr_sync_pattern_.value()),
        DataType::Int,
        true);
    auto wait = IrBuilder::create<kir::BlockSerializeWait>(
        cur_expr_sync_pattern_.value(), alloc->buffer());
    registerInsertBefore(cur_top_level_expr_, alloc);
    registerInsertBefore(cur_top_level_expr_, wait);
    auto release = IrBuilder::create<kir::BlockSerializeRelease>(
        cur_expr_sync_pattern_.value(), alloc->buffer());
    registerInsertAfter(cur_top_level_expr_, release);
  }

 private:
  //! Which Expr* is the current top-level containing the current Expr in the
  //! generated kernel. When serial reductions are encountered, this expression
  //! determines where we will place syncs: they will be placed before and after
  //! this expression.
  //!
  //! For example, if we have
  //!
  //!   FOR iBlockIdx.x
  //!     FOR iS{32}
  //!       y = neg(x);
  //!     ENDFOR iS{32}
  //!     FOR iThreadIdx.x
  //!       z = add(y, x);
  //!     ENDFOR iThreadIdx.x
  //!   ENDFOR iBlockIdx.x
  //!
  //! then when we are processing the `neg` Expr, cur_top_level_expr_ will be
  //! the FOR iS{32} loop. However, when processing the `add` expression,
  //! cur_top_level_expr_ will be nullptr since that expression itself will
  //! appear in the main scope of the generated kernel.
  //!
  //! IfThenElse are treated similar to unparallelized ForLoops; if an
  //! IfThenElse is at the top level, or is contained in a fully parallelized
  //! loop nest, it is treated as a top level expr here. Note that this pass
  //! will likely be run before any IfThenElse are placed in the kernel anyway.
  Expr* cur_top_level_expr_ = nullptr;

  //! If a serial grid reduction is found for the current expr, this indicates
  //! parallel axes that are mapped to reduction domains in the serial
  //! reduction.
  std::optional<ParallelTypeBitmap> cur_expr_sync_pattern_ = std::nullopt;
};

} // namespace

std::vector<Expr*> translateSerialGridReduction(
    const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::translateSerialGridReduction");
  return GridSerializationSyncInserter::insert(exprs);
}

} // namespace nvfuser
