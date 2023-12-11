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

#include <unordered_set>

namespace nvfuser {

namespace {

//! Insert needed syncs in
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

  //! Record cur_expr_sync_pattern_ if this is a serial grid reduction
  void handle(ReductionOp* rop) override {
    if (rop->serialGridReductionRequested()) {
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
  }

  void dispatch(Expr* expr) override {
    if (auto loop = dynamic_cast<kir::ForLoop*>(expr);
        cur_top_level_expr_ || (loop && loop->isTrivial())) {
      // Never sync around trivial loops. Also avoid redefining
      // cur_top_level_expr_ if it is already set
      kir::ExprMutator::dispatch(expr);
      return;
    }
    // Any other expr, i.e. non-trivial loops or regular Exprs, can be synced if
    // it is top-level and either is or contains a serial grid reduction
    cur_top_level_expr_ = expr;
    kir::ExprMutator::dispatch(expr);
    // If a serial grid reduction was found in expr, then cur_expr_sync_pattern_
    // will be set
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

std::vector<Expr*> insertGridSerializationSyncs(
    const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertGridSerializationSyncs");
  return GridSerializationSyncInserter::insert(exprs);
}

} // namespace nvfuser
