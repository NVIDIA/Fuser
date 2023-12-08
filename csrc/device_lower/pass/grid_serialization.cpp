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
  using kir::ExprMutator::handle;

  void handle(ReductionOp* rop) final {
    if (rop->serialGridReductionRequested()) {
    }
  }

  //! Determine whether a ForLoop is a top-level (in generated kernel) loop
  //! before handling.
  void handle(kir::ForLoop* loop) final {
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

  //! If a serial grid reduction is found for the current expr, 
  std::optional<std::tuple<bool, bool, bool>> current_expr_sync_pattern_ =
      std::nullopt;
};

} // namespace

std::vector<Expr*> insertGridSerializationSyncs(
    const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertGridSerializationSyncs");
  return GridSerializationSyncInserter::insert(exprs);
}

} // namespace nvfuser
