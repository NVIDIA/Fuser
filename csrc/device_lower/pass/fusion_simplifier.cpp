// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <ir/builder.h>
#include <kernel_ir_dispatch.h>

#include <device_lower/pass/fusion_simplifier.h>

namespace nvfuser {

namespace {

// Replaces Transpose and View Ops with LoadStoreOps.
class LoadStoreOpInserter : private kir::ExprMutator {
 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    LoadStoreOpInserter inserter(exprs);
    return inserter.exprs_;
  }

 private:
  using kir::ExprMutator::handle;

  LoadStoreOpInserter(const std::vector<Expr*>& exprs) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  void registerReplaceAndPropagate(Expr* old_expr, Expr* new_expr) {
    registerReplace(old_expr, new_expr);
    GpuLower::current()->propagateExprInfo(old_expr, new_expr);
  }

  void handle(SqueezeOp* sop) final {
    auto out = sop->out();
    auto in = sop->in();
    auto container = out->container();
    registerReplaceAndPropagate(
        sop,
        IrBuilder::createInContainer<LoadStoreOp>(
            container, LoadStoreOpType::Set, out, in));
  }

  void handle(ExpandOp* eop) final {
    auto out = eop->out();
    auto in = eop->in();
    auto container = out->container();
    registerReplaceAndPropagate(
        eop,
        IrBuilder::createInContainer<LoadStoreOp>(
            container, LoadStoreOpType::Set, out, in));
  }

  void handle(RepeatOp* op) final {
    auto out = op->out();
    auto in = op->in();
    auto container = out->container();
    registerReplaceAndPropagate(
        op,
        IrBuilder::createInContainer<LoadStoreOp>(
            container, LoadStoreOpType::Set, out, in));
  }

  void handle(ViewOp* vop) final {
    auto out = vop->out();
    auto in = vop->in();
    auto container = out->container();
    registerReplaceAndPropagate(
        vop,
        IrBuilder::createInContainer<LoadStoreOp>(
            container, LoadStoreOpType::Set, out, in));
  }
};

} // namespace

// Transpose, Shift, Gather, and View Ops with LoadStoreOps.
std::vector<Expr*> loadStoreOpInserter(const std::vector<Expr*>& exprs) {
  return LoadStoreOpInserter::insert(exprs);
}

} // namespace nvfuser
