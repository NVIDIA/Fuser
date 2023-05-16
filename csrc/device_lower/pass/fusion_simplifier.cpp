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

// Replaces Transpose, Shift, Gather, and View Ops with Unary Ops.
class UnaryOpInserter : private kir::ExprMutator {
 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    UnaryOpInserter inserter(exprs);
    return inserter.exprs_;
  }

 private:
  using kir::ExprMutator::handle;

  UnaryOpInserter(const std::vector<Expr*>& exprs) {
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
        IrBuilder::create<LoadStoreOp>(
            container, LoadStoreOpType::Set, out, in));
  }

  void handle(ExpandOp* eop) final {
    auto out = eop->out();
    auto in = eop->in();
    auto container = out->container();
    registerReplaceAndPropagate(
        eop,
        IrBuilder::create<LoadStoreOp>(
            container, LoadStoreOpType::Set, out, in));
  }

  void handle(ShiftOp* sop) final {
    auto out = sop->out();
    auto in = sop->in();
    auto container = out->container();
    registerReplaceAndPropagate(
        sop,
        IrBuilder::create<LoadStoreOp>(
            container, LoadStoreOpType::Set, out, in));
  }

  void handle(GatherOp* gop) final {
    auto out = gop->out();
    auto in = gop->in();
    auto container = out->container();
    registerReplaceAndPropagate(
        gop,
        IrBuilder::create<LoadStoreOp>(
            container, LoadStoreOpType::Set, out, in));
  }

  void handle(ViewOp* vop) final {
    auto out = vop->out();
    auto in = vop->in();
    auto container = out->container();
    registerReplaceAndPropagate(
        vop,
        IrBuilder::create<LoadStoreOp>(
            container, LoadStoreOpType::Set, out, in));
  }
};

} // namespace

// Transpose, Shift, Gather, and View Ops with Unary Set Ops
std::vector<Expr*> unarySetOpInserter(const std::vector<Expr*>& exprs) {
  return UnaryOpInserter::insert(exprs);
}

} // namespace nvfuser
