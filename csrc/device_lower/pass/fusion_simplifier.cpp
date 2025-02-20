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

class AliasedTensorSkipper : private kir::ExprMutator {
 public:
  static std::vector<Expr*> run(const std::vector<Expr*>& exprs) {
    AliasedTensorSkipper skipper(exprs);
    return skipper.exprs_;
  }

 private:
  using kir::ExprMutator::dispatch;

  AliasedTensorSkipper(const std::vector<Expr*>& exprs) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  void registerReplaceAndPropagate(Expr* old_expr, Expr* new_expr) {
    registerReplace(old_expr, new_expr);
    GpuLower::current()->propagateExprInfo(old_expr, new_expr);
  }

  void dispatch(Expr* expr) final {
    std::cout << "skipper " << expr->toString() << std::endl;
    if (expr->isOneOf<BroadcastOp, ExpandOp, LoadStoreOp, SqueezeOp>()) {
      NVF_ERROR(expr->inputs().size() == 1);
      NVF_ERROR(expr->outputs().size() == 1);
      auto* in = dynamic_cast<TensorView*>(expr->input(0));
      auto* out = dynamic_cast<TensorView*>(expr->output(0));
      if (in && out && in->getMemoryType() == MemoryType::Shared &&
          out->getMemoryType() == MemoryType::Shared) {
        // TODO: also check that no data movement occurs here
        GpuLower::current()->aliasTensorProducer(out, in);
        return;
      }
    }
    std::vector<Val*> replaced_inputs;
    replaced_inputs.reserve(expr->inputs().size());
    bool has_replacement = false;
    for (Val* inp : expr->inputs()) {
      TensorView* replacement_input =
          GpuLower::current()->getTensorProducerAlias(
              dynamic_cast<TensorView*>(inp));
      if (replacement_input == nullptr) {
        replaced_inputs.push_back(inp);
      } else {
        replaced_inputs.push_back(replacement_input);
        has_replacement = true;
      }
    }
    std::cout << "  has_replacement=" << has_replacement << std::endl;
    if (has_replacement) {
      // An input is aliased to its producer, so use the tensor the alias
      // points to in place of the original.
      registerReplaceAndPropagate(
          expr,
          expr->newObjectFunc()(
              expr->container(),
              replaced_inputs,
              expr->outputs(),
              expr->attributes()));
    }
    kir::ExprMutator::dispatch(expr);
  }
};

} // namespace

// Transpose, Shift, Gather, and View Ops with LoadStoreOps.
std::vector<Expr*> loadStoreOpInserter(const std::vector<Expr*>& exprs) {
  return LoadStoreOpInserter::insert(exprs);
}

std::vector<Expr*> skipToAliasedConsumers(const std::vector<Expr*>& exprs) {
  return AliasedTensorSkipper::run(exprs);
}

} // namespace nvfuser
