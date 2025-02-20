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
    if (!expr->isA<ForLoop>()) {
      std::cout << "skipper " << expr->toString() << std::endl;
    }
    if (expr->isOneOf<BroadcastOp, ExpandOp, LoadStoreOp, SqueezeOp>()) {
      NVF_ERROR(expr->inputs().size() == 1);
      NVF_ERROR(expr->outputs().size() == 1);
      auto* in = dynamic_cast<TensorView*>(expr->input(0));
      auto* out = dynamic_cast<TensorView*>(expr->output(0));
      if (in && out && in->getMemoryType() == MemoryType::Shared &&
          out->getMemoryType() == MemoryType::Shared) {
        std::cout << "Found shared to shared op " << std::endl;
        // TODO: also check that no data movement occurs here
        // I think we only need to check that the allocation domains (ignoring
        // reduction and broadcast IDs) are all exact mapped
        GpuLower::current()->aliasTensorProducer(out, in);
        // registerRemove(expr);
        return;
      } else {
        std::cout << "This is not a Shared to shared op" << std::endl;
      }
    }
    /*
    std::vector<Val*> replaced_inputs;
    replaced_inputs.reserve(expr->inputs().size());
    bool has_replacement = false;
    for (Val* inp : expr->inputs()) {
      TensorView* replacement_input =
          GpuLower::current()->getTensorProducerAlias(
              dynamic_cast<TensorView*>(inp));
      if (replacement_input == nullptr) {
        std::cout << "  Input " << inp->toString() << " " << (void*)inp << " no
    replacement found" << std::endl; replaced_inputs.push_back(inp); } else {
        std::cout << "  Input " << inp->toString() << " " << (void*)inp << "
    replaced by " << replacement_input->toString() << std::endl;
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
    */
    kir::ExprMutator::dispatch(expr);
  }
};

class AliasedTensorRemover : private kir::ExprMutator {
 public:
  static std::vector<Expr*> run(const std::vector<Expr*>& exprs) {
    AliasedTensorRemover remover(exprs);
    return remover.exprs_;
  }

 private:
  using kir::ExprMutator::dispatch;
  using kir::ExprMutator::handle;

  AliasedTensorRemover(const std::vector<Expr*>& exprs) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  void registerReplaceAndPropagate(Expr* old_expr, Expr* new_expr) {
    registerReplace(old_expr, new_expr);
    GpuLower::current()->propagateExprInfo(old_expr, new_expr);
  }

  void dispatch(Expr* expr) final {
    if (!expr->isA<ForLoop>()) {
      std::cout << "remover " << expr->toString() << std::endl;
    }
    for (Val* inp : expr->inputs()) {
      // Record input TensorIndex so we can look it up later
      auto* input_idx = dynamic_cast<kir::TensorIndex*>(inp);
      if (input_idx != nullptr) {
        producer_indices_[input_idx->view()] = input_idx;
      }
    }
    if (expr->outputs().size() == 1 &&
        GpuLower::current()->getTensorProducerAlias(
            ir_utils::getTv(expr->output(0)))) {
      // Remove definitions and initializations of tensors aliased to their
      // producer
      registerRemove(expr);
      return;
    }
    std::vector<Val*> replaced_inputs;
    replaced_inputs.reserve(expr->inputs().size());
    bool has_replacement = false;
    for (Val* inp : expr->inputs()) {
      // Inputs will be scalars or TensorIndex
      // If we encounter an aliased TensorIndex, we need to look up a suitable
      // TensorIndex to use in its place
      auto* inp_ti = dynamic_cast<kir::TensorIndex*>(inp);
      if (inp_ti != nullptr) {
        TensorView* replacement_input =
            GpuLower::current()->getTensorProducerAlias(inp_ti->view());
        if (replacement_input == nullptr) {
          std::cout << "  Input " << inp->toString() << " " << (void*)inp
                    << " no replacement found" << std::endl;
          replaced_inputs.push_back(inp);
        } else {
          std::cout << "  Input " << inp->toString() << " " << (void*)inp
                    << " replaced by " << replacement_input->toString()
                    << std::endl;
          // We cannot reuse the index for inp and just swap with
          // replacement_input as the view. This is because the producer might
          // have been circular buffered, whereas the consumer is not. So we
          // must take care to use the same TensorIndex that would've been used
          // in the skipped expression
          replaced_inputs.push_back(producer_indices_.at(replacement_input));
          has_replacement = true;
        }
      } else {
        replaced_inputs.push_back(inp);
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

  // Remove Allocate nodes corresponding to tensors that are aliased to their
  // producers
  void handle(kir::Allocate* alloc) final {
    if (GpuLower::current()->getTensorProducerAlias(
            ir_utils::getTv(alloc->buffer())) != nullptr) {
      registerRemove(alloc);
    }
  }

 private:
  std::unordered_map<TensorView*, kir::TensorIndex*> producer_indices_;
};

} // namespace

// Transpose, Shift, Gather, and View Ops with LoadStoreOps.
std::vector<Expr*> loadStoreOpInserter(const std::vector<Expr*>& exprs) {
  return LoadStoreOpInserter::insert(exprs);
}

std::vector<Expr*> skipToAliasedConsumers(const std::vector<Expr*>& exprs) {
  return AliasedTensorSkipper::run(exprs);
}

std::vector<Expr*> removeAliasedConsumers(const std::vector<Expr*>& exprs) {
  return AliasedTensorRemover::run(exprs);
}

} // namespace nvfuser
