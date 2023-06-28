// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/remove_empty.h>

#include <ir/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>

#include <algorithm>
#include <deque>

namespace nvfuser::optimization {

namespace {

//! Get a vector of the integer positions of constant zero extent axes in the
//! input domain. This will typically be used like
//! `emptyAxes(TensorDomain::noReductions(tv->getMaybeRFactorDomain()))`
std::vector<size_t> emptyAxes(std::vector<IterDomain*> domain) {
  std::vector<size_t> empty_axes;
  for (auto ax : c10::irange(domain.size())) {
    auto id = domain.at(ax);
    if (id->extent()->isConst() && id->extent()->evaluateInt() == 0) {
      empty_axes.push_back(ax);
    }
  }
  return empty_axes;
}

//! Check whether a TensorView is empty. During concretization, we traverse to
//! find a minimal set of TensorViews that have zero extents, and we then set
//! their extents to a constant 0. Here we check for those constant zero
//! extents.
bool isTVEmpty(TensorView* tv) {
  return !emptyAxes(TensorDomain::noReductions(tv->getMaybeRFactorDomain()))
              .empty();
}

//! removeEmptyPass performs a backward traversal of the Fusion. When it detects
//! a TensorView that has at least one extent that is zero, we do the following:
//!
//!   1. If the empty Tensorview is a Fusion output, we replace it with a
//!   TensorView created by `full` having the same shape. Since the original
//!   tensor is empty, there is nothing to compute, so this eliminates a branch
//!   of trivial code.
//!   2. If the empty TensorView is the input of a `cat` op along the empty
//!   dimension, we replace the cat op with a new one having the empty input
//!   removed.
//!   3. If the empty Tensorview is the input to a ReductionOp or WelfordOp and
//!   the empty dimensions are reduced, we replace the op with `full` since
//!   there are no elements being reduced. Note that if any empty axes are not
//!   reduced, we will not encounter this case since it will have been removed
//!   earlier in the backward traversal under condition 1.
//!   4. If the empty Tensorview is the input to a PadOp (which is not input to
//!   a CatOp) then we replace the pad with `full(pad_value)`.
//!
//! Note that we do not use BackwardVisitor here even though we are doing a
//! backward traversal. This is because we will actually be changing the Fusion
//! graph as we traverse. BackwardVisitor works by creating a queue of
//! statements using InputsOf and a forward traversal, then iteratively pops
//! that static queue of statements. This does not work well when we want to
//! eliminate dead code while traversing, so instead we implement a simple
//! queue-based breadth-first backward traversal manually here.
class EmptyTensorRemover {
 public:
  EmptyTensorRemover(Fusion* fusion) : fusion_(fusion) {
    for (auto outp : fusion->outputs()) {
      stmt_queue_.push_back(outp);
    }
  }

  void run() {
    while (!stmt_queue_.empty()) {
      auto stmt = stmt_queue_.front();
      stmt_queue_.pop_front();
      handle(stmt);
    }
  }

  void handle(Statement* stmt) {
    if (stmt->isVal()) {
      handle(stmt->asVal());
    } else {
      TORCH_INTERNAL_ASSERT(
          stmt->isExpr(),
          "Statement is neither a Val or Expr: ",
          stmt->toString());
      handle(stmt->asExpr());
    }
  }

  void handle(Val* v) {
    if (auto tv = dynamic_cast<TensorView*>(v)) {
      handle(tv);
    } else if (v->definition()) {
      // TensorView vals might be overwritten, in which case we should not keep
      // traversing. For all other Vals, push their definition to the stack.
      stmt_queue_.push_back(v->definition());
    }
  }

  //! If tv is a fusion output, we check whether it is empty and if so, replace
  //! it with full(). For non-outputs that are not inputs, we simply check that
  //! the tensor is not provably empty.
  void handle(TensorView* tv) {
    if (tv->isFusionOutput()) {
      const auto rfactor =
          TensorDomain::noReductions(tv->getMaybeRFactorDomain());
      const auto empty_axes = emptyAxes(rfactor);
      if (!empty_axes.empty()) {
        std::vector<Val*> shape(rfactor.size());
        std::transform(
            rfactor.begin(), rfactor.end(), shape.begin(), [](IterDomain* id) {
              return id->extent();
            });
        for (auto ax : empty_axes) {
          shape[ax] = fusion_->zeroVal();
        }
        auto dtype = tv->getDataType().value();
        auto new_tv = full(shape, fusion_->zeroVal(dtype), dtype);
        replaceTV(tv, new_tv);
        // Do not keep traversing upstream if we've replaced tv
        return;
      }
    } else if (!tv->isFusionInput()) {
      // TODO: This should be a warning instead
      TORCH_INTERNAL_ASSERT(
          !isTVEmpty(tv),
          "Found unexpected empty intermediate TensorView ",
          tv->toString());
    }
    if (tv->definition()) {
      stmt_queue_.push_back(tv->definition());
    }
  }

  //! Push the inputs of an expression onto the statement stack for further
  //! processing.
  void pushInputs(Expr* e) {
    for (auto inp : e->inputs()) {
      stmt_queue_.push_back(inp);
    }
  }

  //! A reduction over empty axes is equal to the initial value of the
  //! reduction, as if the reduction were written as follows:
  //!
  //!   auto result = init_value;
  //!   for (auto element : reduction_elements) {
  //!     result = reduction_op(result, element);
  //!   }
  //!   return result;
  //!
  void handle(ReductionOp* rop) {
    auto in = rop->in()->as<TensorView>();
    auto empty_input_axes =
        emptyAxes(TensorDomain::noReductions(in->getMaybeRFactorDomain()));
    if (empty_input_axes.empty()) {
      // Input is not empty, handle like any other op
      pushInputs(rop);
      return;
    }
    auto out = rop->out()->as<TensorView>();
    // The input is empty in some axes. Assert that they are all reduced
    const auto& out_root = out->getRootDomain();

    std::vector<Val*> shape;
    for (auto id : out_root) {
      if (!id->isReduction() && !id->isStride()) { // same as noReductions()
        shape.push_back(id->extent());
      }
    }

    for (auto ax : empty_input_axes) {
      auto id = out_root.at(ax);
      // Input rfactor domain positions correspond to output root positions
      TORCH_INTERNAL_ASSERT(
          id->isReduction(),
          "Found unexpected unreduced empty axis at position ",
          ax,
          " in expression ",
          rop->toString());
      shape[ax] = fusion_->zeroVal();
    }
    // Find output shape to replace with full

    auto new_tv = full(shape, rop->init(), out->getDataType().value());
    replaceTV(out, new_tv);
  }

  //! A reduction over empty axes is equal to the initial value of the
  //! reduction, as if the reduction were written as follows:
  //!
  //!   auto result = init_value;
  //!   for (auto element : reduction_elements) {
  //!     result = reduction_op(result, element);
  //!   }
  //!   return result;
  //!
  void handle(WelfordOp* wop) {}

  //! A cat op can have input empty tensors and still output a non-empty
  //! tensor. This is only possible if there is more than one input, so we
  //! only need to handle those cases. We find the non-empty inputs to cat
  //! then replace with another cat (or `set` if n=1).
  //!
  //! The `cat` function creates a CatOp object, but its inputs() are not
  //! the original inputs. Rather, they are the inputs after padding to the
  //! output extent in the concatenated dimension. Thus, in the IR graph,
  //! instead of the following:
  //!
  //!    T0  T1   T2
  //!      \  |  /
  //!       CatOp
  //!         |
  //!        T3
  //!
  //! a cat is represented as:
  //!
  //!    T0    T1    T2
  //!     |     |     |
  //!   PadOp PadOp PadOp
  //!       \   |   /
  //!         CatOp
  //!           |
  //!          T3
  //!
  //! If we determine that one of the inputs, T1, is empty in the cat
  //! dimension, then we rewrite this as:
  //!
  //!    T0          T2
  //!     |           |
  //!   PadOp       PadOp
  //!       \       /
  //!         CatOp
  //!           |
  //!          T3
  //!
  //! This is done by simply calling the cat() command with only {T0, T2}.
  void handle(CatOp* cop) {
    auto dim = cop->concatenatedDim();
    std::vector<TensorView*> non_empty_inputs;
    for (auto inp : cop->inputs()) {
      TORCH_INTERNAL_ASSERT(
          inp->definition() && inp->definition()->isA<PadOp>(),
          "Inputs to CatOp must be outputs of PadOps");
      auto tv = inp->definition()->as<PadOp>()->in()->as<TensorView>();
      auto cat_id =
          TensorDomain::noReductions(tv->getMaybeRFactorDomain()).at(dim);
      if (cat_id->extent()->isConst() && cat_id->extent()->evaluateInt() == 0) {
        continue;
      }
      non_empty_inputs.push_back(tv);
    }
    if (non_empty_inputs.size() != cop->inputs().size()) {
      // Replace this op with a new cat op
      auto old_tv = cop->outputs()[0]->as<TensorView>();
      // NOTE: cat() will translate to set() if non_empty_inputs.size() == 1
      auto new_tv = cat(non_empty_inputs, dim);
      replaceTV(old_tv, new_tv);
    }
    for (auto tv : non_empty_inputs) {
      // Continue processing non-empty inputs
      stmt_queue_.push_back(tv);
    }
  }

  //! Replace pad(tv) if tv is empty in any dimension. Note that since we detect
  //! empty tensors by looking for constant extents, the output extents will be
  //! correct here already, so there is no value in removing the empty input
  //! extent when we do the replacement.
  void handle(PadOp* pop) {
    auto in = pop->in()->as<TensorView>();
    auto in_rfactor = TensorDomain::noReductions(in->getMaybeRFactorDomain());
    if (!emptyAxes(in_rfactor).empty()) {
      auto out = pop->out()->as<TensorView>();
      auto out_rfactor =
          TensorDomain::noReductions(out->getMaybeRFactorDomain());
      std::vector<Val*> shape;
      shape.reserve(out_rfactor.size());
      for (auto id : out_rfactor) {
        shape.push_back(id->extent());
      }
      auto new_tv = full(shape, pop->value(), out->getDataType().value());
      replaceTV(out, new_tv);
    } else {
      pushInputs(pop);
    }
  }

  //! Replaces a TensorView in outputs, and in all uses. If old_tv is a Fusion
  //! input, we do not replace it. After replacement, unless it is a Fusion
  //! input, we remove it from the fusion and set the original pointer to zero
  //! (hence why old_tv is passed by reference).
  void replaceTV(TensorView*& old_tv, TensorView* new_tv) {
    if (old_tv->isFusionOutput()) {
      fusion_->replaceOutput(old_tv, new_tv);
    }
    for (auto use : old_tv->uses()) {
      ir_utils::replaceValInExpr(use, old_tv, new_tv);
    }
    if (!old_tv->isFusionInput()) {
      fusion_->removeVal(old_tv);
      old_tv = nullptr;
    }
  }

  void handle(Expr* e) {
    if (auto rop = dynamic_cast<ReductionOp*>(e)) {
      handle(rop);
    } else if (auto wop = dynamic_cast<WelfordOp*>(e)) {
      handle(wop);
    } else if (auto pop = dynamic_cast<CatOp*>(e)) {
      handle(pop);
    } else if (auto pop = dynamic_cast<PadOp*>(e)) {
      handle(pop);
    } else {
      // The handled ops above may terminate this branch of the traversal, so
      // they will need to manually handle their inputs. For unhandled ops, we
      // just handle all inputs here.
      pushInputs(e);
    }
  }

 private:
  Fusion* fusion_;
  std::deque<Statement*> stmt_queue_;
};

} // namespace

void RemoveEmptyPass::runPass(Fusion* fusion) {
  EmptyTensorRemover(fusion).run();
}

} // namespace nvfuser::optimization
