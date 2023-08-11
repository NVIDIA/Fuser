// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/remove_empty.h>

#include <ir/utils.h>
#include <iter_visitor.h>
#include <ops/alias.h>
#include <ops/arith.h>

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <vector>

namespace nvfuser::optimization {

namespace {

//! Get a vector of the integer positions of constant zero extent axes in the
//! input domain. This will typically be used like
//! `emptyAxes(TensorDomain::noReductions(tv->getMaybeRFactorDomain()))`
std::vector<int64_t> emptyAxes(const std::vector<IterDomain*>& domain) {
  std::vector<int64_t> empty_axes;
  for (auto ax : c10::irange(domain.size())) {
    auto id = domain.at(ax);
    if (id->getMaybeExpandedExtent()->isConst() &&
        id->getMaybeExpandedExtent()->evaluateInt() == 0) {
      empty_axes.push_back((int64_t)ax);
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

//! EmptyTensorRemover performs a backward traversal of the Fusion. When it
//! detects a TensorView that has at least one extent that is zero, we do the
//! following:
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
//!   5. If empty TensorViews are input to an MmaOp and they are empty in
//!   contracted axes, we replace with `full({m, n}, zeroVal())`.
//!
class EmptyTensorRemover : public DeadCodeRemover {
 public:
  EmptyTensorRemover(Fusion* fusion) : DeadCodeRemover(fusion) {}

 protected:
  using DeadCodeRemover::handle;

  //! If tv is a fusion output, we check whether it is empty and if so, replace
  //! it with full(). For non-outputs that are not inputs, we simply check that
  //! the tensor is not provably empty.
  void handle(TensorView* tv) final {
    DeadCodeRemover::handle(tv);
    if (isDead(tv)) {
      // DeadCodeRemover::handle might have set this dead, in which case we
      // don't need to process it any further
      return;
    }

    if (isTVEmpty(tv)) {
      if (tv->isFusionInput()) {
        TORCH_INTERNAL_ASSERT(
            allUsesDead(tv),
            "Empty Fusion input ",
            tv,
            " should not have any live uses.");
        // Empty inputs do not have a definition to redefine
        return;
      }

      // Any non-input that we traverse to should be the input to an expression,
      // or a Fusion output. If it's the input to an expression, we should have
      // replaced that expression by handling the appropriate Expr subclass.
      TORCH_INTERNAL_ASSERT(
          tv->isFusionOutput(),
          "Found unexpected empty intermediate TensorView ",
          tv->toString());
      auto shape = noReductionShape(tv);
      auto dtype = tv->getDataType().value();
      auto new_tv = zeros(shape, dtype);
      registerReplacement(tv, new_tv);
    }
  }

  //! Gets a vector of extents for noReduction(tv->getMaybeRFactorDomain())
  static std::vector<Val*> noReductionShape(TensorView* tv) {
    std::vector<Val*> shape;
    for (auto id : TensorDomain::noReductions(tv->getMaybeRFactorDomain())) {
      shape.push_back(id->getMaybeExpandedExtent());
    }
    return shape;
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
  void handle(ReductionOp* rop) final {
    auto in = rop->in()->as<TensorView>();
    auto empty_input_axes =
        emptyAxes(TensorDomain::noReductions(in->getMaybeRFactorDomain()));
    if (empty_input_axes.empty()) {
      // Input is not empty, handle like any other op
      return;
    }
    auto out = rop->out()->as<TensorView>();
    // The input is empty in some axes. Assert that they are all reduced
    for (auto ax : empty_input_axes) {
      auto id = out->getRootDomain().at(ax);
      // Input rfactor domain positions correspond to output root positions
      TORCH_INTERNAL_ASSERT(
          id->isReduction(),
          "Found unexpected unreduced empty axis at position ",
          ax,
          " in expression ",
          rop->toString());
    }

    auto new_tv =
        full(noReductionShape(out), rop->init(), out->getDataType().value());
    registerReplacement(out, new_tv);
  }

  //! A WelfordOp is similar to a ReductionOp, but has three outputs: avg, var,
  //! N. For an empty reduction N will be zero, so we fill the output with zero.
  //! The avg and var is obtained by summing then dividing by N. For empty
  //! reductions this leads to 0.0 / 0 so we fill it with a constant NAN. The
  //! .var variable is actually an unnormalized variance which is a sum without
  //! dividing by N or N-1, so we fill it with zeros.
  void handle(WelfordOp* wop) final {
    auto in = wop->in()->as<TensorView>();
    auto empty_input_axes =
        emptyAxes(TensorDomain::noReductions(in->getMaybeRFactorDomain()));
    if (empty_input_axes.empty()) {
      // Input is not empty, handle like any other op
      return;
    }
    auto avg = wop->outAvg()->as<TensorView>();
    auto var_sum = wop->outVar()->as<TensorView>();
    auto N = wop->outN()->as<TensorView>();
    // The input is empty in some axes. Assert that they are all reduced
    for (auto ax : empty_input_axes) {
      auto id = avg->getRootDomain().at(ax);
      // Input rfactor domain positions correspond to output root positions
      TORCH_INTERNAL_ASSERT(
          id->isReduction(),
          "Found unexpected unreduced empty axis at position ",
          ax,
          " in expression ",
          wop->toString());
    }

    // Since WelfordOp has multiple outputs, we need to check whether each is
    // live before replacing it, to avoid replacing a dead output with a live
    // one.
    auto shape = noReductionShape(avg);
    if (isLive(avg)) {
      auto nan = IrBuilder::create<Val>(
          std::numeric_limits<double>::quiet_NaN(), avg->getDataType().value());
      auto nan_tensor = full(shape, nan, avg->getDataType().value());
      registerReplacement(avg, nan_tensor);
    }
    if (isLive(var_sum)) {
      auto new_var_sum = full(
          shape,
          fusion()->zeroVal(var_sum->getDataType().value()),
          var_sum->getDataType().value());
      registerReplacement(var_sum, new_var_sum);
    }
    if (isLive(N)) {
      auto new_N = zeros(shape, N->getDataType().value());
      registerReplacement(N, new_N);
    }
  }

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
  void handle(CatOp* cop) final {
    auto dim = cop->concatenatedDim();
    std::vector<TensorView*> non_empty_inputs;
    for (auto inp : cop->inputs()) {
      TORCH_INTERNAL_ASSERT(
          inp->definition() && inp->definition()->isA<PadOp>(),
          "Inputs to CatOp must be outputs of PadOps");
      auto tv = inp->definition()->as<PadOp>()->in()->as<TensorView>();
      auto cat_id =
          TensorDomain::noReductions(tv->getMaybeRFactorDomain()).at(dim);
      if (cat_id->getMaybeExpandedExtent()->isConst() &&
          cat_id->getMaybeExpandedExtent()->evaluateInt() == 0) {
        continue;
      }
      non_empty_inputs.push_back(tv);
    }
    if (non_empty_inputs.size() != cop->inputs().size()) {
      // Replace this op with a new cat op
      auto old_tv = cop->outputs()[0]->as<TensorView>();
      // NOTE: cat() will translate to set() if non_empty_inputs.size() == 1.
      // Also note that unless we're careful this call to cat() might result in
      // symbolic axis, since the inputs may have unknown extents in the cat
      // dimension. By default, cat() will make the conservative choice in such
      // a situation and set the output IterType to Symbolic. However, since we
      // have already undergone concretization at this point, we can trust that
      // the original IterType is correct, so we pass it here to avoid creating
      // new Symbolic axes.
      auto iter_type = old_tv->getMaybeRFactorDomain()
                           .at(cop->concatenatedDim())
                           ->getIterType();
      auto new_tv = cat(non_empty_inputs, dim, iter_type);
      registerReplacement(old_tv, new_tv);
    }
  }

  //! Replace pad(tv) if tv is empty in any dimension. Note that since we detect
  //! empty tensors by looking for constant extents, the output extents will be
  //! correct here already, so there is no value in removing the empty input
  //! extent when we do the replacement.
  void handle(PadOp* pop) final {
    auto in = pop->in()->as<TensorView>();
    auto in_rfactor = TensorDomain::noReductions(in->getMaybeRFactorDomain());
    if (!emptyAxes(in_rfactor).empty()) {
      auto out = pop->out()->as<TensorView>();
      auto shape = noReductionShape(out);
      auto dtype = out->getDataType().value();
      auto new_tv = full(shape, pop->value(), dtype);
      registerReplacement(out, new_tv);
    }
  }

  //! We handle MmaOp just as if it were written as a sum ReductionOp.
  void handle(MmaOp* mop) final {
    auto A = mop->inA()->as<TensorView>();
    auto A_rfactor = TensorDomain::noReductions(A->getMaybeRFactorDomain());
    // We only need to check empty axes in A. If any reduced axes are empty
    // here, they will be empty in B also. If any non-reduced axes are empty,
    // the output will also be empty, and this expression will already be dead.
    if (!emptyAxes(A_rfactor).empty()) {
      auto out = mop->out()->as<TensorView>();
      auto shape = noReductionShape(out);
      auto dtype = out->getDataType().value();
      auto new_tv = zeros(shape, dtype);
      registerReplacement(out, new_tv);
    }
  }
};

} // namespace

void RemoveEmptyPass::runPass(Fusion* fusion) {
  EmptyTensorRemover(fusion).run();
}

} // namespace nvfuser::optimization
