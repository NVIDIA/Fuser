// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/utils.h>
#include <dynamic_transform.h>
#include <executor_kernel_arg.h>
#include <executor_utils.h>
#include <expr_evaluator.h>
#include <ir/cloner.h>
#include <ir/utils.h>
#include <ops/arith.h>
#include <ops/utils.h>
#include <transform_iter.h>
#include <transform_view.h>
#include <utils.h>

#include <optional>

namespace nvfuser {

DynamicTransformInitialInfo DynamicTransformInitialInfo::clone(
    IrCloner& ir_cloner) const {
  DynamicTransformInitialInfo cloned_info(
      static_cast<Fusion*>(ir_cloner.container()));
  cloned_info.dynamic_reshapes_.reserve(dynamic_reshapes_.size());
  for (const auto op : dynamic_reshapes_) {
    if (op) {
      cloned_info.dynamic_reshapes_.push_back(ir_cloner.clone(op));
    }
  }
  cloned_info.dynamic_resizes_.reserve(dynamic_resizes_.size());
  for (const auto op : dynamic_resizes_) {
    if (op) {
      cloned_info.dynamic_resizes_.push_back(ir_cloner.clone(op));
    }
  }
  return cloned_info;
}

std::string DynamicTransformInitialInfo::toString() const {
  std::stringstream ss;
  ss << "DynamicTransformInitialInfo\n";
  std::string indent = "  ";
  ss << indent << "Dynamic reshapes:\n";
  for (const auto& op : dynamic_reshapes_) {
    ss << indent << indent << op->toString() << "\n";
  }
  ss << indent << "Dynamic resizes:\n";
  for (const auto& op : dynamic_resizes_) {
    ss << indent << indent << op->toString() << "\n";
  }
  ss << indent << "Root dynamic Vals:\n";
  for (const auto& v : root_dynamic_vals_) {
    ss << indent << indent << v->toString() << "\n";
  }
  return ss.str();
}

//! Gather information about concretizing transformations without
//! concrete input sizes.
class DynamicTransformInitialInfoBuilder : public IterVisitor {
 public:
  DynamicTransformInitialInfoBuilder(Fusion* fusion) : info_(fusion) {
    TORCH_INTERNAL_ASSERT(
        !fusion->isA<kir::Kernel>(),
        "Invalid container. Kernel container not allowed.\n");

    traverseTo(fusion, fusion->getTerminatingOutputs(), false, false);

    finalizeDynamicVals();

    // initial_info_ provides a set of Vals that are used for concretization.
    // Here we check which scalar inputs, if any, correspond to any of those
    // Vals. These will be the inputs that are explicitly used in the cache ID
    // for KernelArgumentHolder.
    auto dyn_vals = info_.getRootDynamicVals();
    for (const auto i : c10::irange(fusion->inputs().size())) {
      auto input = fusion->inputs().at(i);
      if (dyn_vals.find(input) != dyn_vals.end()) {
        info_.scalar_inputs_affecting_concretization_.insert(i);
      }
    }
  }

  const auto& getInfo() const {
    return info_;
  }

 private:
  using IterVisitor::handle;

  //! Find views that have symbolic outputs
  void handle(ViewOp* op) override {
    auto inp_tv = op->in()->as<TensorView>();
    auto out_tv = op->out()->as<TensorView>();
    // If there's no symbolic axis, this is a static reshape op
    if (out_tv->domain()->hasSymbolicAxis()) {
      info_.dynamic_reshapes_.push_back(op);

      // Input and output extent expressions both affect concretization
      for (const auto id :
           TensorDomain::noReductions(inp_tv->getMaybeRFactorDomain())) {
        leaf_dynamic_vals_.push_back(id->extent());
      }
      for (const auto id : out_tv->getMaybeRFactorDomain()) {
        leaf_dynamic_vals_.push_back(id->extent());
      }
    }
  }

  //! Detect dynamic IterDomain transforms when handling TensorViews
  void handle(TensorView* tv) override {
    const auto& rfd = tv->getMaybeRFactorDomain();
    for (auto id : rfd) {
      if (!id->extent()->isConstScalar() || id->extent()->evaluateInt() == 0) {
        info_.dynamic_extent_vals_.insert(id->extent());
        leaf_dynamic_vals_.push_back(id->extent());
      }
      if (!id->definition() || id->getIterType() != IterType::Symbolic) {
        continue;
      }
      if (auto op = dynamic_cast<Resize*>(id->definition())) {
        info_.dynamic_resizes_.push_back(op);
        // extent of output determines its IterType
        leaf_dynamic_vals_.push_back(id->extent());
      }
    }
  }

  //! Process vector of leaf dynamic values by finding inputs and recording the
  //! result into info_
  void finalizeDynamicVals() {
    const auto inputs = InputsOf::outputs(info_.fusion(), leaf_dynamic_vals_);
    info_.root_dynamic_vals_.insert(inputs.begin(), inputs.end());
  }

 private:
  DynamicTransformInitialInfo info_;

  //! This is a collection of scalars that are explicitly checked during
  //! concretization of dynamic ops, meaning they influence the structure of the
  //! resulting concretized Fusion. We track these while traversing the graph
  //! and when we are finished traversing we extract all of the corresponding
  //! non-constant root Vals, which provides us with a minimal list of input
  //! scalars that influence concretization. That list of scalars is then used
  //! to compute a minimal cache key in InputsIdLookup::lookupId().
  std::vector<Val*> leaf_dynamic_vals_;
};

class EmptyBranchFinder : public BackwardVisitor {
 public:
  EmptyBranchFinder(Fusion* fusion, ExpressionEvaluator* expr_eval)
      : expr_eval_(expr_eval) {
    // We do not require the traversal to cover all outputs, because if we
    // replace some outputs with calls to full() then any unused outputs will be
    // ignored entirely.
    must_cover_all_expr_outputs_ = false;
    traverseTo(fusion, fusion->outputs(), false);
  }

  std::vector<EmptyTensorDescriptor> getEmptyTensors() const {
    return empty_tensors_;
  }

 private:
  using BackwardVisitor::handle;

  void handle(TensorView* tv) final {
    std::vector<size_t> empty_axes;
    auto rfactor = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
    bool empty = false;
    for (size_t i : c10::irange(rfactor.size())) {
      auto id = rfactor.at(i);
      auto extent_eval = expr_eval_->evaluate(id->extent());
      TORCH_INTERNAL_ASSERT(
          extent_eval.has_value(),
          "When finding empty tensors: could not evaluate extent of ",
          id->toString());
      if (extent_eval.value().as<int64_t>() == 0) {
        empty_axes.push_back(i);
        empty = true;
      }
    }
    if (empty) {
      if (tv->definition()) {
        // Replace with full. Note that even if the definition was a FullOp, we
        // still mark this tensor for replacement, so that we can ensure the
        // empty axes are marked with constant zeroes
        empty_tensors_.push_back(EmptyTensorDescriptor{tv, empty_axes});
      }
      return;
    } else if (tv->definition()) {
      for (auto v : tv->definition()->inputs()) {
        handle(v);
      }
    }
  }

 private:
  ExpressionEvaluator* expr_eval_;
  std::vector<EmptyTensorDescriptor> empty_tensors_;
};

DynamicTransformConcretizationInfo::DynamicTransformConcretizationInfo(
    Fusion* fusion,
    const DynamicTransformInitialInfo* info,
    ExpressionEvaluator* expr_eval)
    : fusion_(fusion) {
  TORCH_INTERNAL_ASSERT(
      !fusion->isA<kir::Kernel>(),
      "Invalid container. Kernel container not allowed.\n");

  // Ensure we have propagated known values before evaluating extents
  expr_eval->propagateBoundValuesThroughExactMaps(fusion);

  analyzeReshapes(info, expr_eval);

  analyzeResizes(info, expr_eval);

  bool has_empty_tensor = false;
  for (auto ext : info->getDynamicExtentVals()) {
    if (expr_eval->evaluate(ext).value().as<int64_t>() == 0) {
      has_empty_tensor = true;
      break;
    }
  }
  // Find a minimal set of empty tensors to replace with full() calls
  // NOTE: this does a backward traversal from outputs.
  if (has_empty_tensor) {
    empty_tensors_ =
        EmptyBranchFinder(info->fusion(), expr_eval).getEmptyTensors();
  }
}

void DynamicTransformConcretizationInfo::analyzeReshapes(
    const DynamicTransformInitialInfo* info,
    ExpressionEvaluator* expr_eval) {
  for (const auto op : info->getDynamicReshapes()) {
    auto inp_tv = op->in()->as<TensorView>();
    auto out_tv = op->out()->as<TensorView>();

    // If there's no symblic axis, this is a static reshape op
    if (!out_tv->domain()->hasSymbolicAxis()) {
      return;
    }

    TORCH_INTERNAL_ASSERT(
        out_tv->hasRFactor(),
        "Unexpected output tv of ViewOp: ",
        out_tv->toString());

    const auto& inp_dom =
        TensorDomain::noReductions(inp_tv->getMaybeRFactorDomain());

    // Determine input shape using expr evaluator
    std::vector<int64_t> inp_shape(inp_dom.size(), 0);
    for (const auto i : c10::irange(inp_dom.size())) {
      auto inp_id = inp_dom.at(i);
      // This should have been validated when initially creating reshape
      // op, but just in case
      TORCH_INTERNAL_ASSERT(
          !inp_id->maybePartial(),
          "Invalid domain to reshape: ",
          inp_id->toString());
      auto extent_val = expr_eval->evaluate(inp_id->extent());
      TORCH_INTERNAL_ASSERT(
          extent_val.has_value(),
          "Cannot evaluate the extent of an input domain to reshape: ",
          inp_id->toString());
      TORCH_INTERNAL_ASSERT(
          extent_val->isInt(),
          "Invalid evaluated value of domain extent: ",
          inp_id->toString());
      TORCH_INTERNAL_ASSERT(
          extent_val->as<int64_t>() > 0,
          "Invalid input domain extent: ",
          extent_val->as<int64_t>());
      inp_shape.at(i) = extent_val->as<int64_t>();
    }

    const auto& out_dom = out_tv->getMaybeRFactorDomain();

    // Determine output shape using expr evaluator. Note there may be
    // one domain of extent -1
    std::vector<int64_t> out_shape(out_dom.size(), 0);
    bool extent_m1_found = false;
    for (const auto i : c10::irange(out_dom.size())) {
      auto out_id = out_dom.at(i);
      auto extent_val = expr_eval->evaluate(out_id->extent());
      TORCH_INTERNAL_ASSERT(
          extent_val.has_value(),
          "Cannot evaluate the extent of an output domain to reshape: ",
          out_id->toString());
      TORCH_INTERNAL_ASSERT(
          extent_val->isInt(),
          "Invalid evaluated value of domain extent: ",
          out_id->toString());
      const auto extent_int = extent_val->as<int64_t>();
      if (extent_int == -1) {
        TORCH_INTERNAL_ASSERT(
            !extent_m1_found,
            "Multiple output domains of size -1 not allowed",
            out_tv->toString());
        extent_m1_found = true;
      } else {
        TORCH_INTERNAL_ASSERT(
            extent_int > 0, "Invalid output domain extent: ", extent_int);
      }
      out_shape.at(i) = extent_int;
    }

    auto view_result = analyzeView(inp_tv, inp_shape, out_shape);

    reshape_transforms_.emplace_back(out_tv, view_result);
  }
}

void DynamicTransformConcretizationInfo::analyzeResizes(
    const DynamicTransformInitialInfo* info,
    ExpressionEvaluator* expr_eval) {
  for (const auto op : info->getDynamicResizes()) {
    auto out_id = op->out()->as<IterDomain>();

    TORCH_CHECK(
        out_id->getIterType() == IterType::Symbolic,
        "Found non-dynamic Resize in initial concretization info: ",
        op->toString());

    auto extent_val = expr_eval->evaluate(out_id->extent());
    TORCH_INTERNAL_ASSERT(
        extent_val.has_value(),
        "Cannot evaluate the extent of a resized domain: ",
        out_id->toString());
    TORCH_INTERNAL_ASSERT(
        extent_val->isInt(),
        "Invalid evaluated value of resized domain extent: ",
        out_id->toString());
    auto extent_int = extent_val->as<int64_t>();
    TORCH_INTERNAL_ASSERT(
        extent_int > 0,
        "Invalid resized domain extent ",
        extent_int,
        " for domain ",
        out_id->toString());

    auto iter_type =
        extent_int == 1 ? IterType::Broadcast : IterType::Iteration;

    resize_transforms_.emplace_back(out_id, iter_type);
  }
}

bool DynamicTransformConcretizationInfo::operator==(
    const DynamicTransformConcretizationInfo& other) const {
  if (this == &other) {
    return true;
  }

  if (fusion_ != other.fusion_) {
    return false;
  }

  if (reshape_transforms_.size() != other.reshape_transforms_.size() ||
      resize_transforms_.size() != other.resize_transforms_.size()) {
    return false;
  }

  for (const auto i : c10::irange(reshape_transforms_.size())) {
    const auto& transform = reshape_transforms_.at(i);
    const auto& other_transform = other.reshape_transforms_.at(i);
    if (transform != other_transform) {
      return false;
    }
  }

  for (const auto i : c10::irange(resize_transforms_.size())) {
    const auto& transform = resize_transforms_.at(i);
    const auto& other_transform = other.resize_transforms_.at(i);
    if (transform != other_transform) {
      return false;
    }
  }

  for (const auto i : c10::irange(empty_tensors_.size())) {
    const auto& et = empty_tensors_.at(i);
    const auto& other_et = other.empty_tensors_.at(i);
    if (et != other_et) {
      return false;
    }
  }

  return true;
}

DynamicTransformConcretizationInfo DynamicTransformConcretizationInfo::clone(
    IrCloner& ir_cloner) const {
  DynamicTransformConcretizationInfo cloned_info(
      static_cast<Fusion*>(ir_cloner.container()));
  for (const auto& [tv, analyze_result] : reshape_transforms_) {
    cloned_info.reshape_transforms_.emplace_back(
        ir_cloner.clone(tv),
        // reshape_transforms_ holds pairs of TensorView* and AnalyzeViewResult
        // AnalyzeViewResult can be copied directly as it holds no references to
        // Statements that would need cloning, only integer indices of axes.
        analyze_result);
  }
  for (const auto& [id, iter_type] : resize_transforms_) {
    cloned_info.resize_transforms_.emplace_back(
        ir_cloner.clone(id),
        // Similar to reshape_transforms_, we only clone the IterDomains in
        // resize_transforms_
        iter_type);
  }
  for (const auto& [tv, empty_axes] : empty_tensors_) {
    cloned_info.empty_tensors_.emplace_back(
        EmptyTensorDescriptor{ir_cloner.clone(tv), empty_axes});
  }
  return cloned_info;
}

std::string DynamicTransformConcretizationInfo::toString() const {
  std::stringstream ss;
  ss << "DynamicTransformConcretizationInfo\n";
  std::string indent = "  ";
  ss << indent << "Empty tensors:\n";
  for (const auto& kv : empty_tensors_) {
    ss << indent << indent << kv.tv->toString()
       << " has zero extent in these axes:";
    for (auto i : kv.empty_axes) {
      ss << " " << i;
    }
    ss << "\n";
  }
  ss << indent << "Reshape:\n";
  for (const auto& kv : reshape_transforms_) {
    ss << indent << indent << kv.first->toString() << ", "
       << kv.second.toString() << "\n";
  }
  ss << indent << "Resize:\n";
  for (const auto& [id, iter_type] : resize_transforms_) {
    ss << indent << indent << id->toString() << ", " << iter_type << "\n";
  }
  return ss.str();
}

//! Concretize a symbolic fusion with concrete transformation info
class DynamicTransformConcretizer : public OptOutMutator {
 public:
  DynamicTransformConcretizer(const DynamicTransformConcretizationInfo& info)
      : info_(info) {
    concretize();
  }

 private:
  void concretize();

  //! Set definitions of empty tensors to full() calls, replace reductions over
  //! empty axes with full calls.
  void removeEmptyBranches();

  //! Modify the Fusion by replacing tv with output of full() expression in
  //! outputs and all uses.
  void replaceByFull(
      TensorView* tv,
      std::vector<Val*>& new_shape,
      Val* fill_value = nullptr);

  void concretizeReshape();

  void concretizeResize();

  //! Use this instead of calling registerMutation directly, since it will also
  //! check that the concretized value is a valid input to all of its uses.
  void registerConcretization(Val* old_val, Val* new_val) {
    checkConcretizedUses(old_val, new_val);
    registerMutation(old_val, new_val);
  }

  //! Check uses of old_val to ensure that new_val does not violate
  //! assumptions. This is currently only used to check that inputs to SqueezeOp
  //! are marked broadcast during concretization.
  void checkConcretizedUses(Val* old_val, Val* new_val) const;

  using OptOutMutator::mutate;

  void mutate(TensorView* tv) final;

  void mutate(TensorDomain* td) final;

  //! Concretizes the root domain of a symbolic consumer tensor from
  //! its producer domains. Returns true if any root ID is concretized.
  bool propagateFromProducerToConsumer(TensorView* consumer);

 private:
  const DynamicTransformConcretizationInfo& info_;
};

void DynamicTransformConcretizer::concretize() {
  // Concretize all dynamic reshape ops
  concretizeReshape();

  // Set output IterTypes for dynamic resize ops
  concretizeResize();

  // Concretize empty tensors last.
  removeEmptyBranches();

  // Finally, propagate concretized domains
  auto all_stmts = StmtSort::getStmts(info_.fusion(), true);
  for (auto stmt : all_stmts) {
    if (stmt->isA<Val>()) {
      mutate(stmt);
    }
  }
}

void DynamicTransformConcretizer::removeEmptyBranches() {
  for (const auto& empty_tv_descr : info_.getEmptyTensors()) {
    auto tv = empty_tv_descr.tv;
    auto rfactor = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
    std::vector<Val*> new_shape;
    new_shape.reserve(rfactor.size());
    for (auto id : rfactor) {
      new_shape.push_back(id->extent());
    }
    for (auto ax : empty_tv_descr.empty_axes) {
      // Hard-code zero extent for empty axes. This lets us detect empty input
      // and output tensors during scheduling/execution.
      new_shape[ax] = tv->fusion()->zeroVal();
    }

    auto hasEmptyRootReductionAxis = [&empty_tv_descr](TensorView* out_tv) {
      return std::any_of(
          empty_tv_descr.empty_axes.begin(),
          empty_tv_descr.empty_axes.end(),
          [&out_tv](size_t ax) {
            return out_tv->getRootDomain().at(ax)->isReduction();
          });
    };

    // Given a TensorView, get a shape with hard-coded zeroes
    auto reduction_shape = [](TensorView* out_tv) -> std::vector<Val*> {
      auto nored_axes =
          TensorDomain::noReductions(out_tv->getMaybeRFactorDomain());
      // Output shape is simply the same as the original reduction. If there
      // were zeros in the non-Reduction axes, it would be replaced by
      // full() directly.
      std::vector<Val*> out_shape(nored_axes.size());
      std::transform(
          nored_axes.begin(),
          nored_axes.end(),
          out_shape.begin(),
          [](IterDomain* id) -> Val* { return id->extent(); });
      return out_shape;
    };

    // If expr is a ReductionOp or WelfordOp over some empty axes, replace it
    // with a call to full().
    for (auto use : tv->uses()) {
      if (auto rop = dynamic_cast<ReductionOp*>(use)) {
        auto out = rop->out()->as<TensorView>();
        if (hasEmptyRootReductionAxis(out)) {
          auto out_shape = reduction_shape(out);
          replaceByFull(out, out_shape);
        }
      } else if (auto wop = dynamic_cast<WelfordOp*>(use)) {
        auto avg = wop->outAvg()->as<TensorView>();
        auto var = wop->outVar()->as<TensorView>();
        auto N = wop->outN()->as<TensorView>();
        if (hasEmptyRootReductionAxis(avg)) {
          auto out_shape = reduction_shape(avg);
          auto nan = IrBuilder::create<Double>(0.0 / 0.0);
          replaceByFull(avg, out_shape, nan);
          replaceByFull(var, out_shape, nan);
          replaceByFull(N, out_shape);
        }
      }
    }
    replaceByFull(tv, new_shape);
  }
}

void DynamicTransformConcretizer::replaceByFull(
    TensorView* tv,
    std::vector<Val*>& new_shape,
    Val* fill_value) {
  if (!fill_value) {
    fill_value = tv->fusion()->zeroVal();
  }
  if (fill_value->getDataType().value() != tv->getDataType().value()) {
    fill_value = castOp(tv->getDataType().value(), fill_value);
  }
  auto mut_tv = full(new_shape, fill_value, tv->getDataType().value());
  registerConcretization(tv, mut_tv);
  OptOutMutator::mutate(tv);
  // Replace tv in Fusion outputs() if present
  if (tv->isFusionOutput()) {
    tv->fusion()->replaceOutput(tv, mut_tv);
  }
}

void DynamicTransformConcretizer::concretizeReshape() {
  // Concretize each reshape op.
  for (const auto& kv : info_.getReshapeTransforms()) {
    auto incomplete_out_tv = kv.first;
    const auto view_analysis = kv.second;

    auto inp_tv = ir_utils::producerTvsOf(incomplete_out_tv).at(0);

    auto concrete_reshape_out_tv = reshape(inp_tv, view_analysis);

    // We do the replacement directly here, but we must still check that the
    // replacement is valid
    checkConcretizedUses(incomplete_out_tv, concrete_reshape_out_tv);

    // Replace the old tensor with the new concretized tensor
    for (auto use_of_old_tv : incomplete_out_tv->uses()) {
      ir_utils::replaceValInExpr(
          use_of_old_tv, incomplete_out_tv, concrete_reshape_out_tv);
    }

    if (incomplete_out_tv->isFusionOutput()) {
      incomplete_out_tv->fusion()->replaceOutput(
          incomplete_out_tv, concrete_reshape_out_tv);
    }

    incomplete_out_tv->fusion()->removeVal(incomplete_out_tv);
  }
}

void DynamicTransformConcretizer::concretizeResize() {
  // Concretize each resize op.
  for (const auto& [id, iter_type] : info_.getResizeTransforms()) {
    TORCH_CHECK(
        id->definition() && id->definition()->isA<Resize>(),
        "Resized IterDomain must have a Resize definition");
    auto def = id->definition()->as<Resize>();
    auto new_id = IterDomain::resize(
        def->in(),
        def->leftExpand(),
        def->rightExpand(),
        id->isRFactorProduct(),
        iter_type);

    registerConcretization(id, new_id);
  }
}

void DynamicTransformConcretizer::checkConcretizedUses(
    Val* old_val,
    Val* new_val) const {
  for (const auto use : old_val->uses()) {
    use->checkConcretization(old_val, new_val);
  }
}

// Concretizes inherited symbolic domains. Note that when this is
// called, it is assumed that all dynamic ops themselves are
// concretized. Since symbolic IDs may be propagated down to
// consumers, those domains need to be concretized accordingly.
void DynamicTransformConcretizer::mutate(TensorView* tv) {
  if (!tv->domain()->hasSymbolicAxis()) {
    return;
  }

  // First, try to concretize the root domain as there may be symbolic
  // axes inherited from the producers
  propagateFromProducerToConsumer(tv);

  // If no root domain is altered by producer, we don't need to propagate back
  // up to rfactor. We could return early, but instead we go ahead and check the
  // root to rfactor transforms to be sure we have concretized any intermediate
  // IterDomains.

  // At this point, there should be no expr beyond rfactor root
  TORCH_INTERNAL_ASSERT(
      tv->getLeafDomain() == tv->getMaybeRFactorDomain(),
      "Invalid tensor: ",
      tv->toString());

  // If it has an rfactor root domain, the IterTypes of the rfactor
  // IDs may need to be updated as well. Traverse the rfactor exprs
  // and mutate the IterTypes of output IDs if symbolic.
  if (tv->hasRFactor()) {
    // Note that it is assumed that theres's no further expression
    // beyond the rfactor domain as asserted above
    auto all_id_exprs = StmtSort::getExprsBetween(
        tv->fusion(),
        {tv->getRootDomain().begin(), tv->getRootDomain().end()},
        {tv->getMaybeRFactorDomain().begin(),
         tv->getMaybeRFactorDomain().end()});
    for (auto expr : all_id_exprs) {
      // Assume outputs of IterDomain exprs are always IterDomains. If
      // the assumption is invalidated, the logic here would need to
      // be updated. Assert the assumption to immediately detect such
      // a case if happened.
      for (auto out_val : expr->outputs()) {
        TORCH_INTERNAL_ASSERT(
            out_val->isA<IterDomain>(),
            "Unexpected output: ",
            out_val->toString(),
            ". IterDomain was expected.");
      }

      // NOTE: We do not return early if all outputs are concrete as there may
      // still be concrete inputs. For example, a Symbolic IterDomain might be
      // padded with constant pad widths (1, 1), in which case although we do
      // not know the exact extent of the output, we know it is at least as
      // large as the sum of the pad widths, 2. In such cases, the output
      // IterDomain is concrete at definition, since if the extent is >1 we know
      // the IterType is Iteration. In these cases, we must continue to
      // concretize intermediate expressions between the root and R-factor
      // domain. See test DynamicTransform5_CUDA which demonstrates this
      // behavior.
      // NOTE: We also do not assume that if one output ID is symbolic, that
      // they all must be. See test FusionSliceForNanoGPT3_CUDA for an example
      // that does a static split by a factor of 16 of a symbolic input domain.
      // The static split in that case results in a concrete IterDomain with
      // extent 16 along with a symbolic one (extent ceilDiv(n / 16)).

      // Determine the output IterType
      IterType iter_type = IterType::Symbolic;
      for (auto inp_id : ir_utils::filterByType<IterDomain>(expr->inputs())) {
        auto updated_id = maybeMutated(inp_id)->as<IterDomain>();
        iter_type = ops::promoteIterType(iter_type, updated_id->getIterType());
      }
      TORCH_INTERNAL_ASSERT(
          iter_type != IterType::Symbolic,
          "Failed to concretize an output IterType for expression: ",
          expr->toString());

      // Update the IterType of each output
      for (auto out_id : ir_utils::filterByType<IterDomain>(expr->outputs())) {
        if (!out_id->isSymbolic()) {
          continue;
        }
        auto concretized_out_id =
            IterDomainBuilder(out_id).iter_type(iter_type).build();
        registerConcretization(out_id, concretized_out_id);
      }

      // The expr itself needs to be mutated as well in case the outputs are
      // mutated, which can be done by the mutate method
      OptOutMutator::mutate(expr);
    }
  }

  // Root and rfactor domains are updated. First mutate the
  // TensorDomain and then TensorView
  mutate(tv->domain());
  OptOutMutator::mutate(tv);
}

// Almost an exact copy of OptOutMutator::mutate(TensorDomain*), but
// the contiguity vector may need to be updated as well as symbolic
// domains may be mutated to broadcast domains, which means contiguity
// may need to be changed to nullopt
void DynamicTransformConcretizer::mutate(TensorDomain* td) {
  bool mutated = false;

  auto updateIdVec = [&](const std::vector<IterDomain*>& ids) {
    std::vector<IterDomain*> updated_ids;
    for (auto id : ids) {
      auto updated_id = maybeMutated(id)->as<IterDomain>();
      updated_ids.push_back(updated_id);
      if (!updated_id->sameAs(id)) {
        mutated = true;
      }
    }
    return updated_ids;
  };

  std::vector<IterDomain*> root_dom = updateIdVec(td->root());
  std::vector<IterDomain*> rfactor_dom = td->hasRFactor()
      ? updateIdVec(td->maybeRFactor())
      : std::vector<IterDomain*>();
  std::vector<IterDomain*> domain = updateIdVec(td->leaf());

  if (!mutated) {
    return;
  }

  // Update the contiguity vector. Drop the contig val if mutated to broadcast
  auto contig = td->contiguity();

  for (const auto i : c10::irange(td->maybeRFactor().size())) {
    auto original_id = td->maybeRFactor().at(i);
    if (original_id->getIterType() != IterType::Symbolic) {
      continue;
    }

    TORCH_INTERNAL_ASSERT(
        contig.at(i),
        "Unexpected to have a non-contig symbolic domain: ",
        original_id->toString());

    auto updated_id = td->hasRFactor() ? rfactor_dom.at(i) : root_dom.at(i);

    // If the concretized ID is a broadcast domain, drop the contig val
    if (updated_id->isBroadcast()) {
      contig.at(i) = std::nullopt;
    }
  }

  Val* mutated_val = IrBuilder::create<TensorDomain>(
      td->container(), root_dom, rfactor_dom, domain, contig);
  registerConcretization(td, mutated_val);
}

bool DynamicTransformConcretizer::propagateFromProducerToConsumer(
    TensorView* consumer) {
  if (consumer->definition() == nullptr ||
      !consumer->domain()->hasSymbolicAxis()) {
    return false;
  }

  const auto& root_domain = consumer->getRootDomain();

  auto def = consumer->definition();

  bool is_concretized = false;

  for (const auto i : c10::irange(root_domain.size())) {
    auto root_id = root_domain.at(i);
    if (root_id->getIterType() != IterType::Symbolic) {
      continue;
    }

    // Figure out the right IterType of this consumer root ID from its
    // corresponding producer IDs

    std::optional<IterType> id_type;

    for (auto producer : ir_utils::filterByType<TensorView>(def->inputs())) {
      PairwiseRootDomainMap root_map(producer, consumer);
      auto c2p = root_map.mapConsumerToProducer(
          consumer->domain(), producer->domain());

      TORCH_INTERNAL_ASSERT(
          c2p.find(root_id) != c2p.end(),
          "No input ID found to map with output ID: ",
          root_id->toString());

      auto input_id = c2p.at(root_id);
      TORCH_INTERNAL_ASSERT(
          input_id->getIterType() != IterType::Symbolic,
          "Producer ID not concretized: ",
          input_id->toString());

      if (id_type.has_value()) {
        id_type = ops::promoteIterType(*id_type, input_id->getIterType());
      } else {
        id_type = input_id->getIterType();
      }
    }

    TORCH_INTERNAL_ASSERT(
        id_type.has_value(),
        "Did not find id_type for consumer root domain ",
        root_id->toString(),
        ". Perhaps consumer def has no inputs. Consumer definition = ",
        def->toString());

    TORCH_INTERNAL_ASSERT(
        id_type != IterType::Symbolic,
        "Failed to concretize ",
        root_id->toString(),
        " of ",
        consumer->toString());

    auto concretized_id =
        IterDomainBuilder(root_id).iter_type(*id_type).build();

    registerConcretization(root_id, concretized_id);
    is_concretized = true;
  }

  return is_concretized;
}

DynamicTransformInitialInfo DynamicTransform::getInitialInfo(Fusion* fusion) {
  DynamicTransformInitialInfoBuilder builder(fusion);
  return builder.getInfo();
}

DynamicTransformConcretizationInfo DynamicTransform::getConcretizationInfo(
    Fusion* fusion,
    const DynamicTransformInitialInfo* info,
    ExpressionEvaluator* expr_eval) {
  return DynamicTransformConcretizationInfo(fusion, info, expr_eval);
}

DynamicTransformConcretizationInfo DynamicTransform::getConcretizationInfo(
    Fusion* fusion,
    const DynamicTransformInitialInfo* info,
    const KernelArgumentHolder* args) {
  ExpressionEvaluator expr_eval = executor_utils::bindInputs(*args, fusion);
  // Make sure all exactly mapped IDs have the same value in the
  // evaluator when any one of the IDs has a known value
  expr_eval.propagateBoundValuesThroughExactMaps(fusion);
  return DynamicTransformConcretizationInfo(fusion, info, &expr_eval);
}

void DynamicTransform::concretizeFusion(
    const DynamicTransformConcretizationInfo& info) {
  DynamicTransformConcretizer concretizer(info);
}

size_t DynamicTransformConcretizationInfo::hash() const {
  size_t hash = 0;
  for (const auto& [tv, view_result] : getReshapeTransforms()) {
    hash += view_result.hash();
  }
  return hash;
}

} // namespace nvfuser
