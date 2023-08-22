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
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/cloner.h>
#include <ir/utils.h>
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

  cloned_info.dynamic_exprs_.reserve(dynamic_exprs_.size());
  std::transform(
      dynamic_exprs_.begin(),
      dynamic_exprs_.end(),
      std::back_inserter(cloned_info.dynamic_exprs_),
      [&ir_cloner](Expr* e) { return ir_cloner.clone(e); });

  cloned_info.maybe_zero_extents_set_.reserve(maybe_zero_extents_set_.size());
  std::transform(
      maybe_zero_extents_set_.begin(),
      maybe_zero_extents_set_.end(),
      std::inserter(
          cloned_info.maybe_zero_extents_set_,
          cloned_info.maybe_zero_extents_set_.begin()),
      [&ir_cloner](Val* v) { return ir_cloner.clone(v); });

  cloned_info.maybe_zero_extents_.reserve(maybe_zero_extents_.size());
  std::transform(
      maybe_zero_extents_.begin(),
      maybe_zero_extents_.end(),
      std::inserter(
          cloned_info.maybe_zero_extents_,
          cloned_info.maybe_zero_extents_.begin()),
      [&ir_cloner](Val* v) { return ir_cloner.clone(v); });

  cloned_info.root_dynamic_vals_.reserve(root_dynamic_vals_.size());
  std::transform(
      root_dynamic_vals_.begin(),
      root_dynamic_vals_.end(),
      std::inserter(
          cloned_info.root_dynamic_vals_,
          cloned_info.root_dynamic_vals_.begin()),
      [&ir_cloner](Val* v) { return ir_cloner.clone(v); });

  return cloned_info;
}

std::string DynamicTransformInitialInfo::toString() const {
  std::stringstream ss;
  ss << "DynamicTransformInitialInfo\n";
  std::string indent = "  ";
  ss << indent << "Dynamic expressions:\n";
  for (const auto& expr : dynamic_exprs_) {
    ss << indent << indent << expr->toString();
  }
  ss << indent << "Dynamic extent Vals:\n";
  for (const auto& v : maybe_zero_extents_) {
    ss << indent << indent << v->toString() << "\n";
  }
  ss << indent << "Root dynamic Vals:\n";
  for (const auto& v : root_dynamic_vals_) {
    ss << indent << indent << v->toString() << "\n";
  }
  ss << indent << "Input positions affecting concretization:\n";
  for (const auto& pos : scalar_inputs_affecting_concretization_) {
    ss << indent << indent << std::to_string(pos) << "\n";
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

    finalizeMaybeEmptyExtents();
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
      info_.dynamic_exprs_.push_back(op);

      // Input and output extent expressions both affect concretization
      for (const auto& id :
           TensorDomain::noReductions(inp_tv->getMaybeRFactorDomain())) {
        leaf_dynamic_vals_.push_back(id->getMaybeExpandedExtent());
      }
      for (const auto& id : out_tv->getMaybeRFactorDomain()) {
        leaf_dynamic_vals_.push_back(id->getMaybeExpandedExtent());
      }
    }
  }

  //! Detect possibly empty TensorViews and dynamic IterDomain transforms
  void handle(TensorView* tv) override {
    const auto& rfd = tv->getMaybeRFactorDomain();
    for (auto id : rfd) {
      auto ext = id->getMaybeExpandedExtent();
      if (!ext->isConstScalar() || ext->evaluateInt() == 0) {
        info_.maybe_zero_extents_set_.insert(ext);
        leaf_dynamic_vals_.push_back(ext);
      }
      if (!id->definition() || id->getIterType() != IterType::Symbolic) {
        continue;
      }
      if (id->definition()->isA<Resize>()) {
        info_.dynamic_exprs_.push_back(id->definition());
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

    // initial_info_ provides a set of Vals that are used for concretization.
    // Here we check which scalar inputs, if any, correspond to any of those
    // Vals. These will be the inputs that are explicitly used in the cache ID
    // for KernelArgumentHolder.
    auto dyn_vals = info_.getRootDynamicVals();
    for (const auto i : c10::irange(info_.fusion()->inputs().size())) {
      auto input = info_.fusion()->inputs().at(i);
      if (dyn_vals.find(input) != dyn_vals.end()) {
        info_.scalar_inputs_affecting_concretization_.insert(i);
      }
    }
  }

  //! Convert maybe_zero_extents_set_ to a vector so we can index it reliably
  void finalizeMaybeEmptyExtents() {
    info_.maybe_zero_extents_ = std::vector<Val*>(
        info_.maybe_zero_extents_set_.begin(),
        info_.maybe_zero_extents_set_.end());
    // Clear the corresponding set to free memory and speed up cloning
    info_.maybe_zero_extents_set_.clear();
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

DynamicTransformConcretizationInfo::DynamicTransformConcretizationInfo(
    const DynamicTransformInitialInfo* initial_info,
    ExpressionEvaluator* expr_eval)
    : initial_info_(initial_info) {
  TORCH_INTERNAL_ASSERT(
      !fusion()->isA<kir::Kernel>(),
      "Invalid container. Kernel container not allowed.\n");

  // Make sure all exactly mapped IDs have the same value in the
  // evaluator when any one of the IDs has a known value
  expr_eval->propagateBoundValuesThroughExactMaps(initial_info_->fusion());

  for (auto expr : initial_info_->getDynamicExprs()) {
    if (auto vop = dynamic_cast<ViewOp*>(expr)) {
      analyze(vop, expr_eval);
    } else if (auto rop = dynamic_cast<Resize*>(expr)) {
      analyze(rop, expr_eval);
    } else {
      TORCH_CHECK(false, "Unhandled dynamic Expr type: ", expr->toString());
    }
  }

  auto maybe_zero_extents = initial_info_->getMaybeZeroExtents();
  for (auto i : c10::irange(maybe_zero_extents.size())) {
    auto ext = maybe_zero_extents.at(i);
    auto ext_opt = expr_eval->evaluate(ext);
    TORCH_INTERNAL_ASSERT(
        ext_opt.hasValue(),
        "Could not evaluate dynamic extent: ",
        ext->toString());
    if (ext_opt == 0) {
      empty_extents_.push_back(i);
    }
  }
}

void DynamicTransformConcretizationInfo::analyze(
    ViewOp* op,
    ExpressionEvaluator* expr_eval) {
  auto out_tv = op->out()->as<TensorView>();
  auto inp_tv = op->in()->as<TensorView>();

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
        extent_val.hasValue(),
        "Cannot evaluate the extent of an input domain to reshape: ",
        inp_id->toString());
    TORCH_INTERNAL_ASSERT(
        extent_val.is<int64_t>(),
        "Invalid evaluated value of domain extent: ",
        inp_id->toString());
    TORCH_INTERNAL_ASSERT(
        extent_val.as<int64_t>() > 0,
        "Invalid input domain extent: ",
        extent_val.as<int64_t>());
    inp_shape.at(i) = extent_val.as<int64_t>();
  }

  const auto& out_dom = out_tv->getMaybeRFactorDomain();

  // Determine output shape using expr evaluator. Note there may be
  // one domain of extent -1
  std::vector<int64_t> out_shape(out_dom.size(), 0);
  for (const auto i : c10::irange(out_dom.size())) {
    auto out_id = out_dom.at(i);
    auto extent_val = expr_eval->evaluate(out_id->extent());
    TORCH_INTERNAL_ASSERT(
        extent_val.hasValue(),
        "Cannot evaluate the extent of an output domain to reshape: ",
        out_id->toString());
    TORCH_INTERNAL_ASSERT(
        extent_val.is<int64_t>(),
        "Invalid evaluated value of domain extent: ",
        out_id->toString());
    auto extent_int = extent_val.as<int64_t>();
    if (extent_int == -1) {
      // For non-constant Scalar sizes, check that we have not passed -1.
      TORCH_CHECK(
          out_id->extent()->isConst(),
          "Values of -1 passed to reshape must be constant at definition.")
    }
    out_shape.at(i) = extent_int;
  }

  auto view_result = analyzeView(inp_tv, inp_shape, out_shape);

  concretization_descriptors_.emplace_back(view_result);
}

void DynamicTransformConcretizationInfo::analyze(
    Resize* op,
    ExpressionEvaluator* expr_eval) {
  auto out_id = op->out()->as<IterDomain>();

  TORCH_CHECK(
      out_id->getIterType() == IterType::Symbolic,
      "Found non-dynamic Resize in initial concretization info: ",
      op->toString());

  auto extent_val = expr_eval->evaluate(out_id->extent());
  TORCH_INTERNAL_ASSERT(
      extent_val.hasValue(),
      "Cannot evaluate the extent of a resized domain: ",
      out_id->toString());
  TORCH_INTERNAL_ASSERT(
      extent_val.is<int64_t>(),
      "Invalid evaluated value of resized domain extent: ",
      out_id->toString());
  auto extent_int = extent_val.as<int64_t>();
  TORCH_INTERNAL_ASSERT(
      extent_int > 0,
      "Invalid resized domain extent ",
      extent_int,
      " for domain ",
      out_id->toString());

  auto iter_type = extent_int == 1 ? IterType::Broadcast : IterType::Iteration;

  concretization_descriptors_.emplace_back(iter_type);
}

bool DynamicTransformConcretizationInfo::operator==(
    const DynamicTransformConcretizationInfo& other) const {
  if (this == &other) {
    return true;
  }

  if (concretization_descriptors_.size() !=
          other.concretization_descriptors_.size() ||
      empty_extents_.size() != other.empty_extents_.size() ||
      !std::equal(
          concretization_descriptors_.begin(),
          concretization_descriptors_.end(),
          other.concretization_descriptors_.begin()) ||
      !std::equal(
          empty_extents_.begin(),
          empty_extents_.end(),
          other.empty_extents_.begin())) {
    return false;
  }

  return true;
}

std::string DynamicTransformConcretizationInfo::toString() const {
  std::stringstream ss;
  ss << "DynamicTransformConcretizationInfo\n";
  std::string indent = "  ";
  ss << indent << "Empty tensor extents:\n";
  for (const auto& i : empty_extents_) {
    auto ext = initial_info_->getMaybeZeroExtents().at(i);
    ss << indent << indent << ext->toString() << " is zero\n";
  }
  ss << indent << "Dynamic expression concretization descriptors:\n";
  for (const auto& desc_var : concretization_descriptors_) {
    if (auto analyze_result_ptr = std::get_if<AnalyzeViewResult>(&desc_var)) {
      ss << indent << indent
         << "AnalyzeViewResult: " << analyze_result_ptr->toString() << "\n";
    } else if (auto iter_type_ptr = std::get_if<IterType>(&desc_var)) {
      ss << indent << indent << "IterType: " << (*iter_type_ptr) << "\n";
    }
  }
  return ss.str();
}

//! Concretize a symbolic fusion with concrete transformation info
class DynamicTransformConcretizer : public OptOutMutator {
 public:
  DynamicTransformConcretizer(
      Fusion* fusion,
      const DynamicTransformConcretizationInfo* info)
      : info_(info) {
    TORCH_INTERNAL_ASSERT(
        fusion == info->fusion(),
        "Invalid DynamicTransformInitialInfo. The associated Fusion is different from the given Fusion");
    FusionGuard fg(fusion);
    concretize();
  }

 private:
  void concretize();

  void concretizeReshape(ViewOp* op, AnalyzeViewResult& view_analysis);

  void concretizeResize(Resize* op, IterType iter_type);

  void concretizeEmptyExtents();

  //! Use this instead of calling registerMutation directly, since it will
  //! also check that the concretized value is a valid input to all of its
  //! uses.
  void registerConcretization(Val* old_val, Val* new_val) {
    checkConcretizedUses(old_val, new_val);
    registerMutation(old_val, new_val);
  }

  //! Check uses of old_val to ensure that new_val does not violate
  //! assumptions. This is currently only used to check that inputs to
  //! SqueezeOp are marked broadcast during concretization.
  void checkConcretizedUses(Val* old_val, Val* new_val) const;

  using OptOutMutator::mutate;

  void mutate(TensorView* tv) final;

  void mutate(TensorDomain* td) final;

  //! Concretizes the root domain of a symbolic consumer tensor from
  //! its producer domains. Returns true if any root ID is concretized.
  bool propagateFromProducerToConsumer(TensorView* consumer);

 private:
  const DynamicTransformConcretizationInfo* info_;
};

void DynamicTransformConcretizer::concretize() {
  // First, concretize dynamic Exprs. We do this in reverse topological order.
  //
  // To understand why we use reverse topo order, consider what happens when
  // we concretize a dynamic reshape `auto tv3 = reshape(tv2, sh);`. To do
  // this, we create a new TensorView, let's call it `conc_tv` using the
  // particular AnalyzeViewResult in info_. Then, we replace `tv3` with
  // `conc_tv` in all uses of `tv3` using `ir_utils::replaceValInExpr`. That
  // function actually replaces those use Exprs with new ones and removes the
  // old pointers from the Fusion. As long as we do not dereference any of
  // these Exprs after this pass, and we do this pass in reverse topological
  // order, we will not encounter Expr* pointers that have been invalidated
  // due to `replaceValInExpr` like this.
  for (int i = (int)info_->getExprConcretizationDescriptors().size() - 1;
       i >= 0;
       --i) {
    auto op = info_->initialInfo()->getDynamicExprs().at(i);
    TORCH_CHECK(op != nullptr, "Dynamic expression must not be nullptr");
    auto desc_var = info_->getExprConcretizationDescriptors().at(i);
    if (auto vop = dynamic_cast<ViewOp*>(op)) {
      if (auto analyze_result_ptr = std::get_if<AnalyzeViewResult>(&desc_var)) {
        concretizeReshape(vop, *analyze_result_ptr);
      } else {
        TORCH_CHECK(
            false,
            "Dynamic ViewOp expects AnalyzeViewResult descriptor but found variant index ",
            desc_var.index());
      }
    } else if (auto rop = dynamic_cast<Resize*>(op)) {
      if (auto iter_type_ptr = std::get_if<IterType>(&desc_var)) {
        concretizeResize(rop, *iter_type_ptr);
      } else {
        TORCH_CHECK(
            false,
            "Dynamic Resize expects IterType descriptor but found variant index ",
            desc_var.index());
      }
    } else {
      TORCH_CHECK(false, "Unhandled dynamic Expr type: ", op->toString());
    }
  }

  // Registers replacement of all empty extents with zeroVal(). This does not
  // modify the Fusion.
  concretizeEmptyExtents();

  // Finally, propagate concretized domains
  auto all_stmts = StmtSort::getStmts(
      info_->fusion(),
      /*traverse_members*/ true,
      /*traverse_attributes*/ true,
      /*traverse_siblings*/ true);
  for (auto tv : ir_utils::filterByType<TensorView>(all_stmts)) {
    mutate(tv);
  }
}

void DynamicTransformConcretizer::concretizeEmptyExtents() {
  auto fusion = FusionGuard::getCurFusion();
  for (const auto& ext_index : info_->getEmptyExtents()) {
    auto ext = info_->initialInfo()->getMaybeZeroExtents().at(ext_index);
    auto zero = fusion->zeroVal(ext->getDataType().value());
    auto uses = ext->uses();
    for (auto use : uses) {
      ir_utils::replaceValInExpr(use, ext, zero);
    }
    // Register the concretization of this scalar, which allows us to replace
    // it whenever it is used as an extent member of an IterDomain.
    //
    // When we ext in all uses above, it affects downstream expressions. For
    // example we might replace i0 with 0 in (i0 + i1) + i2 to form (0 + i1) +
    // i2. However, i0 itself might be used as the extent, start, or stop
    // values in an IterDomain, so we register the concretization here so that
    // we can replace these values whenever we encounter them.
    registerConcretization(ext, zero);
  }
}

void DynamicTransformConcretizer::concretizeReshape(
    ViewOp* view_op,
    AnalyzeViewResult& view_analysis) {
  auto incomplete_out_tv = view_op->out()->as<TensorView>();
  auto inp_tv = view_op->in()->as<TensorView>();

  auto concrete_reshape_out_tv = reshape(inp_tv, view_analysis);

  // We do the replacement directly here, but we must still check that the
  // replacement is valid
  checkConcretizedUses(incomplete_out_tv, concrete_reshape_out_tv);

  // Extent expressions often change when concretizing a reshape. Here we
  // replace these in all downstream expressions so that the Fusion looks
  // just like it would have if we had used a static reshape instead.
  auto old_rfactor = incomplete_out_tv->getMaybeRFactorDomain();
  auto new_rfactor = concrete_reshape_out_tv->getMaybeRFactorDomain();
  TORCH_INTERNAL_ASSERT(
      old_rfactor.size() == new_rfactor.size(),
      "Concretized reshape rfactor size does not match symbolic rfactor");
  for (auto idx : c10::irange(new_rfactor.size())) {
    auto old_extent = old_rfactor.at(idx)->extent();
    auto new_extent = new_rfactor.at(idx)->extent();
    // If the old extent did not have a definition, we don't need to replace
    // it, since it will get bound whenever this tensor is a segmentation
    // edge.
    if (old_extent->definition() && !new_extent->sameAs(old_extent)) {
      registerConcretization(old_extent, new_extent);
    }
  }

  // Replace the old tensor with the new concretized tensor
  auto uses = incomplete_out_tv->uses();
  for (auto use_of_old_tv : uses) {
    ir_utils::replaceValInExpr(
        use_of_old_tv, incomplete_out_tv, concrete_reshape_out_tv);
  }

  if (incomplete_out_tv->isFusionOutput()) {
    incomplete_out_tv->fusion()->replaceOutput(
        incomplete_out_tv, concrete_reshape_out_tv);
  }

  info_->fusion()->removeVal(incomplete_out_tv);
}

void DynamicTransformConcretizer::concretizeResize(
    Resize* op,
    IterType iter_type) {
  // Concretize each resize op.
  auto out_id = op->out()->as<IterDomain>();
  auto new_id = IterDomain::resize(
      op->in(),
      op->leftExpand(),
      op->rightExpand(),
      out_id->isRFactorProduct(),
      iter_type);

  registerConcretization(out_id, new_id);
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
  for (auto root_id : tv->getRootDomain()) {
    // This will register root_id for mutation if its extent, start, or
    // stop_offset is registered for mutation
    OptOutMutator::mutate(root_id);
  }

  // First, try to concretize the root domain as there may be symbolic
  // axes inherited from the producers
  propagateFromProducerToConsumer(tv);

  // If no root domain is altered by producer, we don't need to propagate back
  // up to rfactor. We could return early, but instead we go ahead and check
  // the root to rfactor transforms to be sure we have concretized any
  // intermediate IterDomains.

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
      // IterDomain is concrete at definition, since if the extent is >1 we
      // know the IterType is Iteration. In these cases, we must continue to
      // concretize intermediate expressions between the root and R-factor
      // domain. See test DynamicTransform5_CUDA which demonstrates this
      // behavior.
      // NOTE: We also do not assume that if one output ID is symbolic, that
      // they all must be. See test FusionSliceForNanoGPT3_CUDA for an example
      // that does a static split by a factor of 16 of a symbolic input
      // domain. The static split in that case results in a concrete
      // IterDomain with extent 16 along with a symbolic one (extent ceilDiv(n
      // / 16)).

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
            IterDomainBuilder(maybeMutated(out_id)->as<IterDomain>())
                .iter_type(iter_type)
                .build();
        registerConcretization(out_id, concretized_out_id);
      }

      // expr must be mutated in order to set it as the definition for the
      // concretized outputs.
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
        IterDomainBuilder(maybeMutated(root_id)->as<IterDomain>())
            .iter_type(*id_type)
            .build();

    registerConcretization(root_id, concretized_id);
    is_concretized = true;
  }

  return is_concretized;
}

DynamicTransformInitialInfo DynamicTransform::getInitialInfo(Fusion* fusion) {
  DynamicTransformInitialInfoBuilder builder(fusion);
  return builder.getInfo();
}

void DynamicTransform::concretizeFusion(
    Fusion* fusion,
    const DynamicTransformConcretizationInfo* info) {
  DynamicTransformConcretizer concretizer(fusion, info);
}

size_t DynamicTransformConcretizationInfo::hash() const {
  size_t hash = 0;
  for (const auto& desc_var : getExprConcretizationDescriptors()) {
    if (auto view_result_ptr = std::get_if<AnalyzeViewResult>(&desc_var)) {
      hashCombine(hash, view_result_ptr->hash());
    } else if (auto iter_type_ptr = std::get_if<IterType>(&desc_var)) {
      hashCombine(hash, (size_t)(*iter_type_ptr));
    }
  }
  return hash;
}

} // namespace nvfuser
