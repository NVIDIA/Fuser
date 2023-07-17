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
#include <ops/alias.h>
#include <ops/arith.h>
#include <ops/indexing.h>
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
  cloned_info.dynamic_reshaped_tvs_.reserve(dynamic_reshaped_tvs_.size());
  for (const auto op : dynamic_reshaped_tvs_) {
    if (op) {
      cloned_info.dynamic_reshaped_tvs_.push_back(ir_cloner.clone(op));
    }
  }
  cloned_info.dynamic_resized_tvs_.reserve(dynamic_resized_tvs_.size());
  for (const auto op : dynamic_resized_tvs_) {
    if (op) {
      cloned_info.dynamic_resized_tvs_.push_back(ir_cloner.clone(op));
    }
  }
  cloned_info.root_dynamic_vals_.reserve(root_dynamic_vals_.size());
  for (const auto v : root_dynamic_vals_) {
    if (v) {
      cloned_info.root_dynamic_vals_.insert(ir_cloner.clone(v));
    }
  }
  return cloned_info;
}

std::string DynamicTransformInitialInfo::toString() const {
  std::stringstream ss;
  ss << "DynamicTransformInitialInfo\n";
  std::string indent = "  ";
  ss << indent << "Dynamic reshaped TensorViews:\n";
  for (const auto& op : dynamic_reshaped_tvs_) {
    ss << indent << indent << op->toString() << "\n";
  }
  ss << indent << "Dynamic resized TensorViews:\n";
  for (const auto& op : dynamic_resized_tvs_) {
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
      info_.dynamic_reshaped_tvs_.push_back(out_tv);

      // Input and output extent expressions both affect concretization
      const auto& inp_dom =
          TensorDomain::noReductions(inp_tv->getMaybeRFactorDomain());
      for (const auto id : inp_dom) {
        leaf_dynamic_vals_.push_back(id->extent());
      }
      const auto& out_dom = out_tv->getMaybeRFactorDomain();
      for (const auto id : out_dom) {
        leaf_dynamic_vals_.push_back(id->extent());
      }
    }
  }

  //! Detect dynamic IterDomain transforms when handling TensorViews
  void handle(TensorView* tv) override {
    if (!tv->definition()) {
      return;
    }
    if (tv->definition()->isA<PadOp>()) {
      // Don't add this PadOp output if it is only used in a CatOp, since that
      // means it is an intermediate tensor used in `cat()`.
      if (!tv->isFusionOutput() &&
          std::all_of(tv->uses().begin(), tv->uses().end(), [](Expr* expr) {
            return expr->isA<CatOp>();
          })) {
        return;
      }
    } else if (
        !tv->definition()->isA<CatOp>() && !tv->definition()->isA<SliceOp>()) {
      return;
    }
    info_.dynamic_resized_tvs_.push_back(tv);
    const auto& rfd = tv->getMaybeRFactorDomain();
    for (auto id : rfd) {
      if (!id->definition() || id->getIterType() != IterType::Symbolic) {
        continue;
      }
      // extents of all symbolic outputs determine their IterTypes
      leaf_dynamic_vals_.push_back(id->extent());
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

void DynamicTransformConcretizationInfo::analyzeReshapes(
    ExpressionEvaluator* expr_eval) {
  const auto& reshape_tvs = initial_info_->getDynamicReshapedTensorViews();
  for (const auto tv_index : c10::irange(reshape_tvs.size())) {
    auto out_tv = reshape_tvs.at(tv_index);
    auto op = out_tv->definition()->as<ViewOp>();
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
    bool extent_m1_found = false;
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
      const auto extent_int = extent_val.as<int64_t>();
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

    reshape_transforms_.emplace_back(tv_index, view_result);
  }
}

void DynamicTransformConcretizationInfo::analyzeResizes(
    ExpressionEvaluator* expr_eval) {
  const auto& resize_tvs = initial_info_->getDynamicResizedTensorViews();
  for (const auto id_index : c10::irange(resize_tvs.size())) {
    auto out_tv = resize_tvs.at(id_index);
    const auto& rfd = out_tv->getMaybeRFactorDomain();
    std::vector<IterType> conc_iter_types;
    conc_iter_types.reserve(rfd.size());
    for (const auto out_id : rfd) {
      if (out_id->getIterType() != IterType::Symbolic) {
        conc_iter_types.push_back(out_id->getIterType());
        continue;
      }

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

      auto iter_type =
          extent_int == 1 ? IterType::Broadcast : IterType::Iteration;

      conc_iter_types.push_back(iter_type);
    }
    resize_itertypes_.emplace_back(id_index, conc_iter_types);
  }
}

bool DynamicTransformConcretizationInfo::operator==(
    const DynamicTransformConcretizationInfo& other) const {
  if (this == &other) {
    return true;
  }

  if (reshape_transforms_.size() != other.reshape_transforms_.size() ||
      resize_itertypes_.size() != other.resize_itertypes_.size()) {
    return false;
  }

  for (const auto i : c10::irange(reshape_transforms_.size())) {
    const auto& analysis = reshape_transforms_.at(i);
    const auto& other_analysis = other.reshape_transforms_.at(i);
    if (analysis != other_analysis) {
      return false;
    }
  }

  for (const auto i : c10::irange(resize_itertypes_.size())) {
    const auto& itertype = resize_itertypes_.at(i);
    const auto& other_itertype = other.resize_itertypes_.at(i);
    if (itertype != other_itertype) {
      return false;
    }
  }

  return true;
}

std::string DynamicTransformConcretizationInfo::toString() const {
  std::stringstream ss;
  ss << "DynamicTransformConcretizationInfo\n";
  std::string indent = "  ";
  ss << indent << "Reshape:\n";
  for (const auto& [tv_index, analyze_result] : reshape_transforms_) {
    auto tv = initial_info_->getDynamicReshapedTensorViews().at(tv_index);
    ss << indent << indent << tv->toString() << " (index=" << tv_index << "), "
       << analyze_result.toString() << "\n";
  }
  ss << indent << "Resize:\n";
  for (const auto& [tv_index, iter_types] : resize_itertypes_) {
    auto tv = initial_info_->getDynamicResizedTensorViews().at(tv_index);
    ss << indent << indent << tv->toString() << " (index=" << tv_index << "), "
       << iter_types << "\n";
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
    concretize();
  }

 private:
  void concretize();

  void concretizeReshape();

  void concretizeResize();

  void replaceTV(TensorView* old_tv, TensorView* new_tv);

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
  const DynamicTransformConcretizationInfo* info_;
};

void DynamicTransformConcretizer::concretize() {
  // First, concretize all dynamic reshape ops
  concretizeReshape();

  // Set output IterTypes for dynamic resize ops
  concretizeResize();

  // Finally, propagate concretized domains
  auto all_stmts = StmtSort::getStmts(info_->fusion(), true);
  for (auto stmt : all_stmts) {
    if (stmt->isA<Val>()) {
      mutate(stmt);
    }
  }
}

void DynamicTransformConcretizer::replaceTV(
    TensorView* old_tv,
    TensorView* new_tv) {
  // We do the replacement directly here, but we must still check that the
  // replacement is valid
  checkConcretizedUses(old_tv, new_tv);

  // Replace the old tensor with the new concretized tensor
  for (auto use_of_old_tv : old_tv->uses()) {
    ir_utils::replaceValInExpr(use_of_old_tv, old_tv, new_tv);
  }

  if (old_tv->isFusionOutput()) {
    old_tv->fusion()->replaceOutput(old_tv, new_tv);
  }

  info_->fusion()->removeVal(old_tv);
}

void DynamicTransformConcretizer::concretizeReshape() {
  // Concretize each reshape op.
  for (const auto& [tv_index, view_analysis] : info_->getReshapeTransforms()) {
    auto incomplete_out_tv =
        info_->initialInfo()->getDynamicReshapedTensorViews().at(tv_index);
    auto view_op = incomplete_out_tv->definition()->as<ViewOp>();
    auto inp_tv = view_op->in()->as<TensorView>();

    auto concrete_reshape_out_tv = reshape(inp_tv, view_analysis);

    replaceTV(incomplete_out_tv, concrete_reshape_out_tv);
  }
}

void DynamicTransformConcretizer::concretizeResize() {
  // Concretize each resize op.

  // Concretizes any Resize op as output iter_type Iteration
  auto concretizeResizesAsIteration = [&](TensorView* tv) {
    const auto& rfd = tv->getMaybeRFactorDomain();
    for (auto rf_id : rfd) {
      auto def = rf_id->definition();
      if (!def || !def->isA<Resize>()) {
        continue;
      }
      auto rop = def->as<Resize>();
      auto new_id = IterDomain::resize(
          rop->in(),
          rop->leftExpand(),
          rop->rightExpand(),
          rf_id->isRFactorProduct(),
          IterType::Iteration);

      registerConcretization(rf_id, new_id);
    }
    // This modifies tv to reflect the new ID
    OptOutMutator::mutate(tv->domain());
    OptOutMutator::mutate(tv);
  };

  for (const auto& [tv_index, iter_types] : info_->getResizeIterTypes()) {
    auto tv = info_->initialInfo()->getDynamicResizedTensorViews().at(tv_index);

    const auto& rfd = tv->getMaybeRFactorDomain();
    TORCH_CHECK(
        iter_types.size() == rfd.size(),
        "Number of IterTypes must match size of rfactor domain");
    // If all Resized iter_types are Iteration, we can just replace each Resize.
    // If we encounter any Resize that outputs a Broadcast, then we should
    // instead translate the TensorView op to something followed by a
    // broadcast().
    bool resize_to_broadcast = false;
    for (auto i : c10::irange(iter_types.size())) {
      auto rf_id = rfd.at(i);
      if (!rf_id->definition() || !rf_id->definition()->isA<Resize>()) {
        continue;
      }
      auto iter_type = iter_types.at(i);
      if (iter_type == IterType::Broadcast) {
        resize_to_broadcast = true;
        break;
      } else {
        TORCH_CHECK(
            iter_type == IterType::Iteration,
            "Concretized IterType should be Iteration or Broadcast. Found ",
            iter_type);
      }
    }

    // Replace entire TensorView here with another op plus a broadcast
    TORCH_CHECK(tv->definition(), "Resized TensorView must have definition");
    if (auto pop = dynamic_cast<PadOp*>(tv->definition())) {
      // NOTE: tv should not be an intermediate pad as part of a CatOp if it
      // appears here, since we only record the output of the CatOp as a dynamic
      // resize, explicitly avoiding the intermediate PadOps.
      if (resize_to_broadcast) { // Convert to select + broadcast
        // Collect symbolic resizes. There might be multiple dimensions that get
        // padded to broadcast size in one PadOp. We perform a separate select()
        // for each of these, then we perform any remaining non-broadcast pads,
        // and finally we do a single broadcast.
        auto rfd = tv->getMaybeRFactorDomain();
        std::vector<bool> broadcast_axes(rfd.size(), false);
        auto pad_widths = pop->getPadWidths();
        std::vector<Val*> new_pad_widths;
        auto selected_tv = pop->in()->as<TensorView>();
        // Loop in reverse order, to make select() dim easier to track, and
        // because getPadWidths does not reverse the order of pad axes
        for (auto i = rfd.size() - 1; i > 0; --i) {
          auto id = rfd.at(i);
          auto iter_type = iter_types.at(i);
          auto left_pad = pad_widths.at(i * 2);
          auto right_pad = pad_widths.at(i * 2 + 1);
          if (id->definition() && id->definition()->isA<Resize>()) {
            if (iter_type == IterType::Iteration) {
              new_pad_widths.push_back(left_pad);
              new_pad_widths.push_back(right_pad);
            } else if (iter_type == IterType::Broadcast) {
              // This corresponds to right_pad = left_pad + 1
              broadcast_axes.at(i) = true;
              selected_tv = select(selected_tv, i, left_pad);
            } else {
              TORCH_CHECK(
                  false,
                  "PadOp IterDomains must concretized to Iteration or Broadcast");
            }
          }
        }
        // select, then pad, then broadcast
        auto padded = pad(selected_tv, new_pad_widths);
        auto broadcasted = broadcast(padded, broadcast_axes);
        replaceTV(tv, broadcasted);
      } else {
        concretizeResizesAsIteration(tv);
      }
    } else if (auto cop = dynamic_cast<CatOp*>(tv->definition())) {
      if (resize_to_broadcast) {
        // If a CatOp results in a broadcast in the cat dimension, then one of
        // the inputs must be broadcast and the others are empty in that
        // dimension, so this should translate to a set operation using the
        // broadcasted input.
        //
        // This case is handled by empty tensor concretization; as empty tensors
        // are removed in RemoveEmptyPass, they are removed from the CatOp. When
        // a single tensor remains, the cat is replaced with set().
      } else {
        // The inputs to CatOp are actually padded versions of the original
        // inputs to cat(). So we concretize those intermediate padded versions
        // here.
        for (auto pad_input : cop->inputs()) {
          concretizeResizesAsIteration(pad_input->as<TensorView>());
        }
        auto cat_id = tv->getMaybeRFactorDomain().at(cop->concatenatedDim());
        auto new_cat_id =
            IterDomainBuilder(cat_id).iter_type(IterType::Iteration).build();
        registerConcretization(cat_id, new_cat_id);
        // This modifies tv to reflect the new ID
        OptOutMutator::mutate(tv->domain());
        OptOutMutator::mutate(tv);
      }
    } else if (dynamic_cast<SliceOp*>(tv->definition())) {
      if (resize_to_broadcast) {
        // Assumes slice expressions are normalized so slices that result in
        // broadcasts are selects + broadcasts.
        TORCH_CHECK(!resize_to_broadcast);
      } else {
        concretizeResizesAsIteration(tv);
      }
    } else {
      TORCH_CHECK(
          false,
          "Unhandled Resize-based op found: ",
          tv->definition()->toString());
    }
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
        auto mut_id = maybeMutated(out_id)->as<IterDomain>();
        if (!mut_id->isSymbolic()) {
          continue;
        }
        auto concretized_out_id =
            IterDomainBuilder(mut_id).iter_type(iter_type).build();
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
      root_map.mapSymbolic(true);
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

void DynamicTransform::concretizeFusion(
    Fusion* fusion,
    const DynamicTransformConcretizationInfo* info) {
  DynamicTransformConcretizer concretizer(fusion, info);
}

size_t DynamicTransformConcretizationInfo::hash() const {
  size_t hash = 0;
  for (const auto& [tv, view_result] : getReshapeTransforms()) {
    hashCombine(hash, (size_t)tv);
    hashCombine(hash, view_result.hash());
  }
  for (const auto& [tv, iter_types] : getResizeIterTypes()) {
    hashCombine(hash, (size_t)tv);
    for (auto iter_type : iter_types) {
      hashCombine(hash, (size_t)iter_type);
    }
  }
  return hash;
}

} // namespace nvfuser
