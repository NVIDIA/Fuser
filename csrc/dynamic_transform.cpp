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
  cloned_info.dynamic_resized_ids_.reserve(dynamic_resized_ids_.size());
  for (const auto op : dynamic_resized_ids_) {
    if (op) {
      cloned_info.dynamic_resized_ids_.push_back(ir_cloner.clone(op));
    }
  }
  cloned_info.dynamic_sliced_tvs_.reserve(dynamic_sliced_tvs_.size());
  for (const auto v : dynamic_sliced_tvs_) {
    if (v) {
      cloned_info.dynamic_sliced_tvs_.push_back(ir_cloner.clone(v));
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
  ss << indent << "Dynamic sliced TensorViews:\n";
  for (const auto& op : dynamic_sliced_tvs_) {
    ss << indent << indent << op->toString() << "\n";
  }
  ss << indent << "Dynamic resized IterDomains:\n";
  for (const auto& op : dynamic_resized_ids_) {
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
    if (tv->definition() && tv->definition()->isA<SliceOp>()) {
      if (tv->domain()->hasSymbolicAxis()) {
        info_.dynamic_sliced_tvs_.push_back(tv);
        auto root_dom = tv->getRootDomain();
        const auto ranges = tv->definition()->as<SliceOp>()->getRanges();
        TORCH_INTERNAL_ASSERT(
            ranges.size() == root_dom.size(),
            "Mismatch between number of slice ranges ",
            ranges.size(),
            " and size of root domain ",
            root_dom.size());
        for (auto i : c10::irange(root_dom.size())) {
          // input extent and start/stop/step values determine slice
          // concretization
          auto root_ext = root_dom.at(i)->getMaybeExpandedExtent();
          leaf_dynamic_vals_.push_back(root_ext);
          auto range = ranges.at(i);
          leaf_dynamic_vals_.push_back(range.start);
          leaf_dynamic_vals_.push_back(range.stop);
          leaf_dynamic_vals_.push_back(range.step);
        }
      }
      return;
    }
    const auto& rfd = tv->getMaybeRFactorDomain();
    for (auto id : rfd) {
      if (!id->definition() || id->getIterType() != IterType::Symbolic) {
        continue;
      }
      if (id->definition()->isA<Resize>()) {
        info_.dynamic_resized_ids_.push_back(id);
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

    reshape_transforms_.emplace_back(tv_index, view_result);
  }
}

void DynamicTransformConcretizationInfo::analyzeSlices(
    ExpressionEvaluator* expr_eval) {
  const auto& sliced_tvs = initial_info_->getDynamicSlicedTensorViews();
  for (auto tv_index : c10::irange(sliced_tvs.size())) {
    auto out_tv = sliced_tvs.at(tv_index);
    auto op = out_tv->definition()->as<SliceOp>();
    auto inp_tv = op->in()->as<TensorView>();
    const auto inp_dom =
        TensorDomain::noReductions(inp_tv->getMaybeRFactorDomain());
    const auto out_dom = out_tv->getMaybeRFactorDomain();
    const auto ranges = op->getRanges();
    std::vector<Concrete1DSliceDescriptor> slice_descs(inp_dom.size());
    for (auto i : c10::irange(inp_dom.size())) {
      const auto& range = ranges.at(i);

      auto start_opt = expr_eval->evaluate(range.start);
      TORCH_INTERNAL_ASSERT(
          start_opt.has_value(),
          "Could not evaluate start of slice range ",
          range.start);
      auto start = start_opt->as<int64_t>();

      auto stop_opt = expr_eval->evaluate(range.stop);
      TORCH_INTERNAL_ASSERT(
          stop_opt.has_value(),
          "Could not evaluate stop of slice range ",
          range.stop);
      auto stop = stop_opt->as<int64_t>();

      auto step_opt = expr_eval->evaluate(range.step);
      TORCH_INTERNAL_ASSERT(
          step_opt.has_value(),
          "Could not evaluate step of slice range ",
          range.step);
      auto step = step_opt->as<int64_t>();

      TORCH_INTERNAL_ASSERT(step != 0, "Slice step must not be zero");
      TORCH_INTERNAL_ASSERT(
          step == 1, "Slicing with step != 1 is not currently supported");

      auto inp_extent_opt =
          expr_eval->evaluate(inp_dom.at(i)->getMaybeExpandedExtent());
      TORCH_INTERNAL_ASSERT(
          inp_extent_opt.has_value(),
          "Could not evaluate slice input extent ",
          inp_dom.at(i)->getMaybeExpandedExtent());
      auto inp_extent = inp_extent_opt->as<int64_t>();

      auto getBranch = [&inp_extent](int64_t a) -> SliceIndexBranch {
        if (a <= -inp_extent) {
          return SliceIndexBranch::AlwaysZero;
        } else if (a < 0) {
          return SliceIndexBranch::Negative;
        } else if (a < inp_extent) {
          return SliceIndexBranch::Positive;
        } else {
          return SliceIndexBranch::AlwaysExtent;
        }
      };
      slice_descs[i].start_branch = getBranch(start);
      slice_descs[i].stop_branch = getBranch(stop);
      slice_descs[i].is_empty = (stop - start) * step <= 0;

      // The dynamic slice output has a purely symbolic extent. Here we evaluate
      // the proper extent to determine the output IterType.
      auto map_int_index = [&inp_extent](const int64_t& a) -> int64_t {
        if (a <= -inp_extent) {
          return 0;
        } else if (a < 0) {
          return -a;
        } else if (a < inp_extent) {
          return a;
        } else {
          return inp_extent;
        }
      };
      // actual size of sliced dimension is ceilDiv(stop - start, step) when
      // step > 0. When step < 0, that expression is off by one and instead the
      // extent in that case is ceilDiv(start - stop, -step).
      auto concrete_sliced_extent = step > 0
          ? (map_int_index(stop) - map_int_index(start) + step - 1) / step
          : (map_int_index(stop) - map_int_index(start) + step + 1) / step;

      if (concrete_sliced_extent == 1) {
        slice_descs[i].iter_type = IterType::Broadcast;
      }

      // Even though we will eventually replace this TV, there will still be
      // references to its extents in downstream uses. We will need to evaluate
      // these properly both in this analysis, and at concretization. Here we
      // bind the output extent so that downstream extents can be properly
      // computed during this analysis. After concretization, this will happen
      // via ExpressionEvaluator::propagateBoundValuesThroughExactMaps.
      expr_eval->bind(out_dom[i]->extent(), concrete_sliced_extent);
    }
    slice_descriptors_.emplace_back(tv_index, slice_descs);
  }
}

void DynamicTransformConcretizationInfo::analyzeResizes(
    ExpressionEvaluator* expr_eval) {
  const auto& resize_ids = initial_info_->getDynamicResizedIterDomains();
  for (const auto id_index : c10::irange(resize_ids.size())) {
    auto out_id = resize_ids.at(id_index);
    auto op = out_id->definition()->as<Resize>();

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

    resize_itertypes_.emplace_back(id_index, iter_type);
  }
}

bool DynamicTransformConcretizationInfo::operator==(
    const DynamicTransformConcretizationInfo& other) const {
  if (this == &other) {
    return true;
  }

  if (reshape_transforms_.size() != other.reshape_transforms_.size() ||
      resize_itertypes_.size() != other.resize_itertypes_.size() ||
      slice_descriptors_.size() != other.slice_descriptors_.size()) {
    return false;
  }

  for (const auto i : c10::irange(reshape_transforms_.size())) {
    const auto& analysis = reshape_transforms_.at(i);
    const auto& other_analysis = other.reshape_transforms_.at(i);
    if (analysis != other_analysis) {
      return false;
    }
  }

  for (const auto i : c10::irange(slice_descriptors_.size())) {
    const auto& desc = slice_descriptors_.at(i);
    const auto& other_desc = other.slice_descriptors_.at(i);
    if (desc != other_desc) {
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
  for (const auto& [id_index, iter_type] : resize_itertypes_) {
    auto id = initial_info_->getDynamicResizedIterDomains().at(id_index);
    ss << indent << indent << id->toString() << " (index=" << id_index << "), "
       << iter_type << "\n";
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

  void concretizeSlice();

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
  const DynamicTransformConcretizationInfo* info_;
};

void DynamicTransformConcretizer::concretize() {
  // First, concretize all dynamic reshape ops
  concretizeReshape();

  // Concretize dynamic slices
  concretizeSlice();

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

void DynamicTransformConcretizer::concretizeReshape() {
  // Concretize each reshape op.
  for (const auto& [tv_index, view_analysis] : info_->getReshapeTransforms()) {
    auto incomplete_out_tv =
        info_->initialInfo()->getDynamicReshapedTensorViews().at(tv_index);
    auto view_op = incomplete_out_tv->definition()->as<ViewOp>();
    auto inp_tv = view_op->in()->as<TensorView>();

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

    info_->fusion()->removeVal(incomplete_out_tv);
  }
}

void DynamicTransformConcretizer::concretizeSlice() {
  auto fusion = FusionGuard::getCurFusion();
  for (const auto& [tv_index, slice_descs] : info_->getSliceDescriptors()) {
    auto incomplete_out_tv =
        info_->initialInfo()->getDynamicSlicedTensorViews().at(tv_index);
    auto slice_op = incomplete_out_tv->definition()->as<SliceOp>();
    auto inp_tv = slice_op->input(0)->as<TensorView>();

    const auto& root_dom = incomplete_out_tv->getRootDomain();
    // Create new rfactor domain with potentially newly-resized root IDs
    std::vector<IterDomain*> new_rfactor(root_dom.size());

    bool is_empty = false;
    bool is_sliced = false;
    const auto ranges = slice_op->getRanges();
    auto map_index = [&fusion](
                         SliceIndexBranch branch, Val* a, Val* extent) -> Val* {
      if (branch == SliceIndexBranch::AlwaysExtent) {
        return extent;
      } else if (branch == SliceIndexBranch::Negative) {
        return SimplifyingIrBuilder::negExpr(a);
      } else if (branch == SliceIndexBranch::Positive) {
        return a;
      } else {
        return fusion->zeroVal();
      }
    };
    std::vector<Slice> new_ranges;
    new_ranges.reserve(ranges.size());
    for (auto i : c10::irange(root_dom.size())) {
      auto desc = slice_descs.at(i);
      if (desc.is_empty) {
        is_empty = true;
        // Use 0:0:1 as the canonical empty slice.
        new_ranges.push_back(
            {fusion->zeroVal(), fusion->zeroVal(), fusion->oneVal()});
      } else {
        auto range = ranges.at(i);
        auto inp_extent = root_dom.at(i)->getMaybeExpandedExtent();
        auto new_start = map_index(desc.start_branch, range.start, inp_extent);
        auto new_stop = map_index(desc.stop_branch, range.stop, inp_extent);
        new_ranges.push_back({new_start, new_stop, range.step});
        // Trivial slices correspond to 0:extent:1
        if (desc.start_branch != SliceIndexBranch::AlwaysZero ||
            desc.stop_branch != SliceIndexBranch::AlwaysExtent ||
            desc.step_branch != SliceStepBranch::One) {
          is_sliced = true;
        }
      }
    }

    TensorView* new_tv = nullptr;

    if (is_empty) {
      std::vector<Val*> new_shape(ranges.size());
      for (auto i : c10::irange(ranges.size())) {
        auto new_range = new_ranges.at(i);
        auto desc = slice_descs.at(i);
        // Depending on the step branch, we can use different output extent
        // expressions
        switch (desc.step_branch) {
          case SliceStepBranch::One:
            new_shape[i] =
                SimplifyingIrBuilder::subExpr(new_range.stop, new_range.start);
            break;
          case SliceStepBranch::GreaterThanOne:
            new_shape[i] = SimplifyingIrBuilder::ceilDivExpr(
                SimplifyingIrBuilder::subExpr(new_range.stop, new_range.start),
                new_range.step);
          case SliceStepBranch::Negative:
            new_shape[i] = SimplifyingIrBuilder::ceilDivExpr(
                SimplifyingIrBuilder::subExpr(new_range.start, new_range.stop),
                SimplifyingIrBuilder::negExpr(new_range.step));
        }
      }
      // TODO: process as empty tensor if is_empty
      auto dtype = incomplete_out_tv->getDataType().value();
      new_tv = full(new_shape, fusion->zeroVal(dtype), dtype);
    } else if (!is_sliced) {
      // Replace the slice with set()
      new_tv = set(inp_tv);
    } else {
      new_tv = slice(inp_tv, new_ranges, /*skip_symbolic*/ true);
    }

    // TODO: We need to update the maybeRFactorDomains of new_tv if there are
    // any Broadcast sliced dimensions.

    // We do the replacement directly here, but we must still check that the
    // replacement is valid
    checkConcretizedUses(incomplete_out_tv, new_tv);

    // Replace the old tensor with the new concretized tensor
    for (auto use_of_old_tv : incomplete_out_tv->uses()) {
      ir_utils::replaceValInExpr(use_of_old_tv, incomplete_out_tv, new_tv);
    }

    if (incomplete_out_tv->isFusionOutput()) {
      incomplete_out_tv->fusion()->replaceOutput(incomplete_out_tv, new_tv);
    }

    info_->fusion()->removeVal(incomplete_out_tv);
  }
}

void DynamicTransformConcretizer::concretizeResize() {
  // Concretize each resize op.
  for (const auto& [id_index, iter_type] : info_->getResizeIterTypes()) {
    auto id = info_->initialInfo()->getDynamicResizedIterDomains().at(id_index);
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

void DynamicTransform::concretizeFusion(
    Fusion* fusion,
    const DynamicTransformConcretizationInfo* info) {
  DynamicTransformConcretizer concretizer(fusion, info);
}

size_t DynamicTransformConcretizationInfo::hash() const {
  size_t hash = 0;
  for (const auto& [tv, view_result] : getReshapeTransforms()) {
    hashCombine(hash, view_result.hash());
  }
  for (const auto& [id, iter_type] : getResizeIterTypes()) {
    hashCombine(hash, (size_t)iter_type);
  }
  return hash;
}

} // namespace nvfuser
