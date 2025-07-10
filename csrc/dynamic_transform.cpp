// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/utils.h>
#include <dynamic_transform.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/cloner.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <ops/utils.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/executor_utils.h>
#include <transform_iter.h>
#include <transform_replay.h>
#include <transform_view.h>
#include <utils.h>

#include <optional>

namespace nvfuser {

DynamicTransformInitialInfo DynamicTransformInitialInfo::clone(
    IrCloner& ir_cloner) const {
  DynamicTransformInitialInfo cloned_info(
      static_cast<Fusion*>(ir_cloner.container()));
  cloned_info.dynamic_reshaped_tvs_.reserve(dynamic_reshaped_tvs_.size());
  for (const auto tv : dynamic_reshaped_tvs_) {
    cloned_info.dynamic_reshaped_tvs_.push_back(ir_cloner.clone(tv));
  }
  cloned_info.dynamic_resized_ids_.reserve(dynamic_resized_ids_.size());
  for (const auto id : dynamic_resized_ids_) {
    cloned_info.dynamic_resized_ids_.push_back(ir_cloner.clone(id));
  }
  cloned_info.dynamic_expanded_tvs_.reserve(dynamic_expanded_tvs_.size());
  for (const auto tv : dynamic_expanded_tvs_) {
    cloned_info.dynamic_expanded_tvs_.push_back(ir_cloner.clone(tv));
  }
  cloned_info.dynamic_factory_tvs_.reserve(dynamic_factory_tvs_.size());
  for (const auto v : dynamic_factory_tvs_) {
    cloned_info.dynamic_factory_tvs_.push_back(ir_cloner.clone(v));
  }
  cloned_info.dynamic_topk_tvs_.reserve(dynamic_topk_tvs_.size());
  for (const auto v : dynamic_topk_tvs_) {
    cloned_info.dynamic_topk_tvs_.push_back(ir_cloner.clone(v));
  }
  cloned_info.maybe_zero_extents_set_.reserve(maybe_zero_extents_set_.size());
  for (const auto v : maybe_zero_extents_set_) {
    cloned_info.maybe_zero_extents_set_.insert(ir_cloner.clone(v));
  }
  cloned_info.maybe_zero_extents_.reserve(maybe_zero_extents_.size());
  for (const auto v : maybe_zero_extents_) {
    cloned_info.maybe_zero_extents_.push_back(ir_cloner.clone(v));
  }
  cloned_info.root_dynamic_vals_.reserve(root_dynamic_vals_.size());
  for (const auto v : root_dynamic_vals_) {
    cloned_info.root_dynamic_vals_.insert(ir_cloner.clone(v));
  }
  return cloned_info;
}

std::string DynamicTransformInitialInfo::toString() const {
  std::stringstream ss;
  ss << "DynamicTransformInitialInfo\n";
  indent(ss, 1) << "Dynamic reshaped TensorViews:\n";
  for (const auto& tv : dynamic_reshaped_tvs_) {
    indent(ss, 2) << tv->toString() << "\n";
  }
  indent(ss, 1) << "Dynamic resized IterDomains:\n";
  for (const auto& id : dynamic_resized_ids_) {
    indent(ss, 2) << id->toString() << "\n";
  }
  indent(ss, 1) << "Dynamic expanded TensorViews:\n";
  for (const auto& tv : dynamic_expanded_tvs_) {
    indent(ss, 2) << tv->toString() << "\n";
  }
  indent(ss, 1) << "Dynamic factory-function output TensorViews:\n";
  for (const auto& tv : dynamic_factory_tvs_) {
    indent(ss, 2) << tv->toString() << "\n";
  }
  indent(ss, 1) << "Dynamic TopK output TensorViews:\n";
  for (const auto& tv : dynamic_topk_tvs_) {
    indent(ss, 2) << tv->toString() << "\n";
  }
  indent(ss, 1) << "Dynamic extent Vals:\n";
  for (const auto& v : maybe_zero_extents_) {
    indent(ss, 2) << v->toInlineString() << "\n";
  }
  indent(ss, 1) << "Root dynamic Vals:\n";
  for (const auto& v : root_dynamic_vals_) {
    indent(ss, 2) << v->toInlineString() << "\n";
  }
  return ss.str();
}

//! Gather information about concretizing transformations without
//! concrete input sizes.
class DynamicTransformInitialInfoBuilder : public IterVisitor {
 public:
  DynamicTransformInitialInfoBuilder(Fusion* fusion) : info_(fusion) {
    NVF_ERROR(
        !fusion->isA<kir::Kernel>(),
        "Invalid container. Kernel container not allowed.\n");

    traverseTo(fusion->getTerminatingOutputs(), false, false);

    finalizeDynamicVals();

    finalizeMaybeEmptyExtents();
  }

  const auto& getInfo() const {
    return info_;
  }

 private:
  using IterVisitor::dispatch;
  using IterVisitor::handle;

  void dispatch(Expr* expr) override {
    // Detect factory methods by checking whether there are no TensorView inputs
    if (std::none_of(
            expr->inputs().begin(), expr->inputs().end(), [](Val* inp) {
              return inp->vtype() == ValType::TensorView;
            })) {
      for (Val* out_val : expr->outputs()) {
        if (TensorView* out_tv = dynamic_cast<TensorView*>(out_val)) {
          const std::vector<IterDomain*>& out_rf = out_tv->getLogicalDomain();
          if (std::any_of(out_rf.begin(), out_rf.end(), [](IterDomain* id) {
                return id->isSymbolic();
              })) {
            info_.dynamic_factory_tvs_.push_back(out_tv);
          }
        } else {
          // Factory ops have only TensorView outputs, so we can skip Exprs that
          // have scalar outputs
          continue;
        }
      }
    }
    IterVisitor::dispatch(expr);
  }

  //! Find views that have symbolic outputs
  void handle(ViewOp* op) override {
    auto inp_tv = op->in()->as<TensorView>();
    auto out_tv = op->out()->as<TensorView>();
    // If there's no symbolic axis, this is a static reshape op
    if (out_tv->domain()->hasSymbolicAxis()) {
      info_.dynamic_reshaped_tvs_.push_back(out_tv);

      // Input and output extent expressions both affect concretization
      for (const auto& id :
           TensorDomain::noReductions(inp_tv->getLogicalDomain())) {
        loop_dynamic_vals_.push_back(id->getMaybeExpandedExtent());
      }
      for (const auto& id : out_tv->getLogicalDomain()) {
        loop_dynamic_vals_.push_back(id->getMaybeExpandedExtent());
      }
    }
  }

  //! Find TopK operations that have symbolic outputs
  void handle(TopKOp* op) override {
    auto out_values = op->outValues()->as<TensorView>();

    // Check if K of TopK is symbolic
    if (op->k()->isConstScalar()) {
      return;
    }

    info_.dynamic_topk_tvs_.push_back(out_values);

    // The K parameter affects concretization
    loop_dynamic_vals_.push_back(op->k());

    const auto topk_dim = op->dim();
    NVF_ERROR(
        topk_dim >= 0 && topk_dim < std::ssize(out_values->getLogicalDomain()),
        "Invalid TopK dimension ",
        topk_dim);

    auto topk_id = out_values->getLogicalDomain()[topk_dim];
    loop_dynamic_vals_.push_back(topk_id->extent());
  }

  //! Find expands that have symbolic outputs. Of those, check whether the
  //! extents match between input and output axes. If not, mark as dynamic.
  void handle(ExpandOp* op) override {
    auto inp_tv = op->in()->as<TensorView>();
    auto out_tv = op->out()->as<TensorView>();
    // If there's no symbolic axis, this is a static expand op
    bool is_dynamic = false;
    // Loop over all axes, check whether any expansions are undetermined
    const std::vector<IterDomain*> inp_logical =
        TensorDomain::noReductions(inp_tv->getLogicalDomain());
    const std::vector<IterDomain*>& out_root = out_tv->getMaybeRootDomain();
    NVF_ERROR(inp_logical.size() == out_root.size());
    for (auto i : arange((int64_t)out_root.size())) {
      IterDomain* out_id = out_root[i];
      if (!out_id->isSymbolic()) {
        continue;
      }
      Val* in_extent = inp_logical[i]->extent();
      Val* out_extent = out_id->extent();
      if (out_extent->sameAs(in_extent)) {
        // Not expanding this axis
        continue;
      }
      loop_dynamic_vals_.push_back(in_extent);
      loop_dynamic_vals_.push_back(out_extent);
      is_dynamic = true;
    }
    if (is_dynamic) {
      info_.dynamic_expanded_tvs_.push_back(out_tv);
    }
  }

  //! Detect possibly empty TensorViews and dynamic IterDomain transforms
  void handle(TensorView* tv) override {
    const auto& logical_dom = tv->getLogicalDomain();
    ExpressionEvaluator ee;
    for (auto id : logical_dom) {
      if (!id->getMaybeExpandedExtent()->isConstScalar() ||
          id->getMaybeExpandedExtent()->evaluate().as<int64_t>() == 0) {
        info_.maybe_zero_extents_set_.insert(id->getMaybeExpandedExtent());
        loop_dynamic_vals_.push_back(id->getMaybeExpandedExtent());
      }
      if (!id->definition() || id->getIterType() != IterType::Symbolic) {
        continue;
      }
      if (id->definition()->isA<Resize>()) {
        info_.dynamic_resized_ids_.push_back(id);
        // extent of output determines its IterType
        loop_dynamic_vals_.push_back(id->extent());
      }
    }
  }

  //! Process vector of loop dynamic values by finding inputs and recording the
  //! result into info_
  void finalizeDynamicVals() {
    const auto inputs = InputsOf::outputs(loop_dynamic_vals_);
    info_.root_dynamic_vals_.insert(inputs.begin(), inputs.end());

    // initial_info_ provides a set of Vals that are used for concretization.
    // Here we check which scalar inputs, if any, correspond to any of those
    // Vals. These will be the inputs that are explicitly used in the cache ID
    // for KernelArgumentHolder.
    auto dyn_vals = info_.getRootDynamicVals();
    for (const auto i : arange((int64_t)info_.fusion()->inputs().size())) {
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
  std::vector<Val*> loop_dynamic_vals_;
};

DynamicTransformConcretizationInfo::DynamicTransformConcretizationInfo(
    const DynamicTransformInitialInfo* initial_info,
    ExpressionEvaluator* expr_eval,
    ExactLogicalDomainMap* exact_map)
    : initial_info_(initial_info) {
  NVF_ERROR(
      !fusion()->isA<kir::Kernel>(),
      "Invalid container. Kernel container not allowed.\n");

  // Make sure all exactly mapped IDs have the same value in the
  // evaluator when any one of the IDs has a known value
  expr_eval->propagateBoundValuesThroughExactMaps(
      initial_info_->fusion(), exact_map);

  analyzeReshapes(expr_eval);

  analyzeResizes(expr_eval);

  analyzeExpands(expr_eval);

  analyzeFactoryOutputs(expr_eval);

  analyzeTopK(expr_eval);

  auto maybe_zero_extents = initial_info_->getMaybeZeroExtents();
  for (auto i : arange((int64_t)maybe_zero_extents.size())) {
    auto ext = maybe_zero_extents.at(i);
    auto ext_opt = expr_eval->evaluate(ext);
    NVF_ERROR(
        ext_opt.hasValue(),
        "Could not evaluate dynamic extent: ",
        ext->toString());
    if (ext_opt == 0) {
      empty_extents_.push_back(i);
    }
  }
}

void DynamicTransformConcretizationInfo::analyzeReshapes(
    ExpressionEvaluator* expr_eval) {
  const auto& reshape_tvs = initial_info_->getDynamicReshapedTensorViews();
  for (const auto tv_index : arange((int64_t)reshape_tvs.size())) {
    auto out_tv = reshape_tvs.at(tv_index);
    auto op = out_tv->definition()->as<ViewOp>();
    auto inp_tv = op->in()->as<TensorView>();

    // If there's no symblic axis, this is a static reshape op
    if (!out_tv->domain()->hasSymbolicAxis()) {
      return;
    }

    NVF_ERROR(
        out_tv->hasRoot(),
        "Unexpected output tv of ViewOp: ",
        out_tv->toString());

    const auto& inp_dom =
        TensorDomain::noReductions(inp_tv->getLogicalDomain());

    // Determine input shape using expr evaluator
    std::vector<int64_t> inp_shape(inp_dom.size(), 0);
    bool is_empty = false;
    for (const auto i : arange((int64_t)inp_dom.size())) {
      auto inp_id = inp_dom.at(i);
      // This should have been validated when initially creating reshape
      // op, but just in case
      NVF_ERROR(
          !inp_id->maybePartial(),
          "Invalid domain to reshape: ",
          inp_id->toString());
      auto extent_val = expr_eval->evaluate(inp_id->getMaybeExpandedExtent());
      NVF_ERROR(
          extent_val.hasValue(),
          "Cannot evaluate the extent of an input domain to reshape: ",
          inp_id->toString());
      NVF_ERROR(
          extent_val.is<int64_t>(),
          "Invalid evaluated value of domain extent: ",
          inp_id->toString());
      NVF_ERROR(
          extent_val.as<int64_t>() >= 0,
          "Invalid input domain extent: ",
          extent_val.as<int64_t>());
      inp_shape.at(i) = extent_val.as<int64_t>();
      if (inp_shape.at(i) == 0l) {
        is_empty = true;
      }
    }

    const auto& out_dom = out_tv->getLogicalDomain();

    // Determine output shape using expr evaluator. Note there may be
    // one domain of extent -1
    std::vector<int64_t> out_shape(out_dom.size(), 0);
    std::vector<int64_t> out_symbolic_sizes;
    for (const auto i : arange((int64_t)out_dom.size())) {
      auto out_id = out_dom.at(i);
      auto extent_val = expr_eval->evaluate(out_id->extent());
      NVF_ERROR(
          extent_val.hasValue(),
          "Cannot evaluate the extent of an output domain to reshape: ",
          out_id->toString());
      NVF_ERROR(
          extent_val.is<int64_t>(),
          "Invalid evaluated value of domain extent: ",
          out_id->toString());
      auto extent_int = extent_val.as<int64_t>();
      if (extent_int == -1) {
        // For non-constant Scalar sizes, check that we have not passed -1.
        NVF_CHECK(
            is_empty || out_id->extent()->isConst(),
            "Values of -1 passed to reshape must be constant at definition.")
      }
      out_shape.at(i) = extent_int;
      if (is_empty) {
        if (extent_int == 1l) {
          // Indicates we should concretize to IterType::Broadcast
          out_symbolic_sizes.push_back(1l);
        } else if (extent_int == 0l || extent_int == -1l) {
          // Indicates we should concretize to IterType::Iteration and
          // concretize extent to 0
          out_symbolic_sizes.push_back(0l);
        } else {
          // Indicates we should concretize to IterType::Iteration
          out_symbolic_sizes.push_back(-1l);
        }
      }
    }

    if (is_empty) {
      reshape_transforms_.emplace_back(tv_index, out_symbolic_sizes);
    } else {
      reshape_transforms_.emplace_back(
          tv_index, analyzeView(inp_tv, inp_shape, out_shape));
    }
  }
}

void DynamicTransformConcretizationInfo::analyzeResizes(
    ExpressionEvaluator* expr_eval) {
  const auto& resize_ids = initial_info_->getDynamicResizedIterDomains();
  for (const auto id_index : arange((int64_t)resize_ids.size())) {
    auto out_id = resize_ids.at(id_index);
    auto op = out_id->definition()->as<Resize>();

    NVF_CHECK(
        out_id->getIterType() == IterType::Symbolic,
        "Found non-dynamic Resize in initial concretization info: ",
        op->toString());

    auto extent_val = expr_eval->evaluate(out_id->getMaybeExpandedExtent());
    NVF_ERROR(
        extent_val.hasValue(),
        "Cannot evaluate the extent of a resized domain: ",
        out_id->toString());
    NVF_ERROR(
        extent_val.is<int64_t>(),
        "Invalid evaluated value of resized domain extent: ",
        out_id->toString());
    auto extent_int = extent_val.as<int64_t>();
    NVF_ERROR(
        extent_int >= 0,
        "Invalid resized domain extent ",
        extent_int,
        " for domain ",
        out_id->toString());

    auto iter_type =
        extent_int == 1 ? IterType::Broadcast : IterType::Iteration;

    resize_itertypes_.emplace_back(id_index, iter_type);
  }
}

void DynamicTransformConcretizationInfo::analyzeExpands(
    ExpressionEvaluator* expr_eval) {
  const std::vector<TensorView*>& expanded_tvs =
      initial_info_->getDynamicExpandedTensorViews();
  for (const auto tv_index : arange((int64_t)expanded_tvs.size())) {
    const TensorView* out_tv = expanded_tvs.at(tv_index);
    const TensorView* inp_tv = out_tv->definition()->as<ExpandOp>()->in();

    const std::vector<IterDomain*>& out_root = out_tv->getMaybeRootDomain();
    const std::vector<IterDomain*> inp_logical =
        TensorDomain::noReductions(inp_tv->getLogicalDomain());

    NVF_ERROR(out_root.size() == inp_logical.size());
    std::vector<bool> expand_axes;
    expand_axes.reserve(out_root.size());
    for (int64_t i : arange((int64_t)out_root.size())) {
      const IterDomain* inp_id = inp_logical[i];
      const IterDomain* out_id = out_root[i];
      if (out_id->isIteration()) {
        expand_axes.push_back(false);
        continue;
      }
      // For Broadcast or Symbolic axes, check the sizes of the input and output
      int64_t out_size = expr_eval->evaluate(out_id->extent()).as<int64_t>();
      // Use getMaybeExpandedExtent() here so we can mark "false" if we are just
      // preserving a pre-existing expansion.
      int64_t in_size =
          expr_eval->evaluate(inp_id->getMaybeExpandedExtent()).as<int64_t>();
      if (in_size == 1) {
        expand_axes.push_back(out_size != in_size);
      } else {
        NVF_CHECK(
            out_size == in_size,
            "Mismatch in sizes when concretizing expand. Expanded or Iteration "
            "domain ",
            inp_id->toString(),
            " has possibly expanded extent ",
            in_size,
            " which is incompatible with expansion to size ",
            out_size,
            ". Note that already-expanded axes may not themselves be "
            "expanded.");
        expand_axes.push_back(false);
      }
    }
    expand_axes_.emplace_back(tv_index, expand_axes);
  }
}

void DynamicTransformConcretizationInfo::analyzeFactoryOutputs(
    ExpressionEvaluator* expr_eval) {
  const std::vector<TensorView*>& factory_tvs =
      initial_info_->getDynamicFactoryOutputs();
  factory_output_itertypes_.reserve(factory_tvs.size());
  for (const auto tv_index : arange((int64_t)factory_tvs.size())) {
    const TensorView* tv = factory_tvs.at(tv_index);
    const std::vector<IterDomain*>& logical_dom = tv->getLogicalDomain();
    std::vector<std::pair<int64_t, IterType>> conc_iter_types;
    for (int64_t pos : arange((int64_t)logical_dom.size())) {
      const IterDomain* id = logical_dom[pos];
      if (!id->isSymbolic()) {
        continue;
      }
      PolymorphicValue extent = expr_eval->evaluate(id->extent());
      NVF_CHECK(
          extent.hasValue(),
          "Could not evaluate dynamic factory op output extent ",
          id->extent());
      NVF_ERROR(
          extent.is<int64_t>(),
          "Expected integer evaluated extent but found ",
          extent);
      IterType iter_type = (extent.as<int64_t>() == 1) ? IterType::Broadcast
                                                       : IterType::Iteration;
      conc_iter_types.emplace_back(pos, iter_type);
    }
    factory_output_itertypes_.push_back(conc_iter_types);
  }
}

void DynamicTransformConcretizationInfo::analyzeTopK(
    ExpressionEvaluator* expr_eval) {
  const auto& topk_tvs = initial_info_->getDynamicTopKTensorViews();

  for (const auto [i, tv] : enumerate(topk_tvs)) {
    auto topk_op = dynamic_cast<TopKOp*>(tv->definition());
    NVF_ERROR(topk_op != nullptr, "Expected TopKOp for TopK TensorView");

    // Evaluate K parameter
    auto k_val = expr_eval->evaluate(topk_op->k());
    NVF_ERROR(k_val.hasValue(), "Could not evaluate K parameter for TopK");

    auto k_int = k_val.as<int64_t>();
    NVF_ERROR(
        k_int >= 0,
        "Invalid TopK K parameter ",
        k_int,
        " for operation ",
        topk_op->toString());
    auto iter_type = (k_int == 1) ? IterType::Broadcast : IterType::Iteration;

    topk_itertypes_.emplace_back(i, iter_type);
  }
}

bool DynamicTransformConcretizationInfo::operator==(
    const DynamicTransformConcretizationInfo& other) const {
  if (this == &other) {
    return true;
  }

  if (reshape_transforms_.size() != other.reshape_transforms_.size() ||
      resize_itertypes_.size() != other.resize_itertypes_.size() ||
      empty_extents_.size() != other.empty_extents_.size() ||
      factory_output_itertypes_.size() !=
          other.factory_output_itertypes_.size() ||
      topk_itertypes_.size() != other.topk_itertypes_.size()) {
    return false;
  }

  for (const auto i : arange((int64_t)reshape_transforms_.size())) {
    const auto& analysis = reshape_transforms_.at(i);
    const auto& other_analysis = other.reshape_transforms_.at(i);
    if (analysis != other_analysis) {
      return false;
    }
  }

  for (const auto i : arange((int64_t)resize_itertypes_.size())) {
    const auto& itertype = resize_itertypes_.at(i);
    const auto& other_itertype = other.resize_itertypes_.at(i);
    if (itertype != other_itertype) {
      return false;
    }
  }

  if (factory_output_itertypes_ != other.factory_output_itertypes_) {
    return false;
  }

  for (const auto [topk_itertype, other_topk_itertype] :
       zip(topk_itertypes_, other.topk_itertypes_)) {
    if (topk_itertype != other_topk_itertype) {
      return false;
    }
  }

  for (const auto i : arange((int64_t)expand_axes_.size())) {
    const auto& expand_axes = expand_axes_.at(i);
    const auto& other_expand_axes = other.expand_axes_.at(i);
    if (expand_axes != other_expand_axes) {
      return false;
    }
  }

  for (const auto i : arange((int64_t)empty_extents_.size())) {
    const auto& ee = empty_extents_.at(i);
    const auto& other_ee = other.empty_extents_.at(i);
    if (ee != other_ee) {
      return false;
    }
  }

  return true;
}

std::string DynamicTransformConcretizationInfo::toString() const {
  std::stringstream ss;
  ss << "DynamicTransformConcretizationInfo\n";
  indent(ss, 1) << "Empty tensor extents:\n";
  for (const auto& i : empty_extents_) {
    auto ext = initial_info_->getMaybeZeroExtents().at(i);
    indent(ss, 2) << ext->toString() << " is zero\n";
  }
  indent(ss, 1) << "Reshape:\n";
  NVF_ERROR(
      reshape_transforms_.size() ==
      initial_info_->getDynamicReshapedTensorViews().size());
  for (const auto& [tv_index, view_info] : reshape_transforms_) {
    auto tv = initial_info_->getDynamicReshapedTensorViews().at(tv_index);
    if (std::holds_alternative<AnalyzeViewResult>(view_info)) {
      indent(ss, 2) << tv->toString() << " (index=" << tv_index << "), "
                    << std::get<AnalyzeViewResult>(view_info).toString()
                    << "\n";
    } else {
      indent(ss, 2) << tv->toString() << " (index=" << tv_index
                    << "), is empty. Symbolic reshape sizes: "
                    << std::get<std::vector<int64_t>>(view_info) << "\n";
    }
  }
  indent(ss, 1) << "Resize:\n";
  NVF_ERROR(
      resize_itertypes_.size() ==
      initial_info_->getDynamicResizedIterDomains().size());
  for (const auto& [id_index, iter_type] : resize_itertypes_) {
    auto id = initial_info_->getDynamicResizedIterDomains().at(id_index);
    indent(ss, 2) << id->toString() << " (index=" << id_index << "), "
                  << iter_type << "\n";
  }
  indent(ss, 1) << "Expand:\n";
  NVF_ERROR(
      expand_axes_.size() ==
      initial_info_->getDynamicExpandedTensorViews().size());
  for (const auto& [tv_index, expand_axes] : expand_axes_) {
    auto tv = initial_info_->getDynamicExpandedTensorViews().at(tv_index);
    indent(ss, 2) << tv->toString() << " (index=" << tv_index << "), {";
    bool first = true;
    for (bool e : expand_axes) {
      if (!first) {
        ss << ", ";
      }
      first = false;
      ss << (e ? "true" : "false");
    }
    ss << "}\n";
  }
  indent(ss, 1) << "Factory Output IterTypes:\n";
  NVF_ERROR(
      factory_output_itertypes_.size() ==
      initial_info_->getDynamicFactoryOutputs().size());
  for (int64_t i : arange((int64_t)factory_output_itertypes_.size())) {
    TensorView* tv = initial_info_->getDynamicFactoryOutputs().at(i);
    indent(ss, 2) << tv->toString() << std::endl;
    for (const auto& [pos, iter_type] : factory_output_itertypes_.at(i)) {
      indent(ss, 3) << tv->getLogicalDomain().at(pos)->toString() << " => "
                    << iter_type << std::endl;
    }
  }
  indent(ss, 1) << "TopK:\n";
  NVF_ERROR(
      topk_itertypes_.size() ==
      initial_info_->getDynamicTopKTensorViews().size());
  for (const auto& [tv_index, iter_type] : topk_itertypes_) {
    auto tv = initial_info_->getDynamicTopKTensorViews().at(tv_index);
    indent(ss, 2) << tv->toString() << " (index=" << tv_index << "), "
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
    NVF_ERROR(
        fusion == info->fusion(),
        "Invalid DynamicTransformInitialInfo. The associated Fusion is "
        "different from the given Fusion");
    FusionGuard fg(fusion);
    concretize();
  }

  //! Return map from original symbolic value to new concrete value.
  std::unordered_map<Val*, Val*> getSymbolicToConcretizedMap() {
    return symbolic_to_concretized_map_;
  }

 private:
  void concretize();

  //! Concretize a single reshape which has a non-empty input tensor
  TensorView* concretizeNonEmptyReshape(
      TensorView* inp_tv,
      TensorView* incomplete_out_tv,
      const AnalyzeViewResult& view_analysis);

  //! Concretize a single reshape given that we know that numel=0.
  //! The symbolic sizes are the actual sizes 0 or 1, or -1 if the size of a
  //! given reshaped dimension is greater than 1.
  TensorView* concretizeEmptyReshape(
      TensorView* inp_tv,
      TensorView* incomplete_out_tv,
      const std::vector<int64_t>& symbolic_sizes);

  void concretizeReshape();

  void concretizeResize();

  void concretizeExpand();

  void concretizeEmptyExtents();

  void concretizeFactoryOutputs();

  void concretizeTopK();

  //! Use this instead of calling registerMutation directly, since it will also
  //! check that the concretized value is a valid input to all of its uses.
  void registerConcretization(Val* old_val, Val* new_val) {
    symbolic_to_concretized_map_.emplace(old_val, new_val);
    checkConcretizedUses(old_val, new_val);
    NVF_ERROR(
        old_val->dtype() == new_val->dtype(),
        "registerConcretization should not be used to change dtype of Val ",
        old_val->toString(),
        ". Old dtype: ",
        old_val->dtype(),
        ". New dtype: ",
        new_val->dtype());
    registerMutation(old_val, new_val);
  }

  //! Check uses of old_val to ensure that new_val does not violate
  //! assumptions. This is currently only used to check that inputs to SqueezeOp
  //! are marked broadcast during concretization.
  void checkConcretizedUses(Val* old_val, Val* new_val) const;

  using OptOutMutator::mutate;

  void mutate(TensorView* tv) final;

  void mutate(TensorDomain* td) final;

  void mutate(IterDomain* id) final;

  void mutate(Expr* expr) final;

  //! Concretizes the root domain of a symbolic consumer tensor from
  //! its producer domains. Returns true if any root ID is concretized.
  bool propagateFromProducerToConsumer(TensorView* consumer);

 private:
  const DynamicTransformConcretizationInfo* info_;

  //! Map from original symbolic value to new concretized_value.
  //! This map is separate from mutation_ to avoid interfering with
  //! OptOutMutator
  std::unordered_map<Val*, Val*> symbolic_to_concretized_map_;
};

void DynamicTransformConcretizer::concretize() {
  // Concretize all dynamic reshape ops
  concretizeReshape();

  // Set output IterTypes for dynamic resize ops
  concretizeResize();

  // Overwrite expanded IterDomains for dynamic expand ops
  concretizeExpand();

  // Registers replacement of all empty extents with zeroVal()
  concretizeEmptyExtents();

  // Set IterTypes for factory op outputs
  concretizeFactoryOutputs();

  // Set IterTypes for TopK op outputs
  concretizeTopK();

  // Finally, propagate concretized domains
  auto all_stmts = StmtSort::getStmts(
      info_->fusion(),
      /*traverse_members*/ true,
      /*traverse_attributes*/ true,
      /*traverse_siblings*/ true);
  for (auto stmt : all_stmts) {
    // We mutate all scalar and TensorView Vals and Exprs in topological order.
    // This alone is enough to modify scalars and their Exprs properly. Above,
    // concretizeEmptyExtents will register some scalar extents as zeroVal(),
    // and those will be properly propagated in this type of traversal.
    //
    // However, an important part of concretization is mutating IterDomains and
    // IterDomain expressions. To do this, we have registered some mutations in
    // the above calls; for example concretizeResize registers IterTypes as
    // Iteration or Broadcast. There may still be many Symbolic IterDomains that
    // are not yet registered for mutation though; these need to be registered
    // by propagating the IterTypes and modified extents through the Fusion in
    // the P2C direction. Importantly, this means traversing across TensorView
    // expressions and using exact mapping between producer and consumer TVs to
    // infer IterTypes of consumer root IterDomains from concretized producer
    // logical IterDomains.
    //
    // The order of StmtSort::getStmts guarantees a topological ordering with
    // respect to our IR graph. That IR does not explicitly hold TensorView
    // dependencies for IterDomains; i.e. we rely on TensorView expressions to
    // infer that one IterDomain is a producer logical which another consumer
    // root domain depends on. So we avoid processing IterDomains and
    // TensorDomains until we reach the TensorView that contains them. Otherwise
    // we would not be able to propagate across exact maps before processing
    // all root->logical IterDomains and expressions.
    if (const auto op = dynamic_cast<Expr*>(stmt); stmt->isA<IterDomain>() ||
        stmt->isA<TensorDomain>() || (op && op->output(0)->isA<IterDomain>())) {
      continue;
    }
    OptOutMutator::dispatchMutate(stmt);
  }

  for (Val* outp : info_->fusion()->outputs()) {
    Val* new_outp = maybeMutated(outp);
    if (new_outp != outp) {
      info_->fusion()->replaceOutput(outp, new_outp);
    }
  }
}

void DynamicTransformConcretizer::concretizeEmptyExtents() {
  auto fusion = FusionGuard::getCurFusion();
  for (const auto& ext_index : info_->getEmptyExtents()) {
    auto ext = info_->initialInfo()->getMaybeZeroExtents().at(ext_index);
    auto zero = fusion->zeroVal(ext->getDataType().value());
    auto uses = ext->uses();
    for (auto use : uses) {
      ir_utils::replaceValInExprInputs(use, ext, zero);
    }
    // Register the concretization of this scalar, which allows us to replace it
    // whenever it is used as an extent member of an IterDomain.
    //
    // When we ext in all uses above, it affects downstream expressions. For
    // example we might replace i0 with 0 in (i0 + i1) + i2 to form (0 + i1) +
    // i2. However, i0 itself might be used as the extent, start, or stop values
    // in an IterDomain, so we register the concretization here so that we can
    // replace these values whenever we encounter them.
    registerConcretization(ext, zero);
  }
}

TensorView* DynamicTransformConcretizer::concretizeNonEmptyReshape(
    TensorView* inp_tv,
    TensorView* incomplete_out_tv,
    const AnalyzeViewResult& view_analysis) {
  TensorView* concrete_reshape_out_tv = reshape(inp_tv, view_analysis);
  // Inherit the mesh from the original output TV instead of the input TV.  If
  // the original output TV doesn't have a mesh, it's subject to sharding
  // propagation so we should assign the new output TV an empty mesh.
  // Otherwise, the original output TV has a user-specified sharding, which
  // TransformReplay::selfReplay will clone (cf. #3950), and we should assign
  // the output TV the same mesh.
  concrete_reshape_out_tv->setDeviceMesh(incomplete_out_tv->getDeviceMesh());

  // Extent expressions often change when concretizing a reshape. Here we
  // replace these in all downstream expressions so that the Fusion looks just
  // like it would have if we had used a static reshape instead.
  //
  // Note that Reduction IterDomains might be present in the concretized
  // reshape. For example, suppose we are given the following dynamic Fusion
  //
  //   Inputs:
  //     T0
  //   Outputs:
  //     T3
  //   T1[ iS2{i0} rS3{i1} ] = sum(T0[ iS0{i0} iS1{i1} ])
  //   T2[ ?S4{i2} ] = view(T1[ iS2{i0} rS3{i1} ])
  //   T3[ ?S4{i2} ] = -T2[ ?S4{i2} ]
  //
  // Then we will concretize this as
  //
  //   Inputs:
  //     T0
  //   Outputs:
  //     T3
  //   T1[ iS2{i0} rS3{i1} ] = sum(T0[ iS0{i0} iS1{i1} ])
  //   T3[ iS4{i0} ] = -T1[ iS2{i0} rS3{i1} ]
  //
  // Notice here that the ViewOp is gone since we recognized that there is no
  // transformation to perform. Instead, T1 is used directly in place of T2.
  // We also replace the extent i2 from the dynamic reshape output T2 with i0,
  // which is what the code below implements. Since T1 includes a Reduction
  // IterDomain, we must ignore it in order to match ?S4{i2} with iS2{i0}.
  auto old_logical = incomplete_out_tv->getLogicalDomain();
  auto new_logical =
      TensorDomain::noReductions(concrete_reshape_out_tv->getLogicalDomain());
  NVF_ERROR(
      old_logical.size() == new_logical.size(),
      "Concretized reshape logical size does not match symbolic logical size");

  TransformReplay::selfReplay(
      incomplete_out_tv->domain(),
      concrete_reshape_out_tv->domain(),
      /*ignore_reductions=*/true);

  for (auto&& [old_id, new_id] : zip(old_logical, new_logical)) {
    Val* old_extent = old_id->extent();
    Val* new_extent = new_id->extent();
    // If the old extent did not have a definition, we don't need to replace
    // it, since it will get bound whenever this tensor is a segmentation
    // edge.
    //
    // Also, if the old extent is already a constant, don't replace it with a
    // non-constant, since this could cause downstream extents to become
    // non-constant. See https://github.com/NVIDIA/Fuser/issues/1572
    if (old_extent->definition() && !new_extent->sameAs(old_extent) &&
        (!old_extent->isConstScalar() || new_extent->isConstScalar())) {
      registerConcretization(old_extent, new_extent);
    }
  }

  return concrete_reshape_out_tv;
}

TensorView* DynamicTransformConcretizer::concretizeEmptyReshape(
    TensorView* inp_tv,
    TensorView* incomplete_out_tv,
    const std::vector<int64_t>& symbolic_sizes) {
  std::vector<Val*> new_shape;
  const std::vector<IterDomain*>& old_logical =
      incomplete_out_tv->getLogicalDomain();
  NVF_ERROR(symbolic_sizes.size() == old_logical.size());
  new_shape.reserve(incomplete_out_tv->getLogicalDomain().size());
  for (size_t i : arange(old_logical.size())) {
    int64_t symbolic_size = symbolic_sizes[i];
    if (symbolic_size == 0l) {
      new_shape.push_back(inp_tv->fusion()->zeroVal(DataType::Index));
    } else if (symbolic_size == 1l) {
      new_shape.push_back(inp_tv->fusion()->oneVal(DataType::Index));
    } else {
      NVF_ERROR(symbolic_size == -1l);
      IterDomain* id = incomplete_out_tv->getLogicalDomain().at(i);
      new_shape.push_back(id->extent());
    }
  }
  TensorView* concrete_reshape_out_tv = full(
      new_shape, inp_tv->fusion()->zeroVal(inp_tv->dtype()), inp_tv->dtype());

  const std::vector<IterDomain*>& new_logical =
      concrete_reshape_out_tv->getLogicalDomain();
  NVF_ERROR(symbolic_sizes.size() == new_logical.size());
  for (size_t i : arange(symbolic_sizes.size())) {
    int64_t symbolic_size = symbolic_sizes[i];
    IterType iter_type =
        symbolic_size == 1l ? IterType::Broadcast : IterType::Iteration;
    IterDomain* new_id = new_logical[i];
    Val* extent = symbolic_size == 0l
        ? new_id->fusion()->zeroVal(DataType::Index)
        : maybeMutated(new_id->extent());
    registerConcretization(old_logical[i]->extent(), extent);
    if (!new_id->isSymbolic()) {
      NVF_ERROR(new_id->getIterType() == iter_type);
      continue;
    }
    // Concretize Symbolic IterDomains that were just created
    registerConcretization(
        new_id,
        IterDomainBuilder(new_id).extent(extent).iter_type(iter_type).build());
  }

  return concrete_reshape_out_tv;
}

void DynamicTransformConcretizer::concretizeReshape() {
  // Concretize each reshape op.
  for (const auto& [tv_index, view_info] : info_->getReshapeTransforms()) {
    auto incomplete_out_tv =
        info_->initialInfo()->getDynamicReshapedTensorViews().at(tv_index);
    auto view_op = incomplete_out_tv->definition()->as<ViewOp>();
    auto inp_tv = view_op->in()->as<TensorView>();

    TensorView* concrete_reshape_out_tv = nullptr;
    if (std::holds_alternative<AnalyzeViewResult>(view_info)) {
      concrete_reshape_out_tv = concretizeNonEmptyReshape(
          inp_tv, incomplete_out_tv, std::get<AnalyzeViewResult>(view_info));
    } else {
      concrete_reshape_out_tv = concretizeEmptyReshape(
          inp_tv, incomplete_out_tv, std::get<std::vector<int64_t>>(view_info));
    }

    // NOTE: The replacement might not yet actually be valid. For example, if
    // inp_tv contains Symbolic domains that need to be squeezed, this check
    // would fail at this point. So we skip checkConcretizedUses here and
    // perform it later in mutate(TensorView*).

    symbolic_to_concretized_map_.emplace(
        incomplete_out_tv, concrete_reshape_out_tv);

    ir_utils::replaceValInAllExprInputsAndFusionOutputs(
        incomplete_out_tv, concrete_reshape_out_tv);

    info_->fusion()->removeVal(incomplete_out_tv);
  }
}

void DynamicTransformConcretizer::concretizeResize() {
  // Concretize each resize op.
  for (const auto& [id_index, iter_type] : info_->getResizeIterTypes()) {
    auto id = info_->initialInfo()->getDynamicResizedIterDomains().at(id_index);
    NVF_CHECK(
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

void DynamicTransformConcretizer::concretizeExpand() {
  // Concretize each expand op.
  for (const auto& [tv_index, axis_is_expanded] : info_->getExpandAxes()) {
    TensorView* symbolic_out_tv =
        info_->initialInfo()->getDynamicExpandedTensorViews().at(tv_index);

    // If no axis is expanded, replace this op with a set()
    if (std::none_of(
            axis_is_expanded.begin(), axis_is_expanded.end(), [](bool b) {
              return b;
            })) {
      TensorView* inp_tv =
          symbolic_out_tv->definition()->input(0)->as<TensorView>();
      TensorView* concretized_tv = set(inp_tv);

      symbolic_to_concretized_map_.emplace(symbolic_out_tv, concretized_tv);

      ir_utils::replaceValInAllExprInputsAndFusionOutputs(
          symbolic_out_tv, concretized_tv);
    }

    // We do not need to replace the ExpandOp, but we do need to mutate all of
    // the IterDomains in the output based on whether each was expanded
    std::vector<IterDomain*> out_logical =
        TensorDomain::noReductions(symbolic_out_tv->getLogicalDomain());
    NVF_ERROR(axis_is_expanded.size() == out_logical.size());
    for (int64_t i : arange((int64_t)out_logical.size())) {
      if (!axis_is_expanded[i]) {
        // Propagate as usual for non-expanded IterDomains
        continue;
      }
      // An expanded IterDomain needs to have an extent of oneVal() and a
      // non-null expandedExtent. However, a Symbolic IterDomain will only have
      // an extent. Here we set the IterType to Broadcast and swap the extent to
      // expandedExtent.
      IterDomain* symbolic_id = out_logical[i];
      Val* one = FusionGuard::getCurFusion()->oneVal(DataType::Index);
      IterDomain* concretized_id = IterDomainBuilder(symbolic_id)
                                       .iter_type(IterType::Broadcast)
                                       .extent(one)
                                       .expanded_extent(symbolic_id->extent())
                                       .build();
      registerConcretization(symbolic_id, concretized_id);
    }
  }
}

void DynamicTransformConcretizer::concretizeFactoryOutputs() {
  const std::vector<TensorView*>& factory_tvs =
      info_->initialInfo()->getDynamicFactoryOutputs();
  const auto& pair_vecs = info_->getFactoryOutputIterTypes();
  NVF_ERROR(factory_tvs.size() == pair_vecs.size());
  for (const int64_t i : arange((int64_t)factory_tvs.size())) {
    TensorView* tv = factory_tvs[i];
    const std::vector<std::pair<int64_t, IterType>>& pair_vec = pair_vecs[i];
    for (auto& [pos, iter_type] : pair_vec) {
      auto* old_id =
          maybeMutated(tv->getLogicalDomain().at(pos))->as<IterDomain>();
      NVF_ERROR(
          old_id->definition() == nullptr,
          "Symbolic factory output has ID definition that would be discarded");
      auto* new_id = IterDomainBuilder(old_id).iter_type(iter_type).build();
      registerConcretization(old_id, new_id);
    }
    mutate(tv->domain());
    OptOutMutator::mutate(tv);
  }
}

void DynamicTransformConcretizer::concretizeTopK() {
  const auto& topk_itertypes = info_->getTopKIterTypes();

  for (const auto& [tv_index, iter_type] : topk_itertypes) {
    auto tv = info_->initialInfo()->getDynamicTopKTensorViews().at(tv_index);
    auto topk_op = dynamic_cast<TopKOp*>(tv->definition());
    NVF_ERROR(topk_op != nullptr, "Expected TopKOp for TopK TensorView");

    const auto topk_dim = topk_op->dim();
    NVF_ERROR(
        topk_dim >= 0 && topk_dim < std::ssize(tv->getLogicalDomain()),
        "Invalid TopK dimension ",
        topk_dim);

    // Concretize the TopK dimension for values output
    auto values_logical = tv->getLogicalDomain();
    auto topk_id = values_logical.at(topk_dim);

    // Just a sanity check. This should be still symbolic.
    NVF_ERROR(topk_id->isSymbolic());
    auto new_id = IterDomainBuilder(topk_id).iter_type(iter_type).build();
    registerConcretization(topk_id, new_id);

    // Concretize the TopK dimension for indices output
    auto indices_tv = topk_op->outIndices()->as<TensorView>();
    auto indices_logical = indices_tv->getLogicalDomain();
    auto indices_topk_id = indices_logical.at(topk_dim);

    NVF_ERROR(indices_topk_id->isSymbolic());
    auto new_indices_id =
        IterDomainBuilder(indices_topk_id).iter_type(iter_type).build();
    registerConcretization(indices_topk_id, new_indices_id);
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
  for (auto root_id : tv->getMaybeRootDomain()) {
    // This will register root_id for mutation if its extent, start, or
    // stop_offset is registered for mutation
    mutate(root_id);
  }

  // First, try to concretize the root domain as there may be symbolic
  // axes inherited from the producers
  propagateFromProducerToConsumer(tv);

  // If no root domain is altered by producer, we don't need to propagate back
  // up to logical domain. We could return early, but instead we go ahead and
  // check the root to logical transforms to be sure we have concretized any
  // intermediate IterDomains.

  // If it has an root domain, the IterTypes of the logical
  // IDs may need to be updated as well. Traverse the rfactor exprs
  // and mutate the IterTypes of output IDs if symbolic.
  if (tv->hasRoot()) {
    // Note that it is assumed that theres's no further expression
    // beyond the logical domain as asserted above
    auto all_id_exprs = StmtSort::getExprsBetween(
        {tv->getRootDomain().begin(), tv->getRootDomain().end()},
        {tv->getLogicalDomain().begin(), tv->getLogicalDomain().end()});
    for (auto expr : all_id_exprs) {
      // Assume outputs of IterDomain exprs are always IterDomains. If
      // the assumption is invalidated, the logic here would need to
      // be updated. Assert the assumption to immediately detect such
      // a case if happened.
      for (auto out_val : expr->outputs()) {
        NVF_ERROR(
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
      const auto input_ids =
          ir_utils::filterByType<IterDomain>(expr->inputs()).vector();
      for (auto i : arange((int64_t)input_ids.size())) {
        auto inp_id = input_ids.at(i);
        auto updated_id = maybeMutated(inp_id)->as<IterDomain>();
        NVF_CHECK(
            updated_id == inp_id || !updated_id->isSymbolic(),
            "Mutated IterDomains between root and logical should not be "
            "symbolic");
        if (i == 0) {
          // ops::promoteIterType will favor Symbolic if it encounters it
          // alongside Broadcast. This is preferable at fusion definition, but
          // here we are propagating, and if we only see Broadcast in some
          // dimension, then we should not retain Symbolic. To work around this,
          // we always overwrite Symbolic with the first concrete IterType we
          // encounter.
          iter_type = updated_id->getIterType();
        } else {
          iter_type =
              ops::promoteIterType(iter_type, updated_id->getIterType());
        }
      }
      // Update the IterType of each output
      for (auto out_id : ir_utils::filterByType<IterDomain>(expr->outputs())) {
        auto mut_id = maybeMutated(out_id)->as<IterDomain>();
        if (!mut_id->isSymbolic()) {
          // We are only concretizing IterType here, so if we have already
          // concretized the iter_type for this ID, we can skip this.
          continue;
        }

        // If out_id is Symbolic, we need to concretize it here. If we did not
        // yet determine its IterType, then we've missed our chance.
        NVF_ERROR(
            iter_type != IterType::Symbolic,
            "Failed to concretize an output IterType for expression: ",
            expr->toString());

        auto concretized_out_id =
            IterDomainBuilder(mut_id).iter_type(iter_type).build();
        registerConcretization(out_id, concretized_out_id);
      }

      // At this point, we might have registered both the input and output
      // IterDomains for concretization, specifying either new extents or
      // IterTypes or both. In this context, we want to create a new Expr
      // having the mutations to both inputs and outputs, so that the
      // concretized IterDomains are connected by the right type of expression.
      //
      // Mutating outputs has the effect of redefining the new outputs. That can
      // be unwanted in the general case since we often mutate a Fusion with the
      // intention of to changing the definition. For this reason,
      // OptOutMutator::mutate will only mutate inputs by default. The
      // mutateExprOutputsOnly call below performs this redefinition which we
      // desire in the current context.
      //
      // Each mutate step below removes the expression so we need to apply the
      // second step to the replacement Expr. The order of these calls is
      // unimportant, except that mutate(Expr*) does not return the replacement
      // Expr*, whereas mutateExprOutputsOnly does.

      // Set expr as the definition for concretized outputs
      expr = mutateExprOutputsOnly(expr);
      // Replace inputs and attributes that were concretized
      mutate(expr);
    }
  }

  // Root and logical domains are updated. First mutate the
  // TensorDomain and then TensorView
  mutate(tv->domain());
  OptOutMutator::mutate(tv);
  // Check concretization is valid after we've done the replacement. See note
  // about squeeze inside concretizeReshape above.
  checkConcretizedUses(tv, tv);
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

  std::vector<IterDomain*> root_dom =
      td->hasRoot() ? updateIdVec(td->root()) : std::vector<IterDomain*>();
  std::vector<IterDomain*> logical_dom = updateIdVec(td->logical());
  std::vector<IterDomain*> loop_domain = updateIdVec(td->loop());
  std::vector<IterDomain*> alloc_dom = td->hasAllocation()
      ? updateIdVec(td->allocation())
      : std::vector<IterDomain*>();

  if (!mutated) {
    return;
  }

  // Update the contiguity vector. Drop the contig val if mutated to broadcast
  auto contig = td->contiguity();

  const auto& new_maybe_alloc = td->hasAllocation() ? alloc_dom : logical_dom;
  const auto& original_alloc = td->maybeAllocation();
  NVF_ERROR(
      new_maybe_alloc.size() == original_alloc.size(),
      "rank of allocation domain shouldn't change in concretization");

  for (const auto i : arange((int64_t)original_alloc.size())) {
    auto original_id = original_alloc.at(i);
    if (original_id->getIterType() != IterType::Symbolic) {
      continue;
    }

    NVF_ERROR(
        contig.at(i),
        "Unexpected to have a non-contig symbolic domain: ",
        original_id->toString());

    auto updated_id = new_maybe_alloc.at(i);

    // If the concretized ID is a broadcast domain, drop the contig val
    if (updated_id->isBroadcast()) {
      contig.at(i) = std::nullopt;
    }
  }

  Val* mutated_val = IrBuilder::createInContainer<TensorDomain>(
      td->container(), root_dom, logical_dom, alloc_dom, loop_domain, contig);
  registerConcretization(td, mutated_val);
}

void DynamicTransformConcretizer::mutate(IterDomain* id) {
  OptOutMutator::mutate(id);
  // Check whether the extent was mutated to zero. If so, ensure that the
  // IterType is set to Iteration
  auto* mut_id = maybeMutated(id)->as<IterDomain>();
  if (mut_id->extent()->isZeroInt()) {
    IterDomain* new_mut_id =
        IterDomainBuilder(mut_id).iter_type(IterType::Iteration).build();
    registerConcretization(id, new_mut_id);
    registerConcretization(mut_id, new_mut_id);
  }
}

//! Returns whether a reduction has any trivial partial reductions. Modifies
//! reduction_axes in place to insert indices of non-trivial reduction axes,
//! relative to squeezed input.
static bool hasTrivialReduction(
    TensorView* in,
    TensorView* out,
    std::vector<int64_t>& reduction_axes) {
  bool has_trivial_reduction = false;
  PairwiseLogicalDomainMap p2c_map(in, out);
  // We need to map broadcasts in order to detect reductions of broadcasts
  p2c_map.mapBroadcast(true);
  auto p2c = p2c_map.mapProducerToConsumer();
  int64_t pos = -1;
  for (IterDomain* in_id : TensorDomain::noReductions(in->getLogicalDomain())) {
    ++pos;
    auto out_it = p2c.find(in_id);
    if (out_it == p2c.end()) {
      continue;
    }
    IterDomain* out_id = out_it->second;
    if (out_id->isReduction()) {
      reduction_axes.push_back(pos);
      if (in_id->isBroadcast() && !in_id->hasExpandedExtent()) {
        has_trivial_reduction = true;
      }
    }
  }
  return has_trivial_reduction;
}

// Maybe insert SqueezeOps on inputs of ReductionOp, to simplify trivial
// reductions.
void DynamicTransformConcretizer::mutate(Expr* expr) {
  if (ReductionOp* rop = dynamic_cast<ReductionOp*>(expr); rop) {
    auto* in = rop->in()->as<TensorView>();
    auto* orig_out = rop->out()->as<TensorView>();
    std::vector<int64_t> reduction_axes;
    if (hasTrivialReduction(in, orig_out, reduction_axes)) {
      // There is at least one trivial reduction that should be squeezed. Use
      // binaryOp to ensure this is done exactly as it is in a non-dynamic
      // fusion
      //
      // Note that keepdim=false always here, since that results in downstream
      // broadcasts which will already have been inserted.
      TensorView* new_out = reductionOp(
          rop->getReductionOpType(),
          reduction_axes,
          rop->init(),
          in,
          /*keep_dim=*/false,
          orig_out->dtype());
      registerConcretization(orig_out, new_out);
    }
  } else if (WelfordOp* wop = dynamic_cast<WelfordOp*>(expr); wop) {
    auto in = wop->in()->as<TensorView>();
    auto orig_avg = wop->outAvg()->as<TensorView>();

    std::vector<int64_t> reduction_axes;
    if (hasTrivialReduction(in, orig_avg, reduction_axes)) {
      // Use Welford to ensure this is done exactly as it is in a non-dynamic
      // fusion
      WelfordResult new_result = Welford(
          in,
          reduction_axes,
          // For avg and variance to be default initialized, they should be
          // given as nullptr. In that case, this constructor actually sets them
          // as a scalar 0. Here we use that to detect whether they are
          // initialized or not.
          dynamic_cast<TensorView*>(wop->initAvg()),
          dynamic_cast<TensorView*>(wop->initVar()),
          wop->initN());
      registerConcretization(orig_avg, new_result.avg);
      registerConcretization(wop->outVar(), new_result.var_sum);
      registerConcretization(wop->outN(), new_result.n);
    }
  }
  OptOutMutator::mutate(expr);
}

bool DynamicTransformConcretizer::propagateFromProducerToConsumer(
    TensorView* consumer) {
  if (consumer->definition() == nullptr ||
      !consumer->domain()->hasSymbolicAxis()) {
    return false;
  }

  const auto& root_domain = consumer->getMaybeRootDomain();

  auto def = consumer->definition();

  // We will loop over IterDomains in the consumer root. For each, we need to
  // inspect the consumer to producer map to all producers. Instead of
  // recomputing these for each root IterDomain, we precompute them for each
  // producer here then re-use them in the following loop.
  std::vector<std::unordered_map<IterDomain*, IterDomain*>> c2p_maps;
  bool is_factory_output = true;
  for (auto producer : ir_utils::filterByType<TensorView>(def->inputs())) {
    PairwiseLogicalDomainMap logical_map(producer, consumer);
    // We map symbolic domains here regardless of whether their extents match.
    // This is safe because we are propagating from a producer which should have
    // already been concretized. The consumer might have a different extent
    // which will be equivalent to (but not necessarily sameAs) the producer's,
    // and we just want to use its IterType to concretize the consumer ID.
    logical_map.mapSymbolic(true);
    c2p_maps.push_back(logical_map.mapConsumerToProducer());
    is_factory_output = false;
  }

  if (is_factory_output) {
    // There is nothing to propagate for factory methods
    return true;
  }

  bool is_concretized = false;

  for (const auto i : arange((int64_t)root_domain.size())) {
    auto root_id = root_domain.at(i);
    if (root_id->getIterType() != IterType::Symbolic) {
      continue;
    }

    // Figure out the right IterType of this consumer root ID from its
    // corresponding producer IDs

    std::optional<IterType> id_type;

    // If a producer ID is an expanded broadcast then the consumer ID is an
    // expanded broadcast unless we encounter a mapped Iteration producer ID,
    // in which case the output IterType will be Iteration.
    bool has_expanded_producer = false;

    bool found = false;
    for (const auto& c2p : c2p_maps) {
      auto p_it = c2p.find(root_id);
      // In some cases, we can exact map to one producer, but not to another.
      // This is the case for index_select, for example, whose first input is
      // the tensor to look up values in and whose second input gives the
      // indices to use for the lookup. In the selected dimension, the first
      // input will not exact map to the output, but the second input will.
      // Here we just require at least one input to map to root_id so that we
      // can propagate an IterType.
      // See https://github.com/NVIDIA/Fuser/issues/1192 for an example
      if (p_it == c2p.end()) {
        continue;
      }
      found = true;
      auto input_id = p_it->second;
      NVF_ERROR(
          input_id == maybeMutated(input_id),
          "Consumer IterDomain ",
          input_id->toString(),
          " is still registered for mutation after traversing to ",
          consumer->toString(),
          ". Replacement is ",
          maybeMutated(input_id)->toString());
      NVF_ERROR(
          input_id->getIterType() != IterType::Symbolic,
          "Producer ID not concretized: ",
          input_id->toString());

      has_expanded_producer =
          has_expanded_producer || input_id->hasExpandedExtent();

      if (id_type.has_value()) {
        id_type = ops::promoteIterType(*id_type, input_id->getIterType());
      } else {
        id_type = input_id->getIterType();
      }
    }

    // Special case: TopK dimensions don't map to producer dimensions
    // If no mapping was found, check if this IterDomain has already been
    // concretized by TopK operations
    if (!found) {
      auto maybe_concretized = maybeMutated(root_id);
      if (maybe_concretized != root_id &&
          !maybe_concretized->as<IterDomain>()->isSymbolic()) {
        // This IterDomain has already been concretized (e.g., by TopK), skip it
        continue;
      }
    }

    NVF_ERROR(
        found,
        "No input ID found to map with output ID: ",
        root_id->toString());

    NVF_ERROR(
        id_type.has_value(),
        "Did not find id_type for consumer root domain ",
        root_id->toString(),
        ". Perhaps consumer def has no inputs. Consumer definition = ",
        def->toString());

    NVF_ERROR(
        id_type != IterType::Symbolic,
        "Failed to concretize ",
        root_id->toString(),
        " of ",
        consumer->toString());

    IterDomain* concretized_id = nullptr;
    if (*id_type == IterType::Broadcast && has_expanded_producer) {
      // Propagate expanded IterDomains by swapping the extent into the expanded
      // extent
      concretized_id =
          IterDomainBuilder(maybeMutated(root_id)->as<IterDomain>())
              .iter_type(*id_type)
              .extent(FusionGuard::getCurFusion()->oneVal(DataType::Index))
              .expanded_extent(
                  maybeMutated(root_id)->as<IterDomain>()->extent())
              .build();
    } else {
      concretized_id =
          IterDomainBuilder(maybeMutated(root_id)->as<IterDomain>())
              .iter_type(*id_type)
              .build();
    }

    registerConcretization(root_id, concretized_id);
    is_concretized = true;
  }

  return is_concretized;
}

DynamicTransformInitialInfo DynamicTransform::getInitialInfo(Fusion* fusion) {
  DynamicTransformInitialInfoBuilder builder(fusion);
  return builder.getInfo();
}

std::unordered_map<Val*, Val*> DynamicTransform::concretizeFusion(
    Fusion* fusion,
    const DynamicTransformConcretizationInfo* info) {
  DynamicTransformConcretizer concretizer(fusion, info);
  return concretizer.getSymbolicToConcretizedMap();
}

std::unordered_map<Val*, Val*> DynamicTransform::concretizeFusion(
    Fusion* fusion,
    const KernelArgumentHolder& args) {
  ExpressionEvaluator expr_eval = executor_utils::bindInputs(args, fusion);
  auto initial_info = getInitialInfo(fusion);
  DynamicTransformConcretizationInfo info(&initial_info, &expr_eval);
  return concretizeFusion(fusion, &info);
}

size_t DynamicTransformConcretizationInfo::hash() const {
  // TODO: This is probably too heavy for the number of concretizations we
  // expect (< 100). We should analyze this and trim out the pieces that are
  // unlikely to change based on real inputs.
  size_t hash = 0;
  for (const auto& [tv, view_info] : getReshapeTransforms()) {
    if (std::holds_alternative<AnalyzeViewResult>(view_info)) {
      hashCombine(hash, std::get<AnalyzeViewResult>(view_info).hash());
    } else {
      for (const int64_t symbolic_size :
           std::get<std::vector<int64_t>>(view_info)) {
        hashCombine(hash, (size_t)symbolic_size);
      }
    }
  }
  for (const auto& extent_idx : getEmptyExtents()) {
    hashCombine(hash, (size_t)extent_idx);
  }
  for (const auto& [id, iter_type] : getResizeIterTypes()) {
    hashCombine(hash, (size_t)iter_type);
  }
  for (const auto& pair_vec : getFactoryOutputIterTypes()) {
    for (const auto& [pos, iter_type] : pair_vec) {
      hashCombine(hash, pos);
      hashCombine(hash, (size_t)iter_type);
    }
  }
  for (const auto& [id, iter_type] : getResizeIterTypes()) {
    hashCombine(hash, (size_t)iter_type);
  }
  for (const auto& [id, expand_axes] : getExpandAxes()) {
    hashCombine(hash, (size_t)id);
    for (bool e : expand_axes) {
      hashCombine(hash, (size_t)e);
    }
  }
  for (const auto& [id, iter_type] : getTopKIterTypes()) {
    hashCombine(hash, (size_t)id);
    hashCombine(hash, (size_t)iter_type);
  }
  return hash;
}

} // namespace nvfuser
