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

    resize_itertypes_.emplace_back(id_index, iter_type);
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

  void mutate(TensorDomain* td) final;
  void mutate(IterDomain* id) final;

 private:
  const DynamicTransformConcretizationInfo* info_;

  //! This map is used during concretization to identify, for a given IterDomain
  //! the set of all IterDomains which are "aligned" with it in some TensorView
  //! expression. This enables us to write mutate(IterDomain*) and propagate
  //! information from producer IterDomains to consumers, which is otherwise not
  //! represented in the graph since we do not connect IterDomains between
  //! TensorViews with expressions.
  std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>
      id_producers_;
};

void DynamicTransformConcretizer::concretize() {
  // First, concretize all dynamic reshape ops
  concretizeReshape();

  // Set output IterTypes for dynamic resize ops
  concretizeResize();

  // The methods above do not traverse the graph. Instead they fill in
  // root->rfactor expressions by replacing the dynamic reshaped TV with a
  // static reshaped one, and by registering concretization of dynamic Resized
  // IterDomains. From this point forward, we will not modify any TensorView
  // expressions. This restriction makes it safe for us to to traverse forward
  // through the graph and mutate IterDomains and TensorDomains in order to
  // properly propagate IterTypes and concretized extent expressions, without
  // breaking the topological ordering of these expressions.
  //
  // When propagating IterTypes across expressions, we need to know the producer
  // IterDomains corresponding to a consumer ID. This mapping helps facilitate
  // this and is used later in mutate(IterDomain*).
  auto all_stmts = StmtSort::getStmts(info_->fusion(), true);
  for (auto expr : ir_utils::filterByType<Expr>(all_stmts)) {
    for (auto consumer : ir_utils::filterByType<TensorView>(expr->outputs())) {
      for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
        PairwiseRootDomainMap root_map(producer, consumer);
        for (auto [cid, pid] : root_map.mapConsumerToProducer(
                 consumer->domain(), producer->domain())) {
          // Initialize set of producer IDs, if we haven't already
          auto& producers = id_producers_.emplace(cid, 0).first->second;
          producers.insert(pid);
        }
      }
    }
  }

  // Finally, propagate concretized domains with forward traversal
  for (auto stmt : all_stmts) {
    mutate(stmt);
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

void DynamicTransformConcretizer::mutate(IterDomain* id) {
  // id might have already been mutated if its definition was updated
  id = maybeMutated(id)->as<IterDomain>();
  if (!id->isSymbolic()) {
    return;
  }
  if (auto def = id->definition()) {
    // Determine concrete IterType based on promotion of inputs to def
    IterType iter_type = IterType::Symbolic;
    for (auto inp_id : ir_utils::filterByType<IterDomain>(def->inputs())) {
      auto updated_id = maybeMutated(inp_id)->as<IterDomain>();
      iter_type = ops::promoteIterType(iter_type, updated_id->getIterType());
    }
    TORCH_INTERNAL_ASSERT(
        iter_type != IterType::Symbolic,
        "Failed to concretize an output IterType for expression: ",
        def->toString());
    auto concretized_id = IterDomainBuilder(id).iter_type(iter_type).build();
    registerConcretization(id, concretized_id);
  } else {
    // IterDomains without definitions might be root domains for the output of a
    // TensorView expression. If so, we should propagate their concretization in
    // the producer to consumer direction.

    auto producers_it = id_producers_.find(id);
    if (producers_it == id_producers_.end()) {
      // id was not a consumer root ID in any TV expression
      return;
    }

    std::optional<IterType> id_type;
    for (auto producer_id : producers_it->second) {
      producer_id = maybeMutated(producer_id)->as<IterDomain>();
      if (id_type.has_value()) {
        id_type = ops::promoteIterType(*id_type, producer_id->getIterType());
      } else {
        id_type = producer_id->getIterType();
      }
    }

    TORCH_INTERNAL_ASSERT(
        id_type.has_value(),
        "Did not find id_type for consumer root domain ",
        id->toString(),
        ". Perhaps consumer def has no inputs.");

    TORCH_INTERNAL_ASSERT(
        id_type != IterType::Symbolic, "Failed to concretize ", id->toString());

    auto concretized_id = IterDomainBuilder(id).iter_type(*id_type).build();

    registerConcretization(id, concretized_id);
  }
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
