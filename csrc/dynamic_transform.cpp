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
#include <ir_cloner.h>
#include <ir_utils.h>
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
  for (const auto tv : dynamic_reshapes_) {
    if (tv) {
      cloned_info.dynamic_reshapes_.push_back(ir_cloner.clone(tv));
    }
  }
  cloned_info.expr_eval_ = expr_eval_.clone(ir_cloner);
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
  return ss.str();
}

//! Gather information about concretizing transformations without
//! concrete input sizes.
class DynamicTransformInitialInfoBuilder : public IterVisitor {
 public:
  DynamicTransformInitialInfoBuilder(Fusion* fusion);

  const auto& getInfo() const {
    return info_;
  }

 private:
  using IterVisitor::handle;

  // Just find views that have symbolic outputs. Do not do replacement
  void handle(ViewOp* op) override {
    auto inp_tv = op->in()->as<TensorView>();
    auto out_tv = op->out()->as<TensorView>();
    // If there's no symbolic axis, this is a static reshape op
    if (out_tv->domain()->hasSymbolicAxis()) {
      info_.dynamic_reshapes_.push_back(op);

      // Input and output extent expressions both affect concretization
      const auto& inp_dom =
          TensorDomain::noReductions(inp_tv->getMaybeRFactorDomain());
      for (const auto id : inp_dom) {
        // Try and evaluate the extent so that intermediate expressions are
        // cached in expr_eval_
        auto ext = info_.expr_eval_.evaluate(id->extent());
        if (!ext.has_value()) {
          leaf_dynamic_vals_.push_back(id->extent());
        }
      }
      const auto& out_dom = out_tv->getMaybeRFactorDomain();
      for (const auto id : out_dom) {
        auto ext = info_.expr_eval_.evaluate(id->extent());
        if (!ext.has_value()) {
          leaf_dynamic_vals_.push_back(id->extent());
        }
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

  std::vector<Val*> leaf_dynamic_vals_;
};

DynamicTransformInitialInfoBuilder::DynamicTransformInitialInfoBuilder(
    Fusion* fusion)
    : info_(fusion) {
  TORCH_INTERNAL_ASSERT(
      !fusion->isA<kir::Kernel>(),
      "Invalid container. Kernel container not allowed.\n");

  traverseTo(fusion, fusion->getTerminatingOutputs(), false, false);

  finalizeDynamicVals();
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

bool DynamicTransformConcretizationInfo::operator==(
    const DynamicTransformConcretizationInfo& other) const {
  if (this == &other) {
    return true;
  }

  if (fusion_ != other.fusion_) {
    return false;
  }

  if (reshape_transforms_.size() != other.reshape_transforms_.size()) {
    return false;
  }

  for (const auto i : c10::irange(reshape_transforms_.size())) {
    const auto& transform = reshape_transforms_.at(i);
    const auto& other_transform = other.reshape_transforms_.at(i);
    if (transform != other_transform) {
      return false;
    }
  }

  return true;
}

DynamicTransformConcretizationInfo DynamicTransformConcretizationInfo::clone(
    IrCloner& ir_cloner) const {
  DynamicTransformConcretizationInfo cloned_info(
      static_cast<Fusion*>(ir_cloner.container()));
  for (auto& pair : reshape_transforms_) {
    cloned_info.reshape_transforms_.emplace_back(
        ir_cloner.clone(pair.first),
        // reshape_transforms_ holds pairs of TensorView* and AnalyzeViewResult
        // AnalyzeViewResult can be copied directly as it holds no references to
        // Statements that would need cloning, only integer indices of axes.
        pair.second);
  }
  return cloned_info;
}

std::string DynamicTransformConcretizationInfo::toString() const {
  std::stringstream ss;
  ss << "DynamicTransformConcretizationInfo\n";
  std::string indent = "  ";
  ss << indent << "Reshape:\n";
  for (const auto& kv : reshape_transforms_) {
    ss << indent << indent << kv.first->toString() << ", "
       << kv.second.toString() << "\n";
  }
  return ss.str();
}

//! Concretize a symbolic fusion with concrete transformation info
class DynamicTransformConcretizer : public OptOutMutator {
 public:
  DynamicTransformConcretizer(
      Fusion* fusion,
      const DynamicTransformConcretizationInfo& info)
      : info_(info) {
    TORCH_INTERNAL_ASSERT(
        fusion == info.fusion(),
        "Invalid DynamicTransformInfo. The associated Fusion is different from the given Fusion");
    concretize();
  }

 private:
  void concretize();

  void concretizeReshape();

  using OptOutMutator::mutate;

  void mutate(TensorView* tv) final;

  void mutate(TensorDomain* td) final;

  //! Concretizes the root domain of a symbolic consumer tensor from
  //! its producer domains. Returns true if any root ID is concretized.
  bool propagateFromProducerToConsumer(TensorView* consumer);

 private:
  const DynamicTransformConcretizationInfo& info_;
  std::unordered_map<IterDomain*, IterDomain*> update_map_;
};

void DynamicTransformConcretizer::concretize() {
  // First, concretize all dynamic reshape ops
  concretizeReshape();

  // Second, propagate concretized domains
  auto all_stmts = StmtSort::getStmts(info_.fusion(), false);
  for (auto stmt : all_stmts) {
    if (stmt->isA<Val>()) {
      mutate(stmt);
    }
  }
}

void DynamicTransformConcretizer::concretizeReshape() {
  // Concretize each reshape op.
  for (const auto& kv : info_.getReshapeTransforms()) {
    auto incomplete_out_tv = kv.first;
    const auto view_analysis = kv.second;

    auto inp_tv = ir_utils::producerTvsOf(incomplete_out_tv).at(0);

    auto concrete_reshape_out_tv = reshape(inp_tv, view_analysis);

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
  auto propagated = propagateFromProducerToConsumer(tv);

  // If no root domain is altered, nothing to do further
  if (!propagated) {
    return;
  }

  // Root IDs are altered. Need to propagate the changes to rfactor
  // domain

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

      // If none of the output IDs is symbolic, nothing to concretize
      if (std::all_of(
              expr->outputs().begin(), expr->outputs().end(), [](Val* output) {
                return output->as<IterDomain>()->getIterType() !=
                    IterType::Symbolic;
              })) {
        continue;
      }
      // If any of output IDs is symbolic, all outputs should be symbolic
      TORCH_INTERNAL_ASSERT(std::all_of(
          expr->outputs().begin(), expr->outputs().end(), [](Val* output) {
            return output->as<IterDomain>()->getIterType() ==
                IterType::Symbolic;
          }));

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
        auto concreteized_out_id =
            IterDomainBuilder(out_id).iter_type(iter_type).build();
        registerMutation(out_id, concreteized_out_id);
      }

      // Outputs are mutated. The expr itself needs to be mutated as
      // well, which can be done by the mutate method
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
  registerMutation(td, mutated_val);
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
        id_type.has_value() && id_type != IterType::Symbolic,
        "Failed to concretize ",
        root_id->toString(),
        " of ",
        consumer->toString());

    auto concretized_id =
        IterDomainBuilder(root_id).iter_type(*id_type).build();

    registerMutation(root_id, concretized_id);
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
  auto expr_eval = info->getExpressionEvaluator();

  // Bind input scalars and tensor metadata to symbolic scalars
  // Here we bind only the inputs that are needed to concretize dynamic
  // transforms.
  TORCH_CHECK(
      args->size() == fusion->inputs().size(),
      "Received ",
      args->size(),
      " inputs but expected ",
      fusion->inputs().size());
  for (auto i : c10::irange(args->size())) {
    const auto& inpi = fusion->inputs()[i];
    const auto argi = (*args)[i];
    if (inpi->isIntegralScalar()) {
      TORCH_CHECK(
          argi->isType(ArgType::Long),
          "Expected integer input at position ",
          i,
          " but found ",
          argTypeToString(argi->type()));

      const int64_t arg_val = *reinterpret_cast<const int64_t*>(argi->arg());
      expr_eval.bind(inpi, arg_val);
    } else if (inpi->isA<TensorView>()) {
      const auto& tv = inpi->as<TensorView>();
      const auto& dom = tv->domain()->maybeRFactor();
      TORCH_CHECK(
          argi->isType(ArgType::Tensor),
          "Expected CUDA tensor at position ",
          i,
          " but found ",
          argTypeToString(argi->type()));
      const TensorArgAbstract* targ =
          reinterpret_cast<const TensorArgAbstract*>(argi);
      for (auto j : c10::irange(dom.size())) {
        expr_eval.bind(dom[j]->extent(), targ->getSize((int64_t)j));
      }
    }
  }
  return DynamicTransformConcretizationInfo(fusion, info, &expr_eval);
}

void DynamicTransform::concretizeFusion(
    Fusion* fusion,
    const DynamicTransformConcretizationInfo& info) {
  DynamicTransformConcretizer concretizer(fusion, info);
}

size_t DynamicTransformConcretizationInfo::hash() const {
  size_t hash = 0;
  for (const auto& [tv, view_result] : getReshapeTransforms()) {
    hash += view_result.hash();
  }
  return hash;
}

} // namespace nvfuser
