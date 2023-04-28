// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <dynamic_transform.h>
#include <expr_evaluator.h>
#include <ir_utils.h>
#include <lower_utils.h>
#include <ops/utils.h>
#include <transform_iter.h>
#include <transform_view.h>
#include <utils.h>

#include <optional>

namespace nvfuser {

//! Gather information about concretizing transformations with
//! concrete input sizes.
class DynamicTransformInfoBuilder : public IterVisitor {
 public:
  DynamicTransformInfoBuilder(Fusion* fusion, ExpressionEvaluator* expr_eval);

  using IterVisitor::handle;

  // Analyze a dynamic reshape and generate AnalyzeViewResult
  void handle(ViewOp* op) override;

  const auto& getInfo() const {
    return info_;
  }

 private:
  ExpressionEvaluator* expr_eval_ = nullptr;

  DynamicTransformConcretizationInfo info_;
};

DynamicTransformInfoBuilder::DynamicTransformInfoBuilder(
    Fusion* fusion,
    ExpressionEvaluator* expr_eval)
    : expr_eval_(expr_eval), info_(fusion) {
  TORCH_INTERNAL_ASSERT(
      !fusion->isA<kir::Kernel>(),
      "Invalid container. Kernel container not allowed.\n");

  // Make sure all exactly mapped IDs have the same value in the
  // evaluator when any one of the IDs has a known value
  expr_eval_->propagateBoundValuesThroughExactMaps(fusion);

  traverseTo(fusion, fusion->getTerminatingOutputs(), false, false);
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

void DynamicTransformInfoBuilder::handle(ViewOp* op) {
  auto inp_tv = op->in()->as<TensorView>();
  auto out_tv = op->out()->as<TensorView>();

  // If there's no symblic axis, this should be a static reshape op
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
    auto extent_val = expr_eval_->evaluate(inp_id->extent());
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
    auto extent_val = expr_eval_->evaluate(out_id->extent());
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

  info_.reshape_transforms_.emplace_back(out_tv, view_result);
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
      tv->domain()->leaf() == tv->getMaybeRFactorDomain(),
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

  std::vector<IterDomain*> root_dom = updateIdVec(td->getRootDomain());
  std::vector<IterDomain*> rfactor_dom = td->hasRFactor()
      ? updateIdVec(td->getMaybeRFactorDomain())
      : std::vector<IterDomain*>();
  std::vector<IterDomain*> domain = updateIdVec(td->leaf());

  if (!mutated) {
    return;
  }

  // Update the contiguity vector. Drop the contig val if mutated to broadcast
  auto contig = td->contiguity();

  for (const auto i : c10::irange(td->getMaybeRFactorDomain().size())) {
    auto original_id = td->getMaybeRFactorDomain().at(i);
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
      contig.at(i) = c10::nullopt;
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
        id_type != IterType::Symbolic,
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

DynamicTransformConcretizationInfo DynamicTransform::getConcretizationInfo(
    Fusion* fusion,
    ExpressionEvaluator* expr_eval) {
  DynamicTransformInfoBuilder builder(fusion, expr_eval);
  return builder.getInfo();
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
