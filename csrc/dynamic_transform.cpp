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
#include <root_domain_map.h>
#include <transform_view.h>
#include <utils.h>

#include <optional>

namespace nvfuser {

class TORCH_CUDA_CU_API DynamicTransformInfoBuilder : public IterVisitor {
 public:
  DynamicTransformInfoBuilder(Fusion* fusion, ExpressionEvaluator* expr_eval)
      : expr_eval_(expr_eval), info_(fusion) {
    TORCH_INTERNAL_ASSERT(
        !fusion->isA<kir::Kernel>(),
        "Invalid container. Kernel container not allowed.\n");

    augmentExprEvaluator();

    traverseTo(fusion, fusion->getTerminatingOutputs(), false, false);
  }

  using IterVisitor::handle;

  void handle(ViewOp* op) override;

  const auto& getInfo() const {
    return info_;
  }

 private:
  void augmentExprEvaluator();

 private:
  ExpressionEvaluator* expr_eval_ = nullptr;

  DynamicTransformInfo info_;
};

// TODO
bool DynamicTransformInfo::operator==(const DynamicTransformInfo& other) const {
  return false;
}

// TODO
std::string DynamicTransformInfo::toString() const {
  std::stringstream ss;
  ss << "DynamicTransformInfo\n";
  std::string indent = "  ";
  ss << indent << "Reshape:\n";
  for (const auto& kv : reshape_transforms_) {
    ss << indent << indent << kv.first->toString() << ", "
       << kv.second.toString() << "\n";
  }
  return ss.str();
}

DynamicTransformInfo DynamicTransformInfo::get(
    Fusion* fusion,
    ExpressionEvaluator* expr_eval) {
  DynamicTransformInfoBuilder builder(fusion, expr_eval);
  return builder.getInfo();
}

void DynamicTransformInfoBuilder::augmentExprEvaluator() {
  const auto mapped_sets = ExactRootDomainMap(info_.fusion()).getMappedSets();

  // std::cerr << "Augmenting ExprEval\n";

  for (const auto& set : mapped_sets.disjointSets()) {
    // std::cerr << "Disjoint set\n";
    int64_t known_size = -1;
    std::vector<Val*> unknown_vals;
    for (const auto id : *set) {
      // std::cerr << "ID: " << id->toString() << std::endl;
      auto eval_val = expr_eval_->evaluate(id->extent());
      if (eval_val.has_value()) {
        TORCH_INTERNAL_ASSERT(eval_val->isInt(), "Invalid extent value");
        int64_t this_size = eval_val->as<int64_t>();
        if (known_size != -1) {
          TORCH_INTERNAL_ASSERT(
              known_size == this_size,
              "Conflicting sizes: ",
              known_size,
              ", ",
              this_size);
        } else {
          known_size = this_size;
        }
      } else {
        unknown_vals.push_back(id->extent());
      }
    }

    if (known_size == -1 || unknown_vals.empty()) {
      continue;
    }

    // Binding unknown vals to known_val
    for (auto unknown_val : unknown_vals) {
      // std::cerr << "Augment: " << unknown_val->toString() << " -> "
      //<< known_size << std::endl;
      expr_eval_->bind(unknown_val, known_size);
    }
  }

  // std::cerr << "Augment done\n";
}

void DynamicTransformInfoBuilder::handle(ViewOp* op) {
  std::cerr << "Reshape: " << op->toString();

  // Determine if this is a dynamic reshape. If the output tv doesn't
  // have an rfactor domain, no view transform is set up, which means
  // its output domain is still just a placeholder

  auto inp_tv = op->in()->as<TensorView>();
  auto out_tv = op->out()->as<TensorView>();

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

  std::cerr << "Input shape: " << inp_shape << std::endl;

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

  std::cerr << "Output shape: " << out_shape << std::endl;

  auto view_result = analyzeView(inp_tv, inp_shape, out_shape);

  info_.reshape_transforms_.emplace_back(out_tv, view_result);
}

void DynamicTransformConcretizer::concretizeFusion(
    Fusion* fusion,
    const DynamicTransformInfo& info) {
  DEBUG_PRINT_SCOPE();

  DynamicTransformConcretizer concretizer(fusion, info);

  concretizer.concretize();
}

void DynamicTransformConcretizer::concretize() {
  DEBUG_PRINT_SCOPE();

  concretizeReshape();

  auto all_stmts = StmtSort::getStmts(info_.fusion(), true);
  for (auto stmt : all_stmts) {
    if (stmt->isA<Val>()) {
      std::cerr << "mutate: " << stmt->toString() << std::endl;
      mutate(stmt);
    }
  }
}

void DynamicTransformConcretizer::concretizeReshape() {
  DEBUG_PRINT_SCOPE();

  // Concretize each reshape op
  for (const auto& kv : info_.getReshapeTransforms()) {
    auto incomplete_out_tv = kv.first;
    const auto view_analysis = kv.second;

    std::cerr << "view: " << view_analysis.toString() << std::endl;

    auto inp_tv = ir_utils::producerTvsOf(incomplete_out_tv).at(0);

    auto concrete_reshape_out_tv = reshape(inp_tv, view_analysis);

    std::cerr << "concrete view out: " << concrete_reshape_out_tv->toString()
              << ", expr: " << concrete_reshape_out_tv->definition()->toString()
              << std::endl;

    if (inp_tv != concrete_reshape_out_tv->definition()->input(0)) {
      std::cerr << "Addnl expr: "
                << concrete_reshape_out_tv->definition()
                       ->input(0)
                       ->definition()
                       ->toString();
    }

    // Replace the old tensor with the new concretized tensor
    for (auto use_of_old_tv : incomplete_out_tv->uses()) {
      std::cerr << "Before replacement: " << use_of_old_tv->toString();
      auto new_use = ir_utils::replaceValInExpr(
          use_of_old_tv, incomplete_out_tv, concrete_reshape_out_tv);
      std::cerr << "After replacement: " << new_use->toString();
    }
  }
}
#if 0
void DynamicTransformConcretizer::mutate(Expr* expr) {
  if (ir_utils::isTvOp(expr)) {
    handleTensorViewExpr(expr);
  } else if (ir_utils::isIterDomainOp(expr)) {
    handleIterDomainExpr(expr);
  }
  OptOutMutator::mutate(expr);
}
#endif
namespace {

bool hasSymbolicAxis(const std::vector<IterDomain*>& ids) {
  return std::any_of(ids.begin(), ids.end(), [](IterDomain* id) {
    return id->getIterType() == IterType::Symbolic;
  });
}

} // namespace
#if 0
void DynamicTransformConcretizer::handleTensorViewExpr(Expr* expr) {
  std::cerr << "TV expr: " << expr->toString();

  for (auto output_tv: ir_utils::filterByType<TensorView>(expr->outputs())) {

    if (std::none_of(output_tv->domain()->domain().begin(),
                     output_tv->domain()->domain().end(),
                     [](IterDomain* id) {
                       return id->getIterType() == IterType::Symbolic;
                     })) {
      continue;
    }

    auto output_root_domain = output_tv->domain()->domain();

    std::vector<IterType> output_domain_types;
    std::transform(output_tv->domain()->domain().begin(),
                   output_tv->domain()->domain().end(),
                   std::back_inserter(output_domain_types),
                   [](auto id) {
                     return id->getIterType();
                   });

    for (const auto i: c10::irange(output_root_domain.size())) {
      auto output_id = output_root_domain.at(i);
      if (output_id->getIterType() != IterType::Symbolic) {
        continue;
      }

      auto output_type = IterType::Symbolic;

      for (auto input_tv: ir_utils::filterByType<TensorView>(expr->inputs())) {
        PairwiseRootDomainMap root_map(input_tv, output_tv);
        auto c2p = root_map.mapConsumerToProducer(output_tv->domain(), input_tv->domain());

        TORCH_INTERNAL_ASSERT(c2p.find(output_id) != c2p.end(),
                              "No input ID found to map with output ID: ", output_id->toString());
        auto input_id = c2p.at(output_id);

        output_type = ops::promoteIterType(output_type, input_id->getIterType());
      }

      TORCH_INTERNAL_ASSERT(output_type != IterType::Symbolic,
                            "Failed to concretize ", output_id->toString(), " of ",
                            output_tv->toString());

      auto concretized_output_id = IterDomainBuilder(output_id).iter_type(output_type).build();
      std::cerr << output_id->toString() << ", new type: " << output_type
                << ", new ID: " << concretized_output_id->toString()
                << std::endl;

      output_root_domain.at(i) = concretized_output_id;
      //update_map_.emplace(output_id, concretized_output_id);
      registerMutation(output_id, concretized_output_id);
    }
  }
}
#endif
void DynamicTransformConcretizer::mutate(TensorView* tv) {
  auto propagated = propagateFromProducerToConsumer(tv);

  if (propagated) {
    // Root domain IDs are updated. First mutate the TensorDomain and
    // then TensorView
    mutate(tv->domain());
    OptOutMutator::mutate(tv);
    std::cerr << "After mutate: " << tv->toString() << std::endl;
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

  std::vector<IterDomain*> root_dom = updateIdVec(td->getRootDomain());
  std::vector<IterDomain*> rfactor_dom = td->hasRFactor()
      ? updateIdVec(td->getMaybeRFactorDomain())
      : std::vector<IterDomain*>();
  std::vector<IterDomain*> domain = updateIdVec(td->domain());

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
  std::cerr << "Mutated new TD: " << mutated_val->toString() << std::endl;
  registerMutation(td, mutated_val);
}

bool DynamicTransformConcretizer::propagateFromProducerToConsumer(
    TensorView* consumer) {
  if (consumer->definition() == nullptr ||
      !consumer->domain()->hasSymbolicAxis()) {
    return false;
  }

  std::cerr << "propagateFromProducerToConsumer: " << consumer->toString()
            << std::endl;

  auto root_domain = consumer->getRootDomain();

  std::vector<IterType> output_domain_types;
  std::transform(
      root_domain.begin(),
      root_domain.end(),
      std::back_inserter(output_domain_types),
      [](auto id) { return id->getIterType(); });

  auto def = consumer->definition();

  for (const auto i : c10::irange(root_domain.size())) {
    auto root_id = root_domain.at(i);
    if (root_id->getIterType() != IterType::Symbolic) {
      continue;
    }

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

      std::cerr << "producer ID: " << input_id->toString() << std::endl;
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
    std::cerr << root_id->toString() << ", new type: " << *id_type
              << ", new ID: " << concretized_id->toString() << std::endl;

    registerMutation(root_id, concretized_id);
  }

  return true;
}

void DynamicTransformConcretizer::handleIterDomainExpr(Expr* expr) {
  std::cerr << "ID expr: " << expr->toString();
}

#if 0
void DynamicTransformConcretizer::handle(TensorView* tv) {
  std::cerr << "TV: " << tv->toString() << std::endl;

  IterVisitor::handle(tv);

  if (!tv->domain()->hasSymbolicAxis()) {
    return;
  }

  auto root_domain = tv->getRootDomain();

  if (hasSymbolicAxis(tv->getRootDomain())) {
    for (auto& id: root_domain) {
      if (id->getIterType() != IterType::Symbolic) {
        continue;
      }
      auto it = update_map_.find(id);
      TORCH_INTERNAL_ASSERT(it != update_map_.end(), "Symbolic ID not concretized: ", id->toString(),
                            " of ", tv->toString());
      id = it->second;
    }

    std::cerr << "New root domain: " << toDelimitedString(root_domain.begin(), root_domain.end()) << std::endl;
  }

  auto rf_domain = tv->getMaybeRFactorDomain();

  if (tv->hasRFactor() && hasSymbolicAxis(rf_domain)) {
    // TODO: refactor
    for (auto& id: rf_domain) {
      if (id->getIterType() != IterType::Symbolic) {
        continue;
      }
      auto it = update_map_.find(id);
      TORCH_INTERNAL_ASSERT(it != update_map_.end(), "Symbolic ID not concretized: ", id->toString(),
                            " of ", tv->toString());
      id = it->second;
    }

    std::cerr << "New root domain: " << toDelimitedString(rf_domain.begin(), rf_domain.end()) << std::endl;
  }

  // At this point, no IterDomain expr should appear beyond the rfactor domain
  TORCH_INTERNAL_ASSERT(tv->domain()->domain() == tv->getMaybeRFactorDomain(),
                        "Unexpected tensor leaf domain: ", tv->toString());

  auto concretized_td = IrBuilder::create<TensorDomain>(
      root_domain, rf_domain, rf_domain, tv->domain()->contiguity());

}
#endif
} // namespace nvfuser
