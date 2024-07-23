// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <c10/util/irange.h>
#include <exceptions.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>

#include <vector>

/*
 * Mutators are the mechanism used to modify IR nodes. Since most nodes are
 * immutable or at least partially immutable changeing them can require creating
 * a new node. Base mutator at the moment is a dumb sample mutator that takes
 * any float of value 1.0 and converts it to 0.0; It is currently used as a
 * dummy example, however, we should make it a simple instantiation of all the
 * mutate functions on all node types so that people can inherit it, and only
 * specialize those nodes which they want to have a particular transformation.
 */

namespace nvfuser {

void OptOutMutator::dispatchMutate(Statement* s) {
  Statement::mutatorDispatch(this, s);
}

void OptOutMutator::dispatchMutate(Val* v) {
  Val::mutatorDispatch(this, v);
}

Val* OptOutMutator::maybeMutated(Val* val) const {
  const auto val_it = mutations_.find(val);
  if (val_it == mutations_.end()) {
    return val;
  }
  // Check whether val is further mutated and throw error if so. This is
  // to prevent errors where we depend on recursive mutation, which can be
  // confusion/ambiguous to support.
  const auto two_hop_it = mutations_.find(val_it->second);
  NVF_ERROR(
      two_hop_it == mutations_.end(),
      "Two-hop mutations are not supported. Found registrations from ",
      val->toString(),
      " to ",
      val_it->second->toString(),
      " to ",
      two_hop_it->second->toString());
  return val_it->second;
}

void OptOutMutator::registerMutation(Val* val, Val* mutation) {
  if (val == mutation) {
    // Avoid registering trivial mutations since they are wasteful and
    // complicate the two-hop check in maybeMutated
    return;
  }
  std::cout << "registerMutation: " << val->toString() << " -> " << mutation->toString() << std::endl;
  bool val_is_ns = val->vtype() == ValType::NamedScalar;
  bool mutation_is_ns = mutation->vtype() == ValType::NamedScalar;
  bool val_is_scalar = val->vtype() == ValType::Others;
  bool mutation_is_scalar = mutation->vtype() == ValType::Others;
  NVF_ERROR(
      mutation->dtype() == val->dtype() &&
          (mutation->vtype() == val->vtype() ||
           ((val_is_ns && mutation_is_scalar) ||
            (mutation_is_ns && val_is_scalar))),
      "Mutations are not allowed to change types, tried to go from: (",
      val->vtype(),
      ", ",
      val->dtype(),
      ") to: (",
      mutation->vtype(),
      ", ",
      mutation->dtype(),
      ")");
  mutations_[val] = mutation;
}

void OptOutMutator::mutate(Val* s) {}

void OptOutMutator::mutate(NamedScalar* ns) {}

void OptOutMutator::mutate(IterDomain* id) {
  std::cout << "mutate IterDomain: " << id->toString() << std::endl;
  Val* start = maybeMutated(id->start());
  Val* extent = maybeMutated(id->extent());
  Val* expanded_extent = nullptr;
  if (id->hasExpandedExtent()) {
    expanded_extent = maybeMutated(id->expandedExtent());
  }
  Val* stop_offset = maybeMutated(id->stopOffset());
  if (start->sameAs(id->start()) && extent->sameAs(id->extent()) &&
      (!id->hasExpandedExtent() ||
       expanded_extent->sameAs(id->expandedExtent())) &&
      stop_offset->sameAs(id->stopOffset())) {
    std::cout << "mutate IterDomain: skipped" << std::endl;
    return;
  }
  auto new_id = IterDomainBuilder(id)
                    .start(start)
                    .extent(extent)
                    .stop_offset(stop_offset)
                    .expanded_extent(expanded_extent)
                    .build();
  registerMutation(
      id,
      new_id);
  std::cout << "mutate IterDomain: done, new_id " << new_id->toString() << std::endl;
}

void OptOutMutator::mutate(TensorDomain* td) {
  bool mutated = false;
  std::cout << "mutate TensorDomain: " << td->toString() << std::endl;

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
  std::vector<IterDomain*> allocation_dom = td->hasAllocation()
      ? updateIdVec(td->allocation())
      : std::vector<IterDomain*>();
  std::vector<IterDomain*> domain = updateIdVec(td->loop());
  std::cout << "mutate TensorDomain new domain: " << std::endl;
  for(auto id : domain) {
    std::cout << "id: " << id->toString() << std::endl;
  }
  for(auto id : root_dom) {
    std::cout << "root_dom: " << id->toString() << std::endl;
  }
  for(auto id : logical_dom) {
    std::cout << "logical_dom: " << id->toString() << std::endl;
  }
  for(auto id : allocation_dom) {
    std::cout << "allocation_dom: " << id->toString() << std::endl;
  }  
  if (!mutated) {
    std::cout << "mutate TensorDomain: not mutated" << std::endl;
    return;
  }

  std::cout << "mutate TensorDomain: createInContainer TensorDomain" << std::endl;
  Val* mutated_val = IrBuilder::createInContainer<TensorDomain>(
      td->container(),
      root_dom,
      logical_dom,
      allocation_dom,
      domain,
      td->contiguity());
  std::cout << "mutate TensorDomain: registerMutation" << std::endl;
  registerMutation(td, mutated_val);
  std::cout << "mutate TensorDomain: done" << std::endl;
}

void OptOutMutator::mutate(TensorView* tv) {
  std::cout << "mutate tv: " << tv->toString() << std::endl;
  TensorDomain* td = maybeMutated(tv->domain())->as<TensorDomain>();
  if (!tv->domain()->sameAs(td)) {
    tv->setDomain(td);
  }
  std::cout << "mutate tv: done" << std::endl;
  // Don't register tv mutations as we just want to update the TD
}

void OptOutMutator::mutate(kir::Predicate*) {
  NVF_ERROR(false, "Not implemented yet.");
}

void OptOutMutator::mutate(kir::TensorIndex*) {
  NVF_ERROR(false, "Not implemented yet.");
}

Expr* OptOutMutator::mutateExpr(
    Expr* op,
    bool replace_outputs,
    bool replace_inputs,
    bool replace_attrs) {
  std::cout << "mutate Expr: " << op->toString() << std::endl;
  std::cout << "replace_outputs: " << replace_outputs << " replace_inputs: " << replace_inputs << " replace_attrs: " << replace_attrs << std::endl;
  std::vector<Val*> mutated_outputs;
  mutated_outputs.reserve(op->outputs().size());
  for (auto output : op->outputs()) {
    if(const auto tv = dynamic_cast<TensorView*>(output)) {
      std::cout << "mutate Expr: tv " << tv->toString() << std::endl;
      tv->printTransforms();
    }
    mutated_outputs.emplace_back(
        replace_outputs ? maybeMutated(output) : output);
  }

  std::vector<Val*> mutated_inputs;
  mutated_inputs.reserve(op->inputs().size());
  for (auto input : op->inputs()) {
    mutated_inputs.emplace_back(replace_inputs ? maybeMutated(input) : input);
  }

  std::vector<Statement*> mutated_attrs;
  mutated_attrs.reserve(op->attributes().size());
  for (auto attr : op->attributes()) {
    if (auto attr_val = dynamic_cast<Val*>(attr)) {
      mutated_attrs.emplace_back(
          replace_inputs ? maybeMutated(attr_val) : attr_val);
    } else {
      mutated_attrs.emplace_back(attr);
    }
  }

  bool all_same = true;
  for (auto i : c10::irange(op->outputs().size())) {
    if (!all_same) {
      break;
    }
    all_same = all_same && mutated_outputs[i] == op->output(i);
  }
  for (auto i : c10::irange(op->inputs().size())) {
    if (!all_same) {
      break;
    }
    all_same = all_same && mutated_inputs[i] == op->input(i);
  }
  for (auto i : c10::irange(op->attributes().size())) {
    if (!all_same) {
      break;
    }
    bool same =
        ((mutated_attrs[i] == nullptr) && (op->attribute(i) == nullptr)) ||
        mutated_attrs[i] == op->attribute(i);
    all_same = all_same && same;
  }

  if (all_same) {
    std::cout << "mutate expr: done all_same" << std::endl;
    return op;
  }

  auto container = op->container();
  auto newObjectFunc = op->newObjectFunc();
  removeExpr(container, op);
  auto new_expr =
      newObjectFunc(container, mutated_inputs, mutated_outputs, mutated_attrs);
  registerNewExpr(new_expr);

  std::cout << "mutate expr: done new_expr " << new_expr->toString() << std::endl;
  return new_expr;
}

void OptOutMutator::removeExpr(IrContainer* container, Expr* expr) const {
  container->removeExpr(expr);
}

} // namespace nvfuser
