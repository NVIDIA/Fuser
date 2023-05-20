// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/cloner.h>

#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>

namespace nvfuser {

IrCloner::IrCloner(IrContainer* container) : ir_container_(container) {}

Statement* IrCloner::clone(const Statement* statement) {
  if (statement == nullptr) {
    return nullptr;
  }

  // Have we already cloned this node?
  const auto it = clones_map_.find(statement);
  if (it != clones_map_.end()) {
    return it->second;
  } else {
    auto new_node = handle(statement);

    // The base cloning constructor (Statement) should have
    // registered the new node. Failure to do so indicates
    // that something went horribly wrong.
    TORCH_INTERNAL_ASSERT(new_node != nullptr);
    TORCH_INTERNAL_ASSERT(clones_map_[statement] == new_node);

    return new_node;
  }
}

void IrCloner::registerClone(const Statement* src, Statement* clone) {
  TORCH_CHECK(src != nullptr);
  TORCH_CHECK(clone != nullptr);
  TORCH_CHECK(clones_map_.insert({src, clone}).second);
}

Statement* IrCloner::handle(const Statement* s) {
  return s->clone(this);
}

TensorView* RecomputeTv::recompute(TensorView* tv) {
  FusionGuard fg(tv->fusion());

  // Disallow recomputation of inputs or outputs. User would have to be aware of
  // these changes and informed they happened somehow.
  TORCH_INTERNAL_ASSERT(
      !tv->isFusionInput(),
      "Cannot recompute buffers that are inputs of the fusion.");

  // Grab all the expressions used to generate the TensorView
  auto exprs = StmtSort::getExprs(tv->fusion(), {tv}, false, false);

  // Run the replicator
  RecomputeTv replicator(tv->fusion());

  // Clone the expressions
  // clang-tidy: Call to virtual method 'RecomputeTv::handle' during
  // construction bypasses virtual dispatch
  for (auto expr : exprs) {
    replicator.handle(expr);
  }

  // Make const version of pointer for lookup
  const auto const_tv = tv;
  // Find the recomputed tensor from the cloner
  auto clone_it = replicator.clones_map_.find(const_tv);
  TORCH_INTERNAL_ASSERT(clone_it != replicator.clones_map_.end());
  auto cloned_val = clone_it->second;
  TORCH_INTERNAL_ASSERT(
      cloned_val->isA<TensorView>(),
      "Cloned value is somehow not a tensor view.");

  // Return the cloned value
  return cloned_val->as<TensorView>();
}

RecomputeTv::RecomputeTv(Fusion* fusion) : IrCloner(fusion), fusion_(fusion) {
  // Add inputs to the clones map to prevent cloning them.
  for (const auto inp : fusion->inputs()) {
    clones_map_[inp] = inp;
  }
  // Adds all scalar values to clones map to prevent cloning them
  for (const auto val : fusion->vals()) {
    if (val->getValType().value() == ValType::Scalar ||
        val->getValType().value() == ValType::NamedScalar) {
      clones_map_[val] = val;
    }
  }
}

Statement* RecomputeTv::handle(const Statement* s) {
  if (s->isA<TensorDomain>()) {
    return handle(s->as<TensorDomain>());
  }
  return s->clone(this);
}

Statement* RecomputeTv::handle(const TensorDomain* td) {
  // Make sure to recompute the history of the iteration domains, explicitly go
  // through the expressions and send them to IrCloner.
  auto exprs =
      StmtSort::getExprs(fusion_, {td->leaf().begin(), td->leaf().end()});

  for (auto expr : exprs) {
    IrCloner::handle(expr);
  }
  return IrCloner::handle(td);
}

} // namespace nvfuser
