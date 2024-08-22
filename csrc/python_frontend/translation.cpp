// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
//
#include <python_frontend/translation.h>

namespace nvfuser::python_frontend {

std::unique_ptr<FusionDefinition> clone(const Fusion* fusion) {
  auto fd = std::make_unique<FusionDefinition>(/*id=*/1);
  // nvfuser::Val - defineScalar, defineTensor, and defineTensor
  // nvfuser::Expr - defineRecord

  std::unordered_map<nvfuser::Val*, size_t> map_val_to_fd_index;
  std::deque<nvfuser::Expr*> to_visit;

  // Handle Fusion inputs
  for (nvfuser::Val* v : fusion->inputs()) {
    // Add inputs to FusionDefinition

    // TODO Only handle TensorViews
    NVF_ERROR(v->isA<TensorView>());

    TensorView* tv = v->as<TensorView>();

    Tensor output = fd->defineTensor(tv->nDims());

    std::vector<int64_t> shape;
    std::transform(
        tv->domain()->logical().begin(),
        tv->domain()->logical().end(),
        std::back_inserter(shape),
        [](IterDomain* id) {
          return (id->extent()->isConstScalar())
              ? id->extent()->evaluate().as<int64_t>()
              : -1;
        });

    fd->defineRecord(new TensorRecord(
        {fd->recordingState(output())},
        shape,
        tv->domain()->contiguity(),
        std::get<PrimDataType>(tv->dtype().type),
        tv->isCpuScalar(),
        tv->domain()->strideOrder()));
    map_val_to_fd_index.emplace(v, output());

    // Add uses for input value to to_visit
    for (Expr* e : v->uses()) {
      to_visit.push_back(e);
    }
  }

  // Topological search of expressions
  std::unordered_set<nvfuser::Expr*> visited;
  while (!to_visit.empty()) {
    Expr* e = to_visit.front();
    to_visit.pop_front();

    // short-circuit: skip if visited
    if (visited.count(e) > 0) {
      continue;
    }

    visited.insert(e);

    // TODO Only handle BinaryOp Add
    NVF_ERROR(e->isA<BinaryOp>());
    BinaryOp* bop = e->as<BinaryOp>();
    NVF_ERROR(bop->lhs()->isA<TensorView>());
    NVF_ERROR(bop->rhs()->isA<TensorView>());

    // Create RecordFunctor given inputs, outputs, and attributes.
    // Add output to values
    TensorView* arg1 = bop->lhs()->as<TensorView>();
    size_t arg1_index = map_val_to_fd_index.at(bop->lhs());
    size_t arg2_index = map_val_to_fd_index.at(bop->rhs());

    Tensor output = fd->defineTensor(arg1->nDims());
    fd->defineRecord(new OpRecord<TensorView*, TensorView*, TensorView*>(
        {fd->recordingState(arg1_index), fd->recordingState(arg2_index)},
        {fd->recordingState(output())},
        ("ops.add"),
        serde::RecordType::Binary_TV,
        static_cast<TensorView* (*)(TensorView*, TensorView*)>(add)));
    map_val_to_fd_index.emplace(bop->out(), output());

    // Add output uses to to_visit
    for (Val* v : e->outputs()) {
      for (Expr* e : v->uses()) {
        to_visit.push_back(e);
      }
    }
  }

  // Outputs and Aliasing
  for (nvfuser::Val* v : fusion->outputs()) {
    // TODO Handle only TensorViews
    NVF_ERROR(v->isA<TensorView>());

    // Add fusion outputs to FusionDefinition
    size_t output_index = map_val_to_fd_index.at(v);
    fd->defineRecord(new OutputRecord<TensorView>(
        {fd->recordingState(output_index)}, serde::RecordType::OutputTv));
  }

  return fd;
}

} // namespace nvfuser::python_frontend
