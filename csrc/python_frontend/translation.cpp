// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
//
#include <dispatch.h>
#include <python_frontend/translation.h>

namespace nvfuser::python_frontend {

namespace {

class FusionTranslator : public OptInConstDispatch {
 public:
  static std::unique_ptr<FusionDefinition> clone(const Fusion* fusion) {
    FusionTranslator cloner(fusion);
    cloner.clone();
    return std::move(cloner.fd_);
  }

 private:
  FusionTranslator(const Fusion* fusion)
      : fusion_(fusion),
        fd_(std::make_unique<FusionDefinition>(/*id=*/std::nullopt)) {}

  using OptInConstDispatch::handle;

  void clone() {
    fd_->setupDefinition();

    // nvfuser::Val - defineScalar, defineTensor, and defineTensor
    // nvfuser::Expr - defineRecord
    std::deque<nvfuser::Expr*> to_visit;

    // Add Fusion inputs to FusionDefinition
    for (nvfuser::Val* v : fusion_->inputs()) {
      OptOutConstDispatch::dispatch(v);

      // Add uses for input value to to_visit
      for (Expr* e : v->uses()) {
        to_visit.push_back(e);
      }
    }

    // Topological search of Fusion expressions
    std::unordered_set<nvfuser::Expr*> visited;
    while (!to_visit.empty()) {
      Expr* e = to_visit.front();
      to_visit.pop_front();

      // short-circuit: skip if already visited
      if (visited.count(e) > 0) {
        continue;
      }

      visited.insert(e);

      OptOutConstDispatch::dispatch(e);

      // Add output uses to to_visit
      for (Val* v : e->outputs()) {
        for (Expr* e : v->uses()) {
          to_visit.push_back(e);
        }
      }
    }

    // Outputs and Aliasing
    for (nvfuser::Val* v : fusion_->outputs()) {
      // TODO Handle only TensorViews
      NVF_ERROR(v->isA<TensorView>());

      // Add fusion outputs to FusionDefinition
      size_t output_index = map_val_to_fd_index_.at(v);
      fd_->defineRecord(new OutputRecord<TensorView>(
          {fd_->recordingState(output_index)}, serde::RecordType::OutputTv));
    }

    fd_->finalizeDefinition();
  }

  void handle(const TensorView* tv) final {
    Tensor output = fd_->defineTensor(tv->nDims());

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

    fd_->defineRecord(new TensorRecord(
        {fd_->recordingState(output())},
        shape,
        tv->domain()->contiguity(),
        std::get<PrimDataType>(tv->dtype().type),
        tv->isCpuScalar(),
        tv->domain()->strideOrder()));

    map_val_to_fd_index_.emplace(tv, output());
  }

  void handle(const BinaryOp* bop) final {
    NVF_ERROR(bop->lhs()->isA<TensorView>());
    NVF_ERROR(bop->rhs()->isA<TensorView>());

    // Create RecordFunctor given inputs, outputs, and attributes.
    // Add output to values
    TensorView* arg1 = bop->lhs()->as<TensorView>();
    size_t arg1_index = map_val_to_fd_index_.at(bop->lhs());
    size_t arg2_index = map_val_to_fd_index_.at(bop->rhs());

    Tensor output = fd_->defineTensor(arg1->nDims());
    fd_->defineRecord(new OpRecord<TensorView*, TensorView*, TensorView*>(
        {fd_->recordingState(arg1_index), fd_->recordingState(arg2_index)},
        {fd_->recordingState(output())},
        ("ops.add"),
        serde::RecordType::Binary_TV,
        static_cast<TensorView* (*)(TensorView*, TensorView*)>(add)));
    map_val_to_fd_index_.emplace(bop->out(), output());
  }

 private:
  const Fusion* fusion_ = nullptr;
  std::unique_ptr<FusionDefinition> fd_;
  std::unordered_map<const nvfuser::Val*, size_t> map_val_to_fd_index_;
};

} // namespace

std::unique_ptr<FusionDefinition> clone(const Fusion* fusion) {
  return FusionTranslator::clone(fusion);
}

} // namespace nvfuser::python_frontend
