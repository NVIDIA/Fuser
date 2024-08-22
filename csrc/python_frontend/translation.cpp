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
#include <vector>

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

      // Create RecordFunctor given inputs, outputs, and attributes.
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
      // Handle only TensorViews
      NVF_ERROR(v->isA<TensorView>());
      handleOutput(v->as<TensorView>());
    }

    fd_->finalizeDefinition();
  }

  void handleOutput(const TensorView* tv) {
    // Add fusion outputs to FusionDefinition
    size_t output_index = map_val_to_fd_index_.at(tv);
    fd_->defineRecord(new OutputRecord<TensorView>(
        {fd_->recordingState(output_index)}, serde::RecordType::OutputTv));
  }

  template <typename ResultType, typename... ArgTypes>
  void handleOpRecord(
      const Expr* e,
      std::string op_name,
      ResultType (*fn)(ArgTypes...),
      serde::RecordType record_type,
      ResultType result,
      ArgTypes... args) {
    std::vector<State> argument_states;
    std::transform(
        e->inputs().begin(),
        e->inputs().end(),
        std::back_inserter(argument_states),
        [&](auto arg) {
          return fd_->recordingState(map_val_to_fd_index_.at(arg));
        });

    fd_->defineRecord(new OpRecord<ResultType, ArgTypes...>(
        argument_states,
        {fd_->recordingState(map_val_to_fd_index_.at(result))},
        "ops." + op_name,
        record_type,
        fn));
  }

  void handle(const BinaryOp* bop) final {
    Tensor output = fd_->defineTensor(bop->out()->as<TensorView>()->nDims());
    map_val_to_fd_index_.emplace(bop->out(), output());
    handleOpRecord(
        bop,
        "add",
        add,
        serde::RecordType::Binary_TV,
        bop->out()->as<TensorView>(),
        bop->lhs()->as<TensorView>(),
        bop->rhs()->as<TensorView>());
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
