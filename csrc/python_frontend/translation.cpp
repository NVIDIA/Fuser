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
#include <python_frontend/translation_utils.h>

#include <vector>

namespace nvfuser::python_frontend {

namespace {

// Given a CPP Fusion and an empty python_frontend FusionDefinition
// FusionTranslator adds the appropriate RecordFunctors corresponding to the
// CPP values and expressions.
//
// Rather than create a new FusionDefinition from the CPP Fusion, we add
// RecordFunctors to a blank FusionDefinition. This is a design decision because
// of the FusionDefinition python class, which inherits from the
// _C._FusionDefinition class created by pybind11. It is easier to operate on
// the child class directly than to create a new child instance from parent
// instance.
class FusionTranslator : public OptInConstDispatch {
 public:
  static void translate(Fusion* fusion, FusionDefinition* fd) {
    NVF_ERROR(
        !fd->completed(),
        "Expected an incomplete definition before fusion translation!");
    FusionTranslator translator(fusion, fd);
    translator.translate();
  }

 private:
  FusionTranslator(Fusion* fusion, FusionDefinition* fd)
      : fusion_(fusion), fd_(fd) {}

  // Check that all of the expression's inputs are defined in FusionDefinition.
  bool checkExpressionDependencies(Expr* e) {
    return std::all_of(
        e->inputs().begin(), e->inputs().end(), [&](const Val* v) {
          return map_val_to_fd_index_.count(v) > 0;
        });
  }

  void translate() {
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

      // short-circuit: add to back of stack if not all of the expression's
      // dependencies are satisfied.
      if (!checkExpressionDependencies(e)) {
        to_visit.push_back(e);
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
      // TODO Add support for aliased outputs
      // Handle only TensorViews
      NVF_ERROR(v->isA<TensorView>());
      handleOutput(v->as<TensorView>());
    }

    fd_->finalizeDefinition();
  }

  // Add scalar value to Fusion Definition
  void handle(const Val* v) final {
    Scalar output = fd_->defineScalar();
    fd_->defineRecord(new ScalarRecord(
        {fd_->recordingState(output())},
        v->value(),
        std::get<PrimDataType>(v->dtype().type)));
    map_val_to_fd_index_.emplace(v, output());
  }

  // Add Tensor value to Fusion Definition
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

  // Add Tensor output to FusionDefinition
  void handleOutput(const TensorView* tv) {
    size_t output_index = map_val_to_fd_index_.at(tv);
    fd_->defineRecord(new OutputRecord<TensorView>(
        {fd_->recordingState(output_index)}, serde::RecordType::OutputTv));
  }

  // Add Broadcast operation to FusionDefinition
  void handle(const BroadcastOp* bcast_op) final {
    Tensor output =
        fd_->defineTensor(bcast_op->out()->as<TensorView>()->nDims());
    fd_->defineRecord(new BroadcastOpRecord(
        {fd_->recordingState(map_val_to_fd_index_.at(bcast_op->in()))},
        {fd_->recordingState(output())},
        "ops.broadcast",
        bcast_op->getBroadcastDimFlags()));
    map_val_to_fd_index_.emplace(bcast_op->out(), output());
  }

  // A generic function to map UnaryOp, BinaryOp, and TernaryOp to
  // python_frontend OpRecord
  template <typename ExprType, typename ResultType, typename... ArgTypes>
  void handleOpRecord(
      const Expr* e,
      serde::RecordType record_type,
      ResultType result,
      ArgTypes... args) {
    NVF_ERROR(e->isA<ExprType>());
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
        "ops." + getString(e->as<ExprType>()),
        record_type,
        getFunction<ResultType, ArgTypes...>(e->as<ExprType>())));
  }

  // Map UnaryOp to python_frontend OpRecord
  void handle(const UnaryOp* uop) final {
    // TODO Add support for other unary ops
    // Handle only Cast operation
    NVF_ERROR(uop->getUnaryOpType() == UnaryOpType::Cast);
    handleCastOp(uop);
  }

  // Map cast UnaryOp to CastOpRecord
  void handleCastOp(const UnaryOp* uop) {
    NVF_ERROR(uop->getUnaryOpType() == UnaryOpType::Cast);

    size_t input_fd_index = map_val_to_fd_index_.at(uop->in());
    if (uop->in()->isA<TensorView>()) {
      Tensor output = fd_->defineTensor(uop->out()->as<TensorView>()->nDims());
      map_val_to_fd_index_.emplace(uop->out(), output());
      fd_->defineRecord(new CastOpRecord<TensorView*, TensorView*>(
          {fd_->recordingState(input_fd_index)},
          {fd_->recordingState(output())},
          "ops.cast",
          serde::RecordType::CastTv,
          static_cast<TensorView* (*)(DataType, TensorView*)>(castOp),
          std::get<PrimDataType>(uop->in()->dtype().type)));
    } else {
      Scalar output = fd_->defineScalar();
      map_val_to_fd_index_.emplace(uop->out(), output());
      fd_->defineRecord(new CastOpRecord<Val*, Val*>(
          {fd_->recordingState(input_fd_index)},
          {fd_->recordingState(output())},
          "ops.cast",
          serde::RecordType::CastVal,
          static_cast<Val* (*)(DataType, Val*)>(castOp),
          std::get<PrimDataType>(uop->in()->dtype().type)));
    }
  }

  // Map BinaryOp to python_frontend OpRecord
  void handle(const BinaryOp* bop) final {
    bool is_lhs_tv = bop->lhs()->isA<TensorView>();
    bool is_rhs_tv = bop->rhs()->isA<TensorView>();

    if (is_lhs_tv || is_rhs_tv) {
      Tensor output = fd_->defineTensor(bop->out()->as<TensorView>()->nDims());
      map_val_to_fd_index_.emplace(bop->out(), output());

      if (is_lhs_tv && is_rhs_tv) {
        handleOpRecord<nvfuser::BinaryOp>(
            bop,
            serde::RecordType::Binary_TV,
            bop->out()->as<TensorView>(),
            bop->lhs()->as<TensorView>(),
            bop->rhs()->as<TensorView>());
      } else if (is_lhs_tv && !is_rhs_tv) {
        handleOpRecord<nvfuser::BinaryOp>(
            bop,
            serde::RecordType::Binary_TV_VAL,
            bop->out()->as<TensorView>(),
            bop->lhs()->as<TensorView>(),
            bop->rhs());
      } else {
        handleOpRecord<nvfuser::BinaryOp>(
            bop,
            serde::RecordType::Binary_VAL_TV,
            bop->out()->as<TensorView>(),
            bop->lhs(),
            bop->rhs()->as<TensorView>());
      }
    } else {
      Scalar output = fd_->defineScalar();
      map_val_to_fd_index_.emplace(bop->out(), output());
      handleOpRecord<nvfuser::BinaryOp>(
          bop,
          serde::RecordType::Binary_VAL,
          bop->out(),
          bop->lhs(),
          bop->rhs());
    }
  }

 private:
  //! The reference CPP fusion to be translated.
  Fusion* fusion_ = nullptr;
  //! The blank FusionDefinition that receives the RecordFunctors for
  //! translated CPP values and expressions.
  FusionDefinition* fd_ = nullptr;
  //! Map nvfuser Val to FusionDefinition index.
  std::unordered_map<const nvfuser::Val*, size_t> map_val_to_fd_index_;
};

} // namespace

void translate(Fusion* fusion, FusionDefinition* fd) {
  FusionTranslator::translate(fusion, fd);
}

} // namespace nvfuser::python_frontend
