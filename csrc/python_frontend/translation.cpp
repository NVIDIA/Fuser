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
  static std::unordered_map<const nvfuser::Val*, size_t> translate(
      Fusion* fusion,
      FusionDefinition* fd) {
    NVF_ERROR(
        !fd->completed(),
        "Expected an incomplete definition before fusion translation!");
    FusionTranslator translator(fusion, fd);
    translator.translate();
    return translator.map_val_to_fd_index_;
  }

 private:
  FusionTranslator(Fusion* fusion, FusionDefinition* fd)
      : fusion_(fusion), fd_(fd) {}

  // The new shape for view operation can be dynamic. Check that all dynamic
  // scalar dependencies are handled before the ViewOp.
  bool checkViewShapeDependency(const ViewOp* vop) {
    const std::vector<IterDomain*>& logical_out_domain =
        vop->out()->as<TensorView>()->domain()->logical();
    std::vector<Val*> logical_domain_extents;
    std::transform(
        logical_out_domain.begin(),
        logical_out_domain.end(),
        std::back_inserter(logical_domain_extents),
        [](IterDomain* id) { return id->getMaybeExpandedExtent(); });
    return std::all_of(
        logical_domain_extents.begin(),
        logical_domain_extents.end(),
        [&](Val* v) {
          return v->definition() == nullptr ||
              map_val_to_fd_index_.count(v) > 0;
        });
  }

  // Gather the expressions necessary to create a scalar value.
  std::vector<Expr*> gatherScalarExpressions(Val* v) {
    NVF_ERROR(v != nullptr);
    NVF_ERROR(v->isScalar());

    // short-circuit: v does not have a definition.
    if (v->definition() == nullptr) {
      return {};
    }

    std::vector<Expr*> expression_chain;
    std::unordered_set<Expr*> visited;
    std::vector<Expr*> to_visit = {v->definition()};
    while (!to_visit.empty()) {
      Expr* e = to_visit.back();
      to_visit.pop_back();

      expression_chain.push_back(e);
      visited.insert(e);

      for (Val* input : e->inputs()) {
        // short-circuit: input does not have a definition.
        if (input->definition() == nullptr) {
          continue;
        }

        // short-circuit: input definition is already visited.
        if (visited.count(input->definition()) > 0) {
          continue;
        }

        to_visit.push_back(input->definition());
      }
    }
    return expression_chain;
  }

  // Gather the scalar expressions necessary to create the logical domain for a
  // TensorView.
  std::vector<Expr*> gatherScalarExpressions(TensorView* tv) {
    NVF_ERROR(tv != nullptr);
    std::vector<Expr*> logical_domain_expressions;
    const std::vector<IterDomain*>& logical_out_domain =
        tv->domain()->logical();
    for (IterDomain* id : logical_out_domain) {
      std::vector<Expr*> extent_definitions =
          gatherScalarExpressions(id->getMaybeExpandedExtent());
      logical_domain_expressions.insert(
          logical_domain_expressions.end(),
          extent_definitions.begin(),
          extent_definitions.end());
    }
    return logical_domain_expressions;
  }

  // Check that all of the expression's inputs are defined in FusionDefinition.
  bool checkExpressionDependencies(Expr* e) {
    bool check_view_dependency =
        !e->isA<ViewOp>() || checkViewShapeDependency(e->as<ViewOp>());
    return check_view_dependency &&
        std::all_of(e->inputs().begin(), e->inputs().end(), [&](const Val* v) {
             return map_val_to_fd_index_.count(v) > 0;
           });
  }

  void translate() {
    fd_->setupDefinition();

    // Add Fusion inputs to FusionDefinition
    for (nvfuser::Val* v : fusion_->inputs()) {
      OptOutConstDispatch::dispatch(v);
    }

    // Gather all expressions in CPP Fusion.
    const std::vector<nvfuser::Expr*> fusion_exprs = fusion_->exprs();
    std::deque<nvfuser::Expr*> to_visit(
        fusion_exprs.begin(), fusion_exprs.end());

    // Scalar expressions are not handled by usedMathVals, so gather them
    // manually.
    for (Expr* e : to_visit) {
      if (e->isA<ViewOp>() || e->isA<ExpandOp>() || e->isA<FullOp>()) {
        std::vector<Expr*> extent_definitions =
            gatherScalarExpressions(e->output(0)->as<TensorView>());
        to_visit.insert(
            to_visit.end(),
            extent_definitions.begin(),
            extent_definitions.end());
      }
    }

    // Topological search of Fusion expressions
    size_t skip_count = 0;
    std::unordered_set<nvfuser::Expr*> visited;
    while (!to_visit.empty()) {
      Expr* e = to_visit.front();
      to_visit.pop_front();

      NVF_ERROR(
          skip_count <= to_visit.size(),
          "Cycle detected: None of the expressions can be processed!");

      // short-circuit: skip if already visited
      if (visited.count(e) > 0) {
        continue;
      }

      // short-circuit: skip Split and Merge expressions created by Reshape
      // short-circuit: skip Resize expressions created by Slice
      if (e->isA<Split>() || e->isA<Merge>() || e->isA<Resize>()) {
        visited.insert(e);
        continue;
      }

      // Handle scalars and constants not generated by separate expression.
      std::vector<Val*> scalars;
      std::copy_if(
          e->inputs().begin(),
          e->inputs().end(),
          std::back_inserter(scalars),
          [](Val* v) { return v->isScalar(); });
      std::for_each(scalars.begin(), scalars.end(), [this](const Val* v) {
        OptOutConstDispatch::dispatch(v);
      });

      // short-circuit: add to back of stack if not all of the expression's
      // dependencies are satisfied.
      if (!checkExpressionDependencies(e)) {
        ++skip_count;
        to_visit.push_back(e);
        continue;
      }

      // Create RecordFunctor given inputs, outputs, and attributes.
      visited.insert(e);
      OptOutConstDispatch::dispatch(e);
      skip_count = 0;
    }

    // Add tensor outputs and handle aliased outputs
    std::unordered_set<nvfuser::Val*> visited_alias_output;
    for (nvfuser::Val* v : fusion_->outputs()) {
      NVF_ERROR(v->isA<TensorView>());
      const AliasInfo& alias_info = fusion_->getOutputAlias(v);
      switch (alias_info.type) {
        case AllocationType::New: {
          handleOutput(v->as<TensorView>());
          break;
        }
        case AllocationType::ReuseBuffer: {
          size_t num_visited = visited_alias_output.count(v);
          if (num_visited == 0) {
            visited_alias_output.insert(v);
            handleOutput(v->as<TensorView>(), alias_info);
          }
          // An alias output can also be returned as a fusion output
          // if it is already aliased or if not hide_output
          if (num_visited > 0 || !alias_info.hide_output) {
            handleOutput(v->as<TensorView>());
          }
          break;
        }
        default:
          NVF_ERROR(false, "Unsupported AllocationType");
      }
    }

    fd_->finalizeDefinition();
  }

  // =================================================================================
  //  Handle define_scalar and define_tensor variants

  // Add scalar value to Fusion Definition
  void handle(const Val* v) final {
    // short-circuit: value already exists in FusionDefinition
    if (map_val_to_fd_index_.count(v) > 0) {
      return;
    }

    Scalar output = fd_->defineScalar();
    fd_->defineRecord(new ScalarRecord(
        {fd_->recordingState(output())},
        v->value(),
        std::get<PrimDataType>(v->dtype().type)));
    map_val_to_fd_index_.emplace(v, output());
  }

  // Add Tensor value to Fusion Definition
  void handle(const TensorView* tv) final {
    // short-circuit: value already exists in FusionDefinition
    if (map_val_to_fd_index_.count(tv) > 0) {
      return;
    }

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

  // =================================================================================
  // Handle add_output variants

  // Add Tensor output to FusionDefinition
  void handleOutput(const TensorView* tv) {
    size_t output_index = map_val_to_fd_index_.at(tv);
    fd_->defineRecord(new OutputRecord<TensorView>(
        {fd_->recordingState(output_index)},
        serde::RecordType::OutputTv,
        tv->domain()->strideOrder()));
  }

  // Alias output Tensor with input tensor
  void handleOutput(const TensorView* tv, const AliasInfo& alias_info) {
    size_t output_index = map_val_to_fd_index_.at(tv);
    size_t input_index = map_val_to_fd_index_.at(alias_info.aliased_io);
    fd_->defineRecord(new OutputRecord<TensorView>(
        {fd_->recordingState(output_index), fd_->recordingState(input_index)},
        serde::RecordType::OutputTv));
  }

  // =================================================================================
  // Map CPP Expression classes to corresponding RecordFunctors in
  // python_frontend

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
          std::get<PrimDataType>(uop->out()->dtype().type)));
    } else {
      Scalar output = fd_->defineScalar();
      map_val_to_fd_index_.emplace(uop->out(), output());
      fd_->defineRecord(new CastOpRecord<Val*, Val*>(
          {fd_->recordingState(input_fd_index)},
          {fd_->recordingState(output())},
          "ops.cast",
          serde::RecordType::CastVal,
          static_cast<Val* (*)(DataType, Val*)>(castOp),
          std::get<PrimDataType>(uop->out()->dtype().type)));
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

std::unordered_map<const nvfuser::Val*, size_t> translate(
    Fusion* fusion,
    FusionDefinition* fd) {
  return FusionTranslator::translate(fusion, fd);
}

} // namespace nvfuser::python_frontend
