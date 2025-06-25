// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
//
#include <dispatch.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <python_frontend/translation.h>
#include <python_frontend/translation_utils.h>
#include <translation_names.h>
#include <utils.h>

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
//
// How to add support for an expression not yet overriden by FusionTranslator?
//  1. Create handle function for expression.
//     a. void handle(const SomeOp* op) final
//
//  2. Add RecordFunctor corresponding to Statement to FusionDefinition.
//     a. fd_->defineRecord(new RecordFunctor(inputs, outputs)
//
//  3. If input argument already exists in FusionDefinition, map expressions
//  input values to FusionDefinition State.
//     a. map_val_to_fd_index_ maps CPP Val to fusion definition index.
//     b. fd_->recordingState(map_val_to_fd_index_.at(op->inputs(...)))
//
//  4. If input argument is a vector, use createVector function.
//
//  5. If input argument is a scalar constant, use createScalar function.
//
//  6. Create output states expressions inputs.
//     a. Tensor output = fd_->defineTensor(v->as<TensorView>()->nDims())
//
//  7. Add CPP Val and output state pair to map_val_to_fd_index_.
class FusionTranslator : public OptInConstDispatch {
 public:
  // Returns a map from the values in the CPP fusion to its corresponding
  // FusionDefinition State index.
  //
  // Why?
  // For segmentation, we divide the original FusionDefinition into its
  // segments. Each segment has a separate index namespace. To run a segment,
  // we need to pass outputs from prior segments as this segment's input
  // arguments. The original FusionDefinition coordinates this argument passing.
  // The map returned by this function is used to a global mapping from the
  // original FusionDefinition's indicies to this segment's indicies.
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

  bool isScheduledTensorView(TensorView* tv) const {
    NVF_ERROR(tv != nullptr);
    const std::vector<IterDomain*>& logical = tv->domain()->logical();
    const std::vector<IterDomain*>& loop = tv->domain()->loop();
    // short-circuit: check same length
    if (logical.size() != loop.size()) {
      return true;
    }

    for (size_t idx : arange(logical.size())) {
      if (logical.at(idx) != loop.at(idx)) {
        return true;
      }
    }
    return false;
  }

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
      dispatch(v);
    }

    // Gather all expressions in CPP Fusion.
    const std::vector<nvfuser::Expr*> fusion_exprs = fusion_->exprs();
    std::deque<nvfuser::Expr*> to_visit(
        fusion_exprs.begin(), fusion_exprs.end());

    // Scalar expressions are not handled by Fusion::exprs, so gather them
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

      bool is_expr_inputs_valid =
          std::all_of(e->inputs().begin(), e->inputs().end(), [this](Val* v) {
            return !v->isA<TensorView>() ||
                !isScheduledTensorView(v->as<TensorView>());
          });
      NVF_ERROR(
          is_expr_inputs_valid,
          "Found a TensorView with scheduled loop domain.");

      // Handle scalars and constants not generated by separate expression.
      std::vector<Val*> scalars;
      std::copy_if(
          e->inputs().begin(),
          e->inputs().end(),
          std::back_inserter(scalars),
          [](Val* v) { return v->isScalar(); });
      std::for_each(scalars.begin(), scalars.end(), [this](const Val* v) {
        dispatch(v);
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
      dispatch(e);
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
  // Filter Functions

  // Gather all TensorViews and FusionDefinition indices
  std::vector<std::pair<const nvfuser::Val*, int64_t>> tensors() {
    std::vector<std::pair<const nvfuser::Val*, int64_t>> tensors;
    std::copy_if(
        map_val_to_fd_index_.begin(),
        map_val_to_fd_index_.end(),
        std::back_inserter(tensors),
        [](std::pair<const nvfuser::Val*, int64_t>&& kv) {
          return kv.first->isA<TensorView>();
        });
    return tensors;
  }

  // =================================================================================
  //  Handle define_scalar and define_tensor variants

  // Create scalar for given nvfuser value. The nvfuser value must not already
  // exist and have a definition. It can be a fusion input, a constant, or a
  // tensor's extent.
  Scalar createScalar(const Val* v) {
    NVF_ERROR(
        v->definition() == nullptr,
        "Value has a definition and should not be created directly.");

    // short-circuit: value already exists in FusionDefinition
    if (map_val_to_fd_index_.count(v) > 0) {
      return Scalar(map_val_to_fd_index_.at(v), fd_);
    }

    Scalar output = fd_->defineScalar();
    map_val_to_fd_index_.emplace(v, output());

    // Since scalars can come from TensorView dimension sizes, search through
    // all TensorViews for an iterDomain whose extent matches the desired
    // value and then create SizeOpRecord.
    for (auto& kv : tensors()) {
      const TensorView* key_tv = kv.first->as<TensorView>();

      std::vector<IterDomain*> filtered_logical_domain =
          TensorDomain::noReductions(key_tv->domain()->logical());
      // Get extents for each IterDomain
      std::vector<Val*> extents;
      extents.reserve(filtered_logical_domain.size());
      std::transform(
          filtered_logical_domain.begin(),
          filtered_logical_domain.end(),
          std::back_inserter(extents),
          [](IterDomain* id) { return id->getMaybeExpandedExtent(); });

      auto iter = std::find(extents.begin(), extents.end(), v);
      // Check if value matches iterdomain extent
      if (iter == extents.end()) {
        continue;
      }

      int64_t dim = std::distance(extents.begin(), iter);
      fd_->defineRecord(new SizeOpRecord(
          {fd_->recordingState(kv.second)},
          {fd_->recordingState(output())},
          dim));
      return output;
    }

    // DataType::Index does not exist in python_frontend, so convert to
    // DataType::Int
    DataType scalar_dtype =
        (v->dtype() == DataType::Index) ? DataType::Int : v->dtype();

    fd_->defineRecord(new ScalarRecord(
        {fd_->recordingState(output())},
        v->value(),
        std::get<PrimDataType>(scalar_dtype.type)));
    return output;
  }

  // Add scalar value to Fusion Definition
  void handle(const Val* v) final {
    // short-circuit: scalar definition has a definition
    if (v->definition() != nullptr) {
      return;
    }
    createScalar(v);
  }

  // Create python_frontend Vector from a vector of CPP scalar values.
  Vector createVector(std::vector<Val*> scalars) {
    // Add CPP values to Fusion Definition if necessary
    std::for_each(scalars.begin(), scalars.end(), [this](const Val* v) {
      OptOutConstDispatch::dispatch(v);
    });

    // Get corresponding index for CPP values
    std::vector<State> inputs;
    std::transform(
        scalars.begin(),
        scalars.end(),
        std::back_inserter(inputs),
        [&](Val* v) {
          return fd_->recordingState(map_val_to_fd_index_.at(v));
        });

    // NOTE There is not an equivalent CPP class for python-frontend vector,
    // so we do not add it to map_val_to_fd_index_.
    Vector output = fd_->defineVector(inputs.size());
    fd_->defineRecord(new VectorRecord(
        inputs, {fd_->recordingState(output())}, DataType::Int));
    return output;
  }

  // Add Tensor value to Fusion Definition
  void handle(const TensorView* tv) final {
    // short-circuit: value already exists in FusionDefinition
    if (map_val_to_fd_index_.count(tv) > 0) {
      return;
    }

    Tensor output = fd_->defineTensor(tv->nDims());
    map_val_to_fd_index_.emplace(tv, output());

    std::vector<int64_t> shape;
    std::transform(
        tv->domain()->logical().begin(),
        tv->domain()->logical().end(),
        std::back_inserter(shape),
        [](IterDomain* id) {
          return (id->getMaybeExpandedExtent()->isConstScalar())
              ? id->getMaybeExpandedExtent()->evaluate().as<int64_t>()
              : -1;
        });

    fd_->defineRecord(new TensorRecord(
        {fd_->recordingState(output())},
        shape,
        tv->domain()->contiguity(),
        std::get<PrimDataType>(tv->dtype().type),
        tv->isCpuScalar(),
        tv->domain()->strideOrder()));
  }

  // =================================================================================
  // Utility functions

  // Create a vector for the logical domain of TensorView.
  // Used with ViewOp and ExpandOp handlers
  Vector getShape(TensorView* tv) {
    const std::vector<IterDomain*>& logical_out_domain =
        tv->domain()->logical();
    std::vector<Val*> logical_domain_extents;
    // Use expanded extent if available for IterDomain.
    std::transform(
        logical_out_domain.begin(),
        logical_out_domain.end(),
        std::back_inserter(logical_domain_extents),
        [](IterDomain* id) { return id->getMaybeExpandedExtent(); });
    return createVector(logical_domain_extents);
  }

  // Find integer index corresponding with reduction iterDomains
  std::vector<int64_t> getReductionAxes(TensorView* tv) {
    std::vector<int64_t> axes;
    const std::vector<IterDomain*>& logical_domain = tv->domain()->logical();
    for (int64_t dim : arange((int64_t)logical_domain.size())) {
      if (logical_domain.at(dim)->isReduction()) {
        axes.push_back(dim);
      }
    }
    return axes;
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
        "ops." + python::toString(e->as<ExprType>()),
        record_type,
        getFunction<ResultType, ArgTypes...>(e->as<ExprType>())));
  }

  // Map UnaryOp to python_frontend OpRecord
  void handle(const UnaryOp* uop) final {
    // short-circuit: Handle cast operation separately
    if (uop->getUnaryOpType() == UnaryOpType::Cast) {
      return handleCastOp(uop);
    }

    // Map remaining UnaryOp to python_frontend OpRecord
    if (uop->in()->isA<TensorView>()) {
      Tensor output = fd_->defineTensor(uop->out()->as<TensorView>()->nDims());
      map_val_to_fd_index_.emplace(uop->out(), output());
      handleOpRecord<nvfuser::UnaryOp>(
          uop,
          serde::RecordType::Unary_TV,
          uop->out()->as<TensorView>(),
          uop->in()->as<TensorView>());
    } else {
      Scalar output = fd_->defineScalar();
      map_val_to_fd_index_.emplace(uop->out(), output());
      handleOpRecord<nvfuser::UnaryOp>(
          uop, serde::RecordType::Unary_VAL, uop->out(), uop->in());
    }
  }

  // Map cast UnaryOp to CastOpRecord
  void handleCastOp(const Expr* op) {
    bool is_cast_op = op->isA<UnaryOp>() &&
        op->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Cast;
    NVF_ERROR(is_cast_op);

    size_t input_fd_index = map_val_to_fd_index_.at(op->input(0));

    // DataType::Index does not exist in python_frontend, so convert to
    // DataType::Int
    DataType scalar_dtype = op->output(0)->dtype();
    if (scalar_dtype == DataType::Index) {
      scalar_dtype = DataType::Int;
    }

    if (op->input(0)->isA<TensorView>()) {
      Tensor output =
          fd_->defineTensor(op->output(0)->as<TensorView>()->nDims());
      map_val_to_fd_index_.emplace(op->output(0), output());
      fd_->defineRecord(new CastOpRecord<TensorView*, TensorView*>(
          {fd_->recordingState(input_fd_index)},
          {fd_->recordingState(output())},
          "ops.cast",
          serde::RecordType::CastTv,
          static_cast<TensorView* (*)(DataType, TensorView*)>(castOp),
          std::get<PrimDataType>(scalar_dtype.type)));
    } else {
      Scalar output = fd_->defineScalar();
      map_val_to_fd_index_.emplace(op->output(0), output());
      fd_->defineRecord(new CastOpRecord<Val*, Val*>(
          {fd_->recordingState(input_fd_index)},
          {fd_->recordingState(output())},
          "ops.cast",
          serde::RecordType::CastVal,
          static_cast<Val* (*)(DataType, Val*)>(castOp),
          std::get<PrimDataType>(scalar_dtype.type)));
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

  // Map TernaryOp to python frontend
  void handle(const TernaryOp* top) final {
    bool is_in1_tv = top->in1()->isA<TensorView>();
    bool is_in2_tv = top->in2()->isA<TensorView>();
    bool is_in3_tv = top->in3()->isA<TensorView>();

    if (is_in1_tv || is_in2_tv || is_in3_tv) {
      Tensor output = fd_->defineTensor(top->out()->as<TensorView>()->nDims());
      map_val_to_fd_index_.emplace(top->out(), output());

      if (is_in1_tv && is_in2_tv && is_in3_tv) {
        handleOpRecord<nvfuser::TernaryOp>(
            top,
            serde::RecordType::Ternary_TV,
            top->out()->as<TensorView>(),
            top->in1()->as<TensorView>(),
            top->in2()->as<TensorView>(),
            top->in3()->as<TensorView>());
      } else if (is_in1_tv && is_in2_tv && !is_in3_tv) {
        handleOpRecord<nvfuser::TernaryOp>(
            top,
            serde::RecordType::Ternary_TV_TV_VAL,
            top->out()->as<TensorView>(),
            top->in1()->as<TensorView>(),
            top->in2()->as<TensorView>(),
            top->in3());
      } else if (is_in1_tv && !is_in2_tv && is_in3_tv) {
        handleOpRecord<nvfuser::TernaryOp>(
            top,
            serde::RecordType::Ternary_TV_VAL_TV,
            top->out()->as<TensorView>(),
            top->in1()->as<TensorView>(),
            top->in2(),
            top->in3()->as<TensorView>());
      } else if (is_in1_tv && !is_in2_tv && !is_in3_tv) {
        handleOpRecord<nvfuser::TernaryOp>(
            top,
            serde::RecordType::Ternary_TV_VAL_VAL,
            top->out()->as<TensorView>(),
            top->in1()->as<TensorView>(),
            top->in2(),
            top->in3());
      } else if (!is_in1_tv && is_in2_tv && is_in3_tv) {
        handleOpRecord<nvfuser::TernaryOp>(
            top,
            serde::RecordType::Ternary_VAL_TV_TV,
            top->out()->as<TensorView>(),
            top->in1(),
            top->in2()->as<TensorView>(),
            top->in3()->as<TensorView>());
      } else if (!is_in1_tv && is_in2_tv && !is_in3_tv) {
        handleOpRecord<nvfuser::TernaryOp>(
            top,
            serde::RecordType::Ternary_VAL_TV_VAL,
            top->out()->as<TensorView>(),
            top->in1(),
            top->in2()->as<TensorView>(),
            top->in3());
      } else if (!is_in1_tv && !is_in2_tv && is_in3_tv) {
        handleOpRecord<nvfuser::TernaryOp>(
            top,
            serde::RecordType::Ternary_VAL_VAL_TV,
            top->out()->as<TensorView>(),
            top->in1(),
            top->in2(),
            top->in3()->as<TensorView>());
      }
    } else {
      Scalar output = fd_->defineScalar();
      map_val_to_fd_index_.emplace(top->out(), output());
      handleOpRecord<nvfuser::TernaryOp>(
          top,
          serde::RecordType::Ternary_VAL,
          top->out(),
          top->in1(),
          top->in2(),
          top->in3());
    }
  }

  // Map ReductionOp to python frontend
  void handle(const ReductionOp* rop) final {
    TensorView* out_tv = rop->out()->as<TensorView>();

    // The min and max reduction operations expect the dtype argument to by
    // PrimDataType::Null
    PrimDataType dtype = (rop->getReductionOpType() == BinaryOpType::Min ||
                          rop->getReductionOpType() == BinaryOpType::Max)
        ? PrimDataType::Null
        : std::get<PrimDataType>(rop->out()->dtype().type);

    Tensor output = fd_->defineTensor(out_tv->nDims());
    map_val_to_fd_index_.emplace(rop->out(), output());
    fd_->defineRecord(new ReductionOpRecord(
        {fd_->recordingState(map_val_to_fd_index_.at(rop->in()))},
        {fd_->recordingState(output())},
        "ops." + python::toString(rop),
        getSerdeType(rop),
        getFunction<
            TensorView*,
            TensorView*,
            const std::vector<int64_t>&,
            bool,
            DataType>(rop),
        getReductionAxes(out_tv),
        /*keep_dim=*/false,
        dtype));
  }

  // Map WelfordOp to python frontend
  void handle(const WelfordOp* wop) final {
    NVF_ERROR(wop->initAvg()->evaluate().as<double>() == 0.0);
    NVF_ERROR(wop->initVar()->evaluate().as<double>() == 0.0);
    NVF_ERROR(wop->initN()->evaluate().as<int64_t>() == 0);

    NVF_ERROR(wop->outAvg()->isA<TensorView>());
    TensorView* out_avg_tv = wop->outAvg()->as<TensorView>();
    Tensor out_avg = fd_->defineTensor(out_avg_tv->nDims());
    map_val_to_fd_index_.emplace(wop->outAvg(), out_avg());

    NVF_ERROR(wop->outVar()->isA<TensorView>());
    TensorView* out_var_tv = wop->outVar()->as<TensorView>();
    Tensor out_var = fd_->defineTensor(out_var_tv->nDims());
    map_val_to_fd_index_.emplace(wop->outVar(), out_var());

    NVF_ERROR(wop->outN()->isA<TensorView>());
    TensorView* out_N_tv = wop->outN()->as<TensorView>();
    Tensor out_N = fd_->defineTensor(out_N_tv->nDims());
    map_val_to_fd_index_.emplace(wop->outN(), out_N());

    fd_->defineRecord(new WelfordOpRecord(
        {fd_->recordingState(map_val_to_fd_index_.at(wop->inAvg()))},
        {fd_->recordingState(out_avg()),
         fd_->recordingState(out_var()),
         fd_->recordingState(out_N())},
        getReductionAxes(out_avg_tv)));
  }

  // If input and output values share the same type, a LoadStoreOp will be
  // created instead of a CastOp.
  void handle(const LoadStoreOp* lsop) final {
    // short-circuit: lsop is a permutation.
    if (lsop->out()->isA<TensorView>() &&
        lsop->out()->as<TensorView>()->hasRoot()) {
      return handlePermute(lsop);
    }

    // Skip set unary operation
    size_t input_fid = map_val_to_fd_index_.at(lsop->in());
    map_val_to_fd_index_.emplace(lsop->out(), input_fid);
  }

  // Add DimsOpRecord to create permutation in FusionDefinition
  void handlePermute(const LoadStoreOp* lsop) {
    TensorView* out_tv = lsop->out()->as<TensorView>();

    std::optional<std::vector<int64_t>> new2old = ir_utils::computePermutation(
        out_tv->getRootDomain(), out_tv->getLogicalDomain());
    NVF_ERROR(new2old.has_value(), "Expected permutation");

    Tensor output = fd_->defineTensor(out_tv->nDims());
    map_val_to_fd_index_.emplace(out_tv, output());
    fd_->defineRecord(new DimsOpRecord<serde::RecordType::PermuteOp>(
        {fd_->recordingState(map_val_to_fd_index_.at(lsop->in()))},
        {fd_->recordingState(output())},
        std::move(new2old.value()),
        "ops.permute"));
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

  // Map SqueezeOp to python frontend
  void handle(const SqueezeOp* sop) final {
    std::vector<int64_t> squeeze_dims;
    const std::vector<bool>& is_squeeze_dims = sop->getSqueezeDimFlags();
    for (int64_t dim : arange((int64_t)is_squeeze_dims.size())) {
      if (is_squeeze_dims.at(dim)) {
        squeeze_dims.push_back(dim);
      }
    }

    // Always squeeze_expanded dimensions
    Tensor output = fd_->defineTensor(sop->out()->as<TensorView>()->nDims());
    map_val_to_fd_index_.emplace(sop->out(), output());
    fd_->defineRecord(new SqueezeOpRecord(
        {fd_->recordingState(map_val_to_fd_index_.at(sop->in()))},
        {fd_->recordingState(output())},
        squeeze_dims,
        /*squeeze_expanded=*/true));
  }

  // Map ViewOp to python frontend
  void handle(const ViewOp* vop) final {
    // Get extent's for output's logical domain
    TensorView* out_tv = vop->out()->as<TensorView>();
    Vector new_shape = getShape(out_tv);

    Tensor output = fd_->defineTensor(out_tv->nDims());
    map_val_to_fd_index_.emplace(out_tv, output());
    fd_->defineRecord(new ReshapeOpRecord(
        {fd_->recordingState(map_val_to_fd_index_.at(vop->in())),
         fd_->recordingState(new_shape())},
        {fd_->recordingState(output())}));
  }

  // Map ExpandOp to python frontend
  void handle(const ExpandOp* eop) final {
    TensorView* in_tv = eop->in()->as<TensorView>();
    TensorView* out_tv = eop->out()->as<TensorView>();
    NVF_ERROR(in_tv->nDims() == out_tv->nDims());
    Vector new_shape = getShape(out_tv);

    Tensor output = fd_->defineTensor(out_tv->nDims());
    map_val_to_fd_index_.emplace(out_tv, output());
    fd_->defineRecord(new ExpandOpRecord(
        {fd_->recordingState(map_val_to_fd_index_.at(eop->in())),
         fd_->recordingState(new_shape())},
        {fd_->recordingState(output())}));
  }

  // Map SliceOp to python frontend
  void handle(const SliceOp* sop) final {
    std::vector<nvfuser::Slice> slices = sop->getRanges();

    std::vector<Val*> start_indices;
    start_indices.reserve(slices.size());

    std::vector<Val*> stop_indices;
    stop_indices.reserve(slices.size());

    std::vector<Val*> strides;
    strides.reserve(slices.size());

    for (const nvfuser::Slice& s : slices) {
      start_indices.push_back(s.start);
      stop_indices.push_back(s.stop);
      strides.push_back(s.step);
    }

    Vector new_start = createVector(start_indices);
    Vector new_stop = createVector(stop_indices);
    Vector new_strides = createVector(strides);

    Tensor output = fd_->defineTensor(sop->out()->as<TensorView>()->nDims());
    map_val_to_fd_index_.emplace(sop->out(), output());
    fd_->defineRecord(new SliceOpRecord(
        {fd_->recordingState(map_val_to_fd_index_.at(sop->in())),
         fd_->recordingState(new_start()),
         fd_->recordingState(new_stop()),
         fd_->recordingState(new_strides())},
        {fd_->recordingState(output())},
        /*manual_normalization=*/true));
  }

  // Map PadOp to python frontend
  void handle(const PadOp* pad_op) final {
    Tensor output = fd_->defineTensor(pad_op->out()->as<TensorView>()->nDims());
    map_val_to_fd_index_.emplace(pad_op->out(), output());

    // Step 1: Get pad widths in normalized order.
    std::vector<Val*> normalized_pad_widths = pad_op->getPadWidths();
    const int64_t total_size = (int64_t)normalized_pad_widths.size();

    // Step 2: Get indices for normalized pad widths.
    std::vector<int64_t> normalized_indices(total_size);
    std::iota(normalized_indices.begin(), normalized_indices.end(), 0);

    // Step 3: Transform to indices for original pad widths
    std::vector<int64_t> original_indices;
    original_indices.reserve(normalized_indices.size());
    std::transform(
        normalized_indices.begin(),
        normalized_indices.end(),
        std::back_inserter(original_indices),
        [=](int64_t normalized_idx) {
          int64_t offset = total_size - normalized_idx;
          int64_t dim = ceilDiv(offset, 2) - 1;

          int64_t original_idx = dim * 2;
          // right pad values require an additional offset
          if (offset % 2 == 1) {
            original_idx += 1;
          }
          return original_idx;
        });

    // Step 4: Get pad widths in original order.
    std::vector<Val*> original_order_pad_widths(total_size, nullptr);
    for (int64_t normalized_idx : normalized_indices) {
      original_order_pad_widths.at(original_indices.at(normalized_idx)) =
          normalized_pad_widths.at(normalized_idx);
    }

    // Check that no pad width values are nullptr.
    NVF_ERROR(std::all_of(
        original_order_pad_widths.begin(),
        original_order_pad_widths.end(),
        [](Val* v) { return v != nullptr; }));

    Vector pad_widths = createVector(original_order_pad_widths);
    fd_->defineRecord(new PadOpRecord(
        {fd_->recordingState(map_val_to_fd_index_.at(pad_op->in())),
         fd_->recordingState(pad_widths()),
         fd_->recordingState(map_val_to_fd_index_.at(pad_op->value()))},
        {fd_->recordingState(output())}));
  }

  // Map CatOp to python frontend
  void handle(const CatOp* cat_op) final {
    Tensor output =
        fd_->defineTensor(cat_op->output(0)->as<TensorView>()->nDims());
    map_val_to_fd_index_.emplace(cat_op->output(0), output());

    std::vector<State> tensor_states;
    tensor_states.reserve(cat_op->inputs().size());
    std::transform(
        cat_op->inputs().begin(),
        cat_op->inputs().end(),
        std::back_inserter(tensor_states),
        [&](Val* v) {
          return fd_->recordingState(map_val_to_fd_index_.at(v));
        });

    fd_->defineRecord(new CatOpRecord(
        tensor_states,
        {fd_->recordingState(output())},
        cat_op->concatenatedDim(),
        /*manual_padding=*/true));
  }

  // Map RNGOp to RandomDistOpRecord
  void handle(const RNGOp* rop) final {
    TensorView* out_tv = rop->output(0)->as<TensorView>();
    Tensor output = fd_->defineTensor(out_tv->nDims());
    map_val_to_fd_index_.emplace(out_tv, output());

    std::vector<State> arg_states;

    // arg1 and arg2 are minval and maxval for uniform.
    // arg1 and arg2 are mean and std for normal.
    std::vector<Val*> params = rop->getParameters();
    if (params.empty()) {
      // Default arg1 and arg2 is (0, 1) for both uniform and normal.
      Scalar zero_value = createScalar(fusion_->zeroVal());
      Scalar one_value = createScalar(fusion_->oneVal());
      arg_states.push_back(fd_->recordingState(zero_value()));
      arg_states.push_back(fd_->recordingState(one_value()));
    } else {
      NVF_ERROR(
          params.size() == 2,
          "Expect only two parameters for uniform and normal random ops.");
      std::transform(
          params.begin(),
          params.end(),
          std::back_inserter(arg_states),
          [&](Val* v) {
            return fd_->recordingState(map_val_to_fd_index_.at(v));
          });
    }

    Vector out_shape = createVector(rop->getShape());
    arg_states.push_back(fd_->recordingState(out_shape()));

    // The philox seed and offset are optional.
    if (rop->getRNGSeedVal() != nullptr) {
      arg_states.push_back(
          fd_->recordingState(map_val_to_fd_index_.at(rop->getRNGSeedVal())));
    }
    if (rop->getRNGOffsetVal() != nullptr) {
      arg_states.push_back(
          fd_->recordingState(map_val_to_fd_index_.at(rop->getRNGOffsetVal())));
    }

    switch (rop->getRNGOpType()) {
      case RNGOpType::Uniform:
      case RNGOpType::UniformRange:
        fd_->defineRecord(
            new RandomDistOpRecord<serde::RecordType::UniformDistOp>(
                arg_states,
                {fd_->recordingState(output())},
                std::get<PrimDataType>(out_tv->dtype().type)));
        break;
      case RNGOpType::NormalStandard:
      case RNGOpType::NormalGeneral:
        fd_->defineRecord(
            new RandomDistOpRecord<serde::RecordType::NormalDistOp>(
                arg_states,
                {fd_->recordingState(output())},
                std::get<PrimDataType>(out_tv->dtype().type)));
        break;
      default:
        NVF_ERROR(false, "Unsupported RNGOpType.");
    }
  }

  // Map LinearOp to python frontend
  void handle(const LinearOp* lop) final {
    TensorView* out_tv = lop->out()->as<TensorView>();
    Tensor output = fd_->defineTensor(out_tv->nDims());
    map_val_to_fd_index_.emplace(out_tv, output());

    if (lop->bias() != nullptr) {
      fd_->defineRecord(
          new OpRecord<TensorView*, TensorView*, TensorView*, TensorView*>(
              {fd_->recordingState(map_val_to_fd_index_.at(lop->inA())),
               fd_->recordingState(map_val_to_fd_index_.at(lop->inB())),
               fd_->recordingState(map_val_to_fd_index_.at(lop->bias()))},
              {fd_->recordingState(output())},
              ("ops.linear"),
              serde::RecordType::Ternary_TV,
              static_cast<
                  TensorView* (*)(TensorView*, TensorView*, TensorView*)>(
                  linear)));
    } else {
      fd_->defineRecord(new OpRecord<TensorView*, TensorView*, TensorView*>(
          {fd_->recordingState(map_val_to_fd_index_.at(lop->inA())),
           fd_->recordingState(map_val_to_fd_index_.at(lop->inB()))},
          {fd_->recordingState(output())},
          ("ops.linear"),
          serde::RecordType::Binary_TV,
          static_cast<TensorView* (*)(TensorView*, TensorView*)>(linear)));
    }
  }

  // Map FullOp to python frontend
  void handle(const FullOp* fop) final {
    TensorView* out_tv = fop->output(0)->as<TensorView>();
    Vector tensor_shape = getShape(out_tv);

    Scalar fill_value = createScalar(fop->getFillValue());

    Tensor output = fd_->defineTensor(out_tv->nDims());
    map_val_to_fd_index_.emplace(out_tv, output());

    fd_->defineRecord(new FullOpRecord(
        {fd_->recordingState(tensor_shape()),
         fd_->recordingState(fill_value())},
        {fd_->recordingState(output())},
        std::get<PrimDataType>(out_tv->dtype().type)));
  }

  // Map IotaOp to python frontend
  void handle(const IotaOp* iop) final {
    TensorView* out_tv = iop->output(0)->as<TensorView>();
    Tensor output = fd_->defineTensor(out_tv->nDims());
    map_val_to_fd_index_.emplace(out_tv, output());

    Scalar length = createScalar(iop->length());
    Scalar start = createScalar(iop->start());
    Scalar step = createScalar(iop->step());

    fd_->defineRecord(new IotaOpRecord(
        {fd_->recordingState(length()),
         fd_->recordingState(start()),
         fd_->recordingState(step())},
        {fd_->recordingState(output())},
        std::get<PrimDataType>(iop->dtype().type)));
  }

  // Map IndexSelectOp to IndexSelectOpRecord
  void handle(const IndexSelectOp* isop) final {
    TensorView* out_tv = isop->output(0)->as<TensorView>();
    Tensor output = fd_->defineTensor(out_tv->nDims());
    map_val_to_fd_index_.emplace(out_tv, output());

    fd_->defineRecord(new IndexSelectOpRecord(
        {fd_->recordingState(map_val_to_fd_index_.at(isop->lookupTv())),
         fd_->recordingState(map_val_to_fd_index_.at(isop->indexTv()))},
        {fd_->recordingState(output())},
        isop->dim()));
  }

  // Map SelectOp to IndexSelectOpRecord
  void handle(const SelectOp* sop) final {
    TensorView* out_tv = sop->output(0)->as<TensorView>();
    Tensor output = fd_->defineTensor(out_tv->nDims());
    map_val_to_fd_index_.emplace(out_tv, output());

    fd_->defineRecord(new SelectOpRecord(
        {fd_->recordingState(map_val_to_fd_index_.at(sop->lookupTv())),
         fd_->recordingState(map_val_to_fd_index_.at(sop->input(1)))},
        {fd_->recordingState(output())},
        sop->dim()));
  }

  // Map ScatterOp to python frontend
  void handle(const ScatterOp* sop) final {
    TensorView* out_tv = sop->output(0)->as<TensorView>();
    Tensor output = fd_->defineTensor(out_tv->nDims());
    map_val_to_fd_index_.emplace(out_tv, output());

    fd_->defineRecord(new ScatterOpRecord(
        {fd_->recordingState(map_val_to_fd_index_.at(sop->selfTv())),
         fd_->recordingState(map_val_to_fd_index_.at(sop->indexTv())),
         fd_->recordingState(map_val_to_fd_index_.at(sop->srcTv()))},
        {fd_->recordingState(output())},
        sop->dim()));
  }

  // Map ArgsortOp to python frontend
  void handle(const ArgsortOp* argsortop) final {
    TensorView* out_tv = argsortop->output(0)->as<TensorView>();
    Tensor output = fd_->defineTensor(out_tv->nDims());
    map_val_to_fd_index_.emplace(out_tv, output());

    fd_->defineRecord(new ArgsortOpRecord(
        {fd_->recordingState(map_val_to_fd_index_.at(argsortop->in()))},
        {fd_->recordingState(output())},
        argsortop->dim(),
        argsortop->isDescending(),
        argsortop->isStable()));
  }

  // Map GroupedMmaOp to python frontend
  void handle(const GroupedMmaOp* gmm_op) final {
    TensorView* out_tv = gmm_op->output(0)->as<TensorView>();
    Tensor output = fd_->defineTensor(
        TensorDomain::noReductions(out_tv->getLogicalDomain()).size());
    map_val_to_fd_index_.emplace(out_tv, output());

    // TODO: add support for scale1, scale2, alpha, bias and beta
    if (gmm_op->inputs().size() == 3) {
      fd_->defineRecord(
          new OpRecord<TensorView*, TensorView*, TensorView*, TensorView*>(
              {fd_->recordingState(map_val_to_fd_index_.at(gmm_op->matrix1())),
               fd_->recordingState(map_val_to_fd_index_.at(gmm_op->matrix2())),
               fd_->recordingState(map_val_to_fd_index_.at(gmm_op->offsets()))},
              {fd_->recordingState(output())},
              ("ops.grouped_mm"),
              serde::RecordType::Ternary_TV,
              static_cast<
                  TensorView* (*)(TensorView*, TensorView*, TensorView*)>(
                  [](TensorView* matrix1,
                     TensorView* matrix2,
                     TensorView* offsets) {
                    return grouped_mm(matrix1, matrix2, offsets);
                  })));
    } else {
      fd_->defineRecord(new ScaledGroupedMmaOpRecord(
          {fd_->recordingState(map_val_to_fd_index_.at(gmm_op->matrix1())),
           fd_->recordingState(map_val_to_fd_index_.at(gmm_op->matrix2())),
           fd_->recordingState(map_val_to_fd_index_.at(gmm_op->offsets())),
           gmm_op->hasScale()
               ? fd_->recordingState(map_val_to_fd_index_.at(gmm_op->scale1()))
               : State(/*_index=*/0, /*_stype=*/serde::StateType::None),
           gmm_op->hasScale()
               ? fd_->recordingState(map_val_to_fd_index_.at(gmm_op->scale2()))
               : State(/*_index=*/0, /*_stype=*/serde::StateType::None),
           gmm_op->hasAlpha()
               ? fd_->recordingState(map_val_to_fd_index_.at(gmm_op->alpha()))
               : State(/*_index=*/0, /*_stype=*/serde::StateType::None),
           gmm_op->hasBias()
               ? fd_->recordingState(map_val_to_fd_index_.at(gmm_op->bias()))
               : State(/*_index=*/0, /*_stype=*/serde::StateType::None),
           gmm_op->hasBeta()
               ? fd_->recordingState(map_val_to_fd_index_.at(gmm_op->beta()))
               : State(/*_index=*/0, /*_stype=*/serde::StateType::None)},
          {fd_->recordingState(output())},
          std::get<PrimDataType>(out_tv->dtype().type)));
    }
  }

  // Map TopKOp to python frontend
  void handle(const TopKOp* topkop) final {
    // Create outputs for this RecordFunctor
    std::vector<State> fd_outputs;
    fd_outputs.reserve(topkop->outputs().size());
    std::transform(
        topkop->outputs().begin(),
        topkop->outputs().end(),
        std::back_inserter(fd_outputs),
        [&](Val* v) {
          NVF_ERROR(v->isA<TensorView>());
          Tensor output = fd_->defineTensor(v->as<TensorView>()->nDims());
          map_val_to_fd_index_.emplace(v, output());
          return fd_->recordingState(output());
        });

    fd_->defineRecord(new TopKOpRecord(
        {fd_->recordingState(map_val_to_fd_index_.at(topkop->in())),
         fd_->recordingState(map_val_to_fd_index_.at(topkop->k()))},
        fd_outputs,
        topkop->dim(),
        topkop->isLargest(),
        topkop->isSorted()));
  }

  // Map GatherOp to python frontend
  void handle(const GatherOp* gop) final {
    TensorView* out_tv = gop->output(0)->as<TensorView>();
    Tensor output = fd_->defineTensor(out_tv->nDims());
    map_val_to_fd_index_.emplace(out_tv, output());

    fd_->defineRecord(new GatherOpRecord(
        {fd_->recordingState(map_val_to_fd_index_.at(gop->lookupTv())),
         fd_->recordingState(map_val_to_fd_index_.at(gop->indexTv()))},
        {fd_->recordingState(output())},
        gop->dim()));
  }

  // Map MatmulOp to TensorView-Only OpRecord
  void handle(const MatmulOp* matmul_op) final {
    Tensor output =
        fd_->defineTensor(matmul_op->out()->as<TensorView>()->nDims());
    map_val_to_fd_index_.emplace(matmul_op->out(), output());

    fd_->defineRecord(new OpRecord<TensorView*, TensorView*, TensorView*>(
        {fd_->recordingState(map_val_to_fd_index_.at(matmul_op->inA())),
         fd_->recordingState(map_val_to_fd_index_.at(matmul_op->inB()))},
        {fd_->recordingState(output())},
        ("ops.matmul"),
        serde::RecordType::Binary_TV,
        static_cast<TensorView* (*)(TensorView*, TensorView*)>(matmul)));
  }

  // Map SdpaFwdOp to SdpaFwdOpRecord
  void handle(const SdpaFwdOp* sdpa_fwd_op) final {
    // Create outputs for this RecordFunctor
    std::vector<State> fd_outputs;
    fd_outputs.reserve(sdpa_fwd_op->outputs().size());
    std::transform(
        sdpa_fwd_op->outputs().begin(),
        sdpa_fwd_op->outputs().end(),
        std::back_inserter(fd_outputs),
        [&](Val* v) {
          NVF_ERROR(v->isA<TensorView>());
          Tensor output = fd_->defineTensor(v->as<TensorView>()->nDims());
          map_val_to_fd_index_.emplace(v, output());
          return fd_->recordingState(output());
        });

    State dropout_p_state = (sdpa_fwd_op->dropout_p() != nullptr)
        ? fd_->recordingState(map_val_to_fd_index_.at(sdpa_fwd_op->dropout_p()))
        : State(/*_index=*/0, /*_stype=*/serde::StateType::None);

    State is_causal_state = (sdpa_fwd_op->is_causal() != nullptr)
        ? fd_->recordingState(map_val_to_fd_index_.at(sdpa_fwd_op->is_causal()))
        : State(/*_index=*/0, /*_stype=*/serde::StateType::None);

    State scale_state = (sdpa_fwd_op->scale() != nullptr)
        ? fd_->recordingState(map_val_to_fd_index_.at(sdpa_fwd_op->scale()))
        : State(/*_index=*/0, /*_stype=*/serde::StateType::None);

    fd_->defineRecord(new SdpaFwdOpRecord(
        {fd_->recordingState(map_val_to_fd_index_.at(sdpa_fwd_op->query())),
         fd_->recordingState(map_val_to_fd_index_.at(sdpa_fwd_op->key())),
         fd_->recordingState(map_val_to_fd_index_.at(sdpa_fwd_op->value())),
         dropout_p_state,
         is_causal_state,
         scale_state},
        fd_outputs));
  }

  // Map SdpaBwdOp to SdpaBwdOpRecord
  void handle(const SdpaBwdOp* sdpa_bwd_op) final {
    // Create outputs for this RecordFunctor
    std::vector<State> fd_outputs;
    fd_outputs.reserve(sdpa_bwd_op->outputs().size());
    std::transform(
        sdpa_bwd_op->outputs().begin(),
        sdpa_bwd_op->outputs().end(),
        std::back_inserter(fd_outputs),
        [&](Val* v) {
          NVF_ERROR(v->isA<TensorView>());
          Tensor output = fd_->defineTensor(v->as<TensorView>()->nDims());
          map_val_to_fd_index_.emplace(v, output());
          return fd_->recordingState(output());
        });

    State dropout_p_state = (sdpa_bwd_op->dropout_p() != nullptr)
        ? fd_->recordingState(map_val_to_fd_index_.at(sdpa_bwd_op->dropout_p()))
        : State(/*_index=*/0, /*_stype=*/serde::StateType::None);

    State is_causal_state = (sdpa_bwd_op->is_causal() != nullptr)
        ? fd_->recordingState(map_val_to_fd_index_.at(sdpa_bwd_op->is_causal()))
        : State(/*_index=*/0, /*_stype=*/serde::StateType::None);

    State scale_state = (sdpa_bwd_op->scale() != nullptr)
        ? fd_->recordingState(map_val_to_fd_index_.at(sdpa_bwd_op->scale()))
        : State(/*_index=*/0, /*_stype=*/serde::StateType::None);

    fd_->defineRecord(new SdpaBwdOpRecord(
        {fd_->recordingState(map_val_to_fd_index_.at(sdpa_bwd_op->grad_attn())),
         fd_->recordingState(map_val_to_fd_index_.at(sdpa_bwd_op->query())),
         fd_->recordingState(map_val_to_fd_index_.at(sdpa_bwd_op->key())),
         fd_->recordingState(map_val_to_fd_index_.at(sdpa_bwd_op->value())),
         fd_->recordingState(map_val_to_fd_index_.at(sdpa_bwd_op->attn_out())),
         fd_->recordingState(map_val_to_fd_index_.at(sdpa_bwd_op->logsumexp())),
         dropout_p_state,
         is_causal_state,
         fd_->recordingState(
             map_val_to_fd_index_.at(sdpa_bwd_op->philox_seed())),
         fd_->recordingState(
             map_val_to_fd_index_.at(sdpa_bwd_op->philox_offset())),
         scale_state},
        fd_outputs));
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
