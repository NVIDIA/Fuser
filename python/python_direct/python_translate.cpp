// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <python_utils.h>
#include <translation_names.h>

#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/container.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <type.h>
#include <utils.h>

#include <ranges>
#include <type_traits>

namespace nvfuser::python {

namespace {

// Check if a type is an optional via type_traits
template <typename T>
struct is_optional : std::false_type {};

template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_optional_v = is_optional<T>::value;

class PythonPrinter {
 public:
  PythonPrinter(std::ostream& os) : os_(os) {}

  // Generate a python string for a string value.
  std::string toString(const std::string& s) {
    return s;
  }

  // Generate a python string for a boolean value.
  std::string toString(bool b) {
    return b ? "True" : "False";
  }

  // Generate a python string for an int64_t value.
  std::string toString(int64_t i) {
    return std::to_string(i);
  }

  // Generate a python string for an size_t value.
  std::string toString(size_t i) {
    return std::to_string(i);
  }

  // Generate a python string for a complex double value.
  std::string toString(std::complex<double> c) {
    std::stringstream ss;
    ss << std::showpoint << std::real(c) << "+" << std::showpoint
       << std::imag(c) << "j";
    return ss.str();
  }

  // Generate a python string for a double value.
  std::string toString(double d) {
    if (std::isinf(d)) {
      if (std::signbit(d)) {
        return "float(\"-inf\")";
      } else {
        return "float(\"inf\")";
      }
    } else if (std::isnan(d)) {
      return "float(\"nan\")";
    } else {
      std::stringstream ss;
      ss << std::showpoint << d;
      return ss.str();
    }
  }

  // Generate a python string for a Datatype.
  std::string toString(DataType dtype) {
    return dtypeToPyString(std::get<PrimDataType>(dtype.type));
  }

  // Generate a python string for a PolymorphicValue with simple types.
  std::string toString(const PolymorphicValue& pv) {
    if (pv.is<bool>()) {
      return toString(pv.as<bool>());
    } else if (pv.is<int64_t>()) {
      return toString(pv.as<int64_t>());
    } else if (pv.is<std::complex<double>>()) {
      return toString(pv.as<std::complex<double>>());
    } else if (pv.is<double>()) {
      return toString(pv.as<double>());
    } else if (pv.is<std::monostate>()) {
      return "None";
    } else {
      NVF_THROW("Unsupported PolymorphicValue type");
    }
  }

  // Generate a unique name for a Val. Map val to name to track Val's lifetime.
  std::string toString(const nvfuser::Val* v) {
    std::stringstream ss;
    if (v->isA<TensorView>()) {
      ss << "tv" << v->name();
    } else {
      ss << "c" << v->name();
    }
    return ss.str();
  }

  // Generate a python string for an optional value.
  template <typename T>
  std::string toString(std::optional<T> optional, bool skip_none = true) {
    if (optional.has_value()) {
      return toString(optional.value());
    } else if (!skip_none) {
      return "None";
    } else {
      return "";
    }
  }

  // Generate a python string for a keyword argument.
  template <typename T>
  std::string toString(
      const std::string& name,
      T value,
      const std::string& separator) {
    std::string result = toString(value);
    if (result.empty()) {
      return "";
    }
    return separator + name + "=" + result;
  }

  // Generate a python list of values.
  template <typename T>
  std::string toString(const std::vector<T>& vec, bool is_list = true) {
    std::stringstream ss;
    if (is_list) {
      ss << "[";
    }
    for (auto&& [i, val] : enumerate(vec)) {
      if constexpr (is_optional_v<T>) {
        ss << toString(val, /*skip_none=*/false);
      } else {
        ss << toString(val);
      }
      if (i < vec.size() - 1) {
        ss << ", ";
      }
    }
    if (is_list) {
      ss << "]";
    }
    return ss.str();
  }

  // Generate a python list of values.
  template <typename... Ts>
  std::string generateList(std::tuple<Ts...> const& args) {
    if (sizeof...(Ts) == 0) {
      return "";
    }
    std::stringstream ss;
    std::apply(
        [&](Ts const&... tuple_args) {
          size_t i = 0;
          (((ss << (i > 0 ? ", " : "") << toString(tuple_args)), ++i), ...);
        },
        args);
    return ss.str();
  }

  // Generate a python list of values with string keyword arguments.
  template <typename... Ts>
  std::string generateNamedList(
      const std::vector<std::string>& argument_names,
      std::tuple<Ts...> const& args) {
    NVF_ERROR(
        argument_names.size() == sizeof...(Ts),
        "Input argument names and inputs must have the same size.");
    std::stringstream ss;
    std::apply(
        [this, &ss, &argument_names](Ts const&... tuple_args) {
          size_t i = 0;
          (((ss << toString(
                 argument_names[i], tuple_args, (i > 0 ? ", " : ""))),
            ++i),
           ...);
        },
        args);
    return ss.str();
  }

  // Generate a python operation with a list of inputs and outputs.
  void generateOperation(
      const std::string& op_name,
      const std::vector<const nvfuser::Val*>& inputs,
      const std::vector<const nvfuser::Val*>& outputs) {
    os_ << kTab;
    if (!outputs.empty()) {
      os_ << toString(outputs, /*is_list=*/false) << " = ";
    }
    os_ << op_name << "(" << toString(inputs, /*is_list=*/false) << ")\n";
  }

  // Generate a python operation with a list of inputs and outputs.
  // A string keyword argument is added for each input.
  template <typename... arg_types, typename... kwargs_types>
  void generateKwargsOperation(
      const std::string& op_name,
      const std::tuple<arg_types...>& args,
      const std::vector<std::string>& kwargs_names,
      const std::tuple<kwargs_types...>& kwargs,
      const std::vector<const nvfuser::Val*>& outputs) {
    std::string connect = (sizeof...(arg_types) == 0) ? "" : ", ";
    os_ << kTab << toString(outputs, /*is_list=*/false) << " = " << op_name
        << "(" << generateList(args) << connect
        << generateNamedList(kwargs_names, kwargs) << ")\n";
  }

  template <typename... arg_types, typename... kwargs_types>
  void generateKwargsOperation(
      const std::string& op_name,
      const std::tuple<arg_types...>& args,
      const std::vector<std::string>& kwargs_names,
      const std::tuple<kwargs_types...>& kwargs,
      const std::string& output_name) {
    std::string connect = (sizeof...(arg_types) == 0) ? "" : ", ";
    os_ << kTab << output_name << " = " << op_name << "(" << generateList(args)
        << connect << generateNamedList(kwargs_names, kwargs) << ")\n";
  }

  // Generate a python definition for a FusionDefinition.
  void generateFusionDefinition() {
    os_ << "def nvfuser_fusion(fd : FusionDefinition) -> None :\n";
  }

 private:
  //! The stream to print the python function to.
  std::ostream& os_;
  //! Indentation for python code.
  static constexpr const char* kTab = "    ";
};

// PythonTranslator converts CPP Fusion to an equivalent python definition.
//
// How to add support for an expression not yet overriden by FusionTranslator?
//  1. Create handle function for expression.
//     a. void handle(const SomeOp* op) final
//  2. Create output string for Statement.
//  3. If input argument already exists, map expression's input values to
//     their string names.
//     a. map_val_to_name_.at(op->inputs(...))
//  4. If input argument is a vector, use createVector function.
//  5. If input argument is a scalar constant, use createScalar function.
//  6. Get function name for operation.
//  7. Add CPP Val and output string pair to map_val_to_name_.
//  8. Create string for operation.
//     a. output = operation(inputs...)
class PythonTranslator : public OptInConstDispatch {
 public:
  // Returns a map from the values in the CPP fusion to its corresponding
  // FusionDefinition State index.
  static void print(std::ostream& os, Fusion* fusion) {
    PythonTranslator translator(os, fusion);
    translator.translate();
  }

 private:
  PythonTranslator(std::ostream& os, Fusion* fusion)
      : printer_(os), fusion_(fusion) {}

  bool isScheduledTensorView(TensorView* tv) const {
    NVF_ERROR(tv != nullptr);
    const std::vector<IterDomain*>& logical = tv->domain()->logical();
    const std::vector<IterDomain*>& loop = tv->domain()->loop();
    // short-circuit: check same length
    if (logical.size() != loop.size()) {
      return true;
    }

    for (size_t idx : c10::irange(logical.size())) {
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
          return v->definition() == nullptr || visited_vals_.count(v) > 0;
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
    // short-circuit: Found a view operation without all its shape dependencies.
    if (e->isA<ViewOp>() && !checkViewShapeDependency(e->as<ViewOp>())) {
      return false;
    }
    return std::all_of(
        e->inputs().begin(), e->inputs().end(), [&](const Val* v) {
          return visited_vals_.count(v) > 0;
        });
  }

  void translate() {
    printer_.generateFusionDefinition();

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

      // TODO: short-circuit: skip Split and Merge expressions created by
      // Reshape
      // TODO: short-circuit: skip Resize expressions created by Slice

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

      // Create string representation given inputs, outputs, and attributes.
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
          NVF_THROW("Not implemented");
          break;
        }
        default:
          NVF_THROW("Unsupported AllocationType");
      }
    }
  }

  // =================================================================================
  // Filter Functions

  // Gather all TensorViews and FusionDefinition indices
  std::vector<const nvfuser::Val*> tensors() {
    std::vector<const nvfuser::Val*> tensors;
    std::copy_if(
        visited_vals_.begin(),
        visited_vals_.end(),
        std::back_inserter(tensors),
        [](const nvfuser::Val* v) { return v->isA<TensorView>(); });
    return tensors;
  }

  // =================================================================================

  // Create scalar for given nvfuser value. The nvfuser value must not already
  // exist and have a definition. It can be a fusion input, a constant, or a
  // tensor's extent.
  void handle(const Val* v) final {
    NVF_ERROR(v != nullptr);
    // short-circuit: scalar definition has a definition
    if (v->definition() != nullptr) {
      return;
    }
    // short-circuit: value already exists in FusionDefinition
    if (visited_vals_.count(v) > 0) {
      return;
    }
    visited_vals_.insert(v);

    // Since scalars can come from TensorView dimension sizes, search through
    // all TensorViews for an iterDomain whose extent matches the desired
    // value and then use size op.
    for (const nvfuser::Val* tv_val : tensors()) {
      const TensorView* tv = tv_val->as<TensorView>();

      // Get extents for each IterDomain
      std::vector<IterDomain*> filtered_logical_domain =
          TensorDomain::noReductions(tv->domain()->logical());
      std::vector<Val*> extents;
      extents.reserve(filtered_logical_domain.size());
      std::transform(
          filtered_logical_domain.begin(),
          filtered_logical_domain.end(),
          std::back_inserter(extents),
          [](IterDomain* id) { return id->getMaybeExpandedExtent(); });

      // Check if value matches iterdomain extent
      auto iter = std::find(extents.begin(), extents.end(), v);
      if (iter == extents.end()) {
        continue;
      }

      int64_t dim = std::distance(extents.begin(), iter);
      static const std::vector<std::string> argument_names = {"dim"};
      printer_.generateKwargsOperation(
          "fd.ops.size",
          std::make_tuple(tv),
          argument_names,
          std::make_tuple(dim),
          {v});
      return;
    }

    // DataType::Index does not exist in python_frontend, so convert to
    // DataType::Int
    DataType scalar_dtype =
        (v->dtype() == DataType::Index) ? DataType::Int : v->dtype();

    static const std::vector<std::string> argument_names = {"dtype"};
    printer_.generateKwargsOperation(
        "fd.define_scalar",
        std::make_tuple(v->value()),
        argument_names,
        std::make_tuple(scalar_dtype),
        {v});
  }

  // Add Tensor value to Fusion Definition
  void handle(const TensorView* tv) final {
    NVF_ERROR(tv != nullptr);
    // short-circuit: value already exists in FusionDefinition
    if (visited_vals_.count(tv) > 0) {
      return;
    }
    visited_vals_.insert(tv);

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

    const std::vector<int64_t>& stride_order = tv->domain()->strideOrder();

    static const std::vector<std::string> argument_names = {
        "shape", "contiguity", "dtype", "is_cpu", "stride_order"};
    printer_.generateKwargsOperation(
        "fd.define_tensor",
        {},
        argument_names,
        std::make_tuple(
            shape,
            tv->domain()->contiguity(),
            tv->dtype(),
            tv->isCpuScalar(),
            (stride_order.empty()) ? std::nullopt
                                   : std::make_optional(stride_order)),
        {tv});
  }

  // =================================================================================
  // Utility functions

  // Create a vector for the logical domain of TensorView.
  // Used with ViewOp and ExpandOp handlers
  std::vector<Val*> getShape(TensorView* tv) {
    const std::vector<IterDomain*>& logical_out_domain =
        tv->domain()->logical();
    std::vector<Val*> logical_domain_extents;
    // Use expanded extent if available for IterDomain.
    std::transform(
        logical_out_domain.begin(),
        logical_out_domain.end(),
        std::back_inserter(logical_domain_extents),
        [](IterDomain* id) { return id->getMaybeExpandedExtent(); });
    return logical_domain_extents;
  }

  // Find integer index corresponding with reduction iterDomains
  std::vector<int64_t> getReductionAxes(TensorView* tv) {
    std::vector<int64_t> axes;
    const std::vector<IterDomain*>& logical_domain = tv->domain()->logical();
    for (int64_t dim : c10::irange((int64_t)logical_domain.size())) {
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
    NVF_ERROR(tv != nullptr);
    printer_.generateOperation("fd.add_output", {tv}, {});
  }

  // =================================================================================
  // Map CPP Expression classes to corresponding RecordFunctors in
  // python_frontend

  // Map UnaryOp to python_frontend OpRecord
  void handle(const UnaryOp* uop) final {
    NVF_ERROR(uop != nullptr);
    // short-circuit: Handle cast operation separately
    if (uop->getUnaryOpType() == UnaryOpType::Cast) {
      return handleCastOp(uop);
    }

    // Map remaining UnaryOp to python_frontend OpRecord
    visited_vals_.insert(uop->out());
    printer_.generateOperation(
        "fd.ops." + nvfuser::python::toString(uop), {uop->in()}, {uop->out()});
  }

  // Map cast UnaryOp to CastOpRecord
  void handleCastOp(const UnaryOp* uop) {
    NVF_ERROR(uop->getUnaryOpType() == UnaryOpType::Cast);
    visited_vals_.insert(uop->out());

    // DataType::Index does not exist in python_frontend, so convert to
    // DataType::Int
    DataType scalar_dtype = uop->out()->dtype();
    if (scalar_dtype == DataType::Index) {
      scalar_dtype = DataType::Int;
    }

    static const std::vector<std::string> argument_names = {"dtype"};
    printer_.generateKwargsOperation(
        "fd.ops.cast",
        std::make_tuple(uop->in()),
        argument_names,
        std::make_tuple(scalar_dtype),
        {uop->out()});
  }

  // Map BinaryOp to python_frontend OpRecord
  void handle(const BinaryOp* bop) final {
    NVF_ERROR(bop != nullptr);
    if (visited_vals_.count(bop->out()) > 0) {
      return;
    }
    visited_vals_.insert(bop->out());
    printer_.generateOperation(
        "fd.ops." + nvfuser::python::toString(bop),
        {bop->lhs(), bop->rhs()},
        {bop->out()});
  }

  // Map ReductionOp to python frontend
  void handle(const ReductionOp* rop) final {
    NVF_ERROR(rop != nullptr);
    NVF_ERROR(rop->out()->isA<TensorView>());
    visited_vals_.insert(rop->out());

    // The min and max reduction operations expect the dtype argument to by
    // PrimDataType::Null
    DataType dtype = (rop->getReductionOpType() == BinaryOpType::Min ||
                      rop->getReductionOpType() == BinaryOpType::Max)
        ? DataType::Null
        : rop->out()->dtype();

    static const std::vector<std::string> argument_names = {
        "dims", "keep_dim", "dtype"};
    printer_.generateKwargsOperation(
        "fd.ops." + nvfuser::python::toString(rop),
        std::make_tuple(rop->in()),
        argument_names,
        std::make_tuple(
            getReductionAxes(rop->out()->as<TensorView>()), false, dtype),
        {rop->out()});
  }

  // Add Broadcast operation to FusionDefinition
  void handle(const BroadcastOp* bcast_op) final {
    NVF_ERROR(bcast_op != nullptr);
    visited_vals_.insert(bcast_op->out());
    static const std::vector<std::string> broadcast_argument_names = {
        "is_broadcast_dim"};
    printer_.generateKwargsOperation(
        "fd.ops.broadcast",
        std::make_tuple(bcast_op->in()),
        broadcast_argument_names,
        std::make_tuple(bcast_op->getBroadcastDimFlags()),
        {bcast_op->out()});
  }

  // Map MatmulOp to TensorView-Only OpRecord
  void handle(const MatmulOp* matmul_op) final {
    NVF_ERROR(matmul_op != nullptr);
    visited_vals_.insert(matmul_op->out());

    printer_.generateOperation(
        "fd.ops.matmul",
        {matmul_op->inA(), matmul_op->inB()},
        {matmul_op->out()});
  }

  // Map LinearOp to python frontend
  void handle(const LinearOp* lop) final {
    NVF_ERROR(lop != nullptr);
    visited_vals_.insert(lop->out());

    if (lop->bias() != nullptr) {
      printer_.generateOperation(
          "fd.ops.linear", {lop->inA(), lop->inB(), lop->bias()}, {lop->out()});
    } else {
      printer_.generateOperation(
          "fd.ops.linear", {lop->inA(), lop->inB()}, {lop->out()});
    }
  }

  // Map ViewOp to python frontend
  void handle(const ViewOp* vop) final {
    NVF_ERROR(vop != nullptr);

    // Get extent's for output's logical domain
    TensorView* out_tv = vop->out()->as<TensorView>();
    std::vector<Val*> new_shape = getShape(out_tv);

    // TODO Check if new_shape is a vector of symbolic fusion inputs
    // TODO Use define_vector to create more pythonic syntax

    // Add CPP values to Fusion Definition if necessary
    static const std::vector<std::string> reshape_argument_names = {
        "new_shape"};
    std::for_each(new_shape.begin(), new_shape.end(), [this](const Val* v) {
      OptOutConstDispatch::dispatch(v);
    });
    visited_vals_.insert(vop->out());
    printer_.generateKwargsOperation(
        "fd.ops.reshape",
        std::make_tuple(vop->in()),
        reshape_argument_names,
        std::make_tuple(new_shape),
        {vop->out()});
  }

  // If input and output values share the same type, a LoadStoreOp will be
  // created instead of a CastOp.
  void handle(const LoadStoreOp* lsop) final {
    // TODO short-circuit: lsop is a permutation.
    if (lsop->out()->isA<TensorView>() &&
        lsop->out()->as<TensorView>()->hasRoot()) {
      return handlePermute(lsop);
    }

    visited_vals_.insert(lsop->out());
    static const std::vector<std::string> argument_names = {"dtype"};
    printer_.generateKwargsOperation(
        "fd.ops.cast",
        std::make_tuple(lsop->in()),
        argument_names,
        std::make_tuple(lsop->out()->dtype()),
        {lsop->out()});
  }

  void handlePermute(const LoadStoreOp* lsop) {
    TensorView* out_tv = lsop->out()->as<TensorView>();

    std::optional<std::vector<int64_t>> new2old = ir_utils::computePermutation(
        out_tv->getRootDomain(), out_tv->getLogicalDomain());
    NVF_ERROR(new2old.has_value(), "Expected permutation");

    visited_vals_.insert(lsop->out());
    static const std::vector<std::string> argument_names = {"dims"};
    printer_.generateKwargsOperation(
        "fd.ops.permute",
        std::make_tuple(lsop->in()),
        argument_names,
        std::make_tuple(new2old.value()),
        {lsop->out()});
  }

 private:
  //! Convert CPP values to python syntax.
  PythonPrinter printer_;
  //! The reference CPP fusion to be translated.
  Fusion* fusion_ = nullptr;
  //! Set of NvFuser Val's created in the Fusion.
  std::unordered_set<const nvfuser::Val*> visited_vals_;
};

} // namespace

std::string translateFusion(nvfuser::Fusion* f) {
  std::stringstream ss;
  PythonTranslator::print(ss, f);
  return ss.str();
}

} // namespace nvfuser::python
