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
#include <utility>

namespace nvfuser::python {

namespace {

// Check if a type is an optional via type_traits
template <typename T>
struct is_optional : std::false_type {};

template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_optional_v = is_optional<T>::value;

// A struct to hold default values for keyword arguments.
template <typename T>
struct KeywordArgument {
  using type = T; // The data type of the argument
  std::string name;
  std::optional<T> default_value;
};

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
    if (v == nullptr) {
      return "None";
    } else if (v->isA<TensorView>()) {
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
  std::string generateOutputs(const std::vector<const nvfuser::Val*>& vec) {
    std::stringstream ss;
    for (auto&& [i, val] : enumerate(vec)) {
      if (val == nullptr) {
        ss << "_";
      } else {
        ss << toString(val);
      }
      if (i < vec.size() - 1) {
        ss << ", ";
      }
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
        "Input argument names and args must have the same size.");
    // Use std::apply to unpack tuple of arguments into a lambda. The lambda
    // contains a C++17 fold expression on a comma operator that writes each
    // argument to stringstream and increments argument position.
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

  // Generate a python list of values with string keyword arguments.
  //  * A tuple of default argument values is provided to the function.
  //  * If the default argument is optional and equal to the provided argument,
  //    then skip printing the keyword-argument pair.
  template <typename... Ds, typename... Ts>
  std::string generateNamedList(
      std::tuple<Ds...> const& default_args,
      std::tuple<Ts...> const& args) {
    NVF_ERROR(
        sizeof...(Ds) == sizeof...(Ts),
        "The default and given arguments must have the same size.");
    // This immediately-invoked generic lambda uses a C++17 fold expression
    // to emulate a loop over the tuple elements.
    //
    // 1. `std::make_index_sequence` generates a compile-time sequence of
    // indices (0, 1, 2...).
    // 2. The lambda accepts this sequence, deducing the indices into the
    // template pack `Is...`.
    // 3. A fold expression over the comma operator expands the code for each
    // index.
    // 4. A ternary operator `(condition ? ... : ...)` performs the conditional
    // logic.
    // 5. If condition is true, another comma operator
    // `(write_to_stream, increment_counters)` chains the side effects of
    // writing to the stringstream and advancing the counters.
    // 6. If condition is false, increment `printed_arg_pos` counter by zero.
    std::stringstream ss;
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      size_t printed_arg_pos = 0;
      (((!std::get<Is>(default_args).default_value.has_value() ||
         std::get<Is>(default_args).default_value.value() != std::get<Is>(args))
            ? ((ss << toString(
                    std::get<Is>(default_args).name,
                    std::get<Is>(args),
                    (printed_arg_pos > 0 ? ", " : ""))),
               ++printed_arg_pos)
            : (printed_arg_pos += 0)),
       ...);
    }(std::make_index_sequence<sizeof...(Ts)>{}); // Generate indices 0..N-1
    return ss.str();
  }

  // Generate a python operation with a list of inputs and outputs.
  void generateOperation(
      const std::string& op_name,
      const std::vector<const nvfuser::Val*>& inputs,
      const std::vector<const nvfuser::Val*>& outputs) {
    os_ << kTab;
    if (!outputs.empty()) {
      os_ << generateOutputs(outputs) << " = ";
    }
    os_ << op_name << "(" << toString(inputs, /*is_list=*/false) << ")\n";
  }

  // Generate a python operation with a list of inputs and outputs.
  // A string keyword argument is added for each input. The default_kwargs
  // argument allows skipping arguments if it isn't strictly necessary.
  template <
      typename... arg_types,
      typename... default_kwarg_types,
      typename... kwargs_types>
  void generateKwargsOperation(
      const std::string& op_name,
      const std::tuple<arg_types...>& args,
      const std::tuple<default_kwarg_types...>& default_kwargs,
      const std::tuple<kwargs_types...>& kwargs,
      const std::vector<const nvfuser::Val*>& outputs) {
    std::string kwargs_str = generateNamedList(default_kwargs, kwargs);
    constexpr bool any_arguments = sizeof...(arg_types) == 0;
    std::string connect = (any_arguments || kwargs_str.empty()) ? "" : ", ";
    os_ << kTab << generateOutputs(outputs) << " = " << op_name << "("
        << generateList(args) << connect << kwargs_str << ")\n";
  }

  // Generate a python operation with a list of inputs and outputs.
  // A string is added for each keyword argument.
  //
  // NOTES
  // ------
  //  - args and kwargs are a tuple, so it accepts a fixed set of arguments of
  //    any type at compile-time.
  //  - outputs is a vector of nvfuser values that is converted into a string.
  template <typename... arg_types, typename... kwargs_types>
  void generateKwargsOperation(
      const std::string& op_name,
      const std::tuple<arg_types...>& args,
      const std::vector<std::string>& kwargs_names,
      const std::tuple<kwargs_types...>& kwargs,
      const std::vector<const nvfuser::Val*>& outputs) {
    std::string connect = (sizeof...(arg_types) == 0) ? "" : ", ";
    os_ << kTab << generateOutputs(outputs) << " = " << op_name << "("
        << generateList(args) << connect
        << generateNamedList(kwargs_names, kwargs) << ")\n";
  }

  // Generate a python operation with a list of inputs and a single output.
  // A string is added for each keyword argument.
  //
  // NOTES
  // ------
  //  - args and kwargs are a tuple, so it accepts a fixed set of arguments of
  //    any type at compile-time.
  //  - output_name is a string.
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

  // Generate a python operation with a list of inputs and outputs.
  // A string is added for each keyword argument.
  //
  // NOTES
  // ------
  //  - args and outputs are vectors of nvfuser values that are converted into
  //    strings.
  //  - kwargs is a tuple, so it accepts a fixed set of arguments of
  //    any type at compile-time.
  template <typename... kwargs_types>
  void generateKwargsOperation(
      const std::string& op_name,
      const std::vector<Val*>& args,
      const std::vector<std::string>& kwargs_names,
      const std::tuple<kwargs_types...>& kwargs,
      const std::vector<const nvfuser::Val*>& outputs) {
    std::string connect = (args.size() == 0) ? "" : ", ";
    os_ << kTab << generateOutputs(outputs) << " = " << op_name << "("
        << toString(args) << connect << generateNamedList(kwargs_names, kwargs)
        << ")\n";
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
      // TODO: direct bindings does not support scheduled expressions.

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
          // Only apply aliasing once
          if (visited_alias_output.count(v) == 0) {
            visited_alias_output.insert(v);
            handleOutput(v->as<TensorView>(), alias_info);
          }
          // If not hide_output, then the aliased output is returned as a
          // fusion output.
          if (!alias_info.hide_output) {
            handleOutput(v->as<TensorView>());
          }
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
        std::make_tuple(),
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

  // Alias output Tensor with input tensor
  void handleOutput(const TensorView* tv, const AliasInfo& alias_info) {
    NVF_ERROR(tv != nullptr);
    printer_.generateOperation(
        "fd.add_output", {tv, alias_info.aliased_io}, {});
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

  // Map TernaryOp to python frontend
  void handle(const TernaryOp* top) final {
    NVF_ERROR(top != nullptr);
    visited_vals_.insert(top->out());
    printer_.generateOperation(
        "fd.ops." + nvfuser::python::toString(top),
        {top->in1(), top->in2(), top->in3()},
        {top->out()});
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
    std::vector<int64_t> dims = getReductionAxes(rop->out()->as<TensorView>());

    // TODO: keep_dim is always False in ReductionOp because a separate
    // BroadcastOp node exists if keep_dim is True. Detect this pattern to
    // minify the python definition.
    static const auto default_args = std::make_tuple(
        KeywordArgument<decltype(dims)>{"dims", std::nullopt},
        KeywordArgument<bool>{"keep_dim", false},
        KeywordArgument<DataType>{"dtype", DataType::Null});
    printer_.generateKwargsOperation(
        "fd.ops." + nvfuser::python::toString(rop),
        std::make_tuple(rop->in()),
        default_args,
        std::make_tuple(dims, false, dtype),
        {rop->out()});
  }

  // Map ScanOp to python frontend
  void handle(const ScanOp* sop) final {
    NVF_ERROR(sop != nullptr);
    visited_vals_.insert(sop->out());
    static const auto default_args =
        std::make_tuple(KeywordArgument<int64_t>{"dim", std::nullopt});
    printer_.generateKwargsOperation(
        "fd.ops." + toString(sop),
        std::make_tuple(sop->in()),
        default_args,
        std::make_tuple(sop->dim()),
        {sop->out()});
  }

  // Map WelfordOp to python frontend
  void handle(const WelfordOp* wop) final {
    NVF_ERROR(wop != nullptr);
    NVF_ERROR(wop->initAvg()->evaluate().as<double>() == 0.0);
    NVF_ERROR(wop->initVar()->evaluate().as<double>() == 0.0);
    NVF_ERROR(wop->initN()->evaluate().as<int64_t>() == 0);

    visited_vals_.insert(wop->outAvg());
    visited_vals_.insert(wop->outVar());
    visited_vals_.insert(wop->outN());

    static const std::vector<std::string> argument_names = {"dims"};
    printer_.generateKwargsOperation(
        "fd.ops.welford",
        std::make_tuple(wop->in()),
        argument_names,
        std::make_tuple(getReductionAxes(wop->outAvg()->as<TensorView>())),
        {wop->outAvg(), wop->outVar(), wop->outN()});
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

    static const auto default_args =
        std::make_tuple(KeywordArgument<TensorView*>{"bias", nullptr});
    printer_.generateKwargsOperation(
        "fd.ops.linear",
        std::make_tuple(lop->inA(), lop->inB()),
        default_args,
        std::make_tuple(lop->bias()),
        {lop->out()});
  }

  // Map GroupedMmaOp to python frontend
  void handle(const GroupedMmaOp* gmm_op) final {
    NVF_ERROR(gmm_op != nullptr);
    TensorView* out_tv = gmm_op->out();
    visited_vals_.insert(gmm_op->out());

    int64_t out_block_scale_size = 0;
    PrimDataType out_block_scale_dtype = DataType::BFloat16;
    bool out_gamma = false;

    TensorView* out_block_scale_tv = gmm_op->outScale();
    if (out_block_scale_tv != nullptr) {
      visited_vals_.insert(gmm_op->outScale());
      const std::vector<IterDomain*>& logical =
          out_block_scale_tv->getLogicalDomain();
      Val* block_size_extent = logical.at(logical.size() - 1)->extent();
      NVF_CHECK(
          block_size_extent->isConstInt(),
          "Block size extent needs to be a constant integer");
      out_block_scale_size = block_size_extent->evaluate().as<int64_t>();
      out_block_scale_dtype =
          std::get<PrimDataType>(out_block_scale_tv->dtype().type);
    }

    TensorView* out_gamma_tv = gmm_op->outGamma();
    if (out_gamma_tv != nullptr) {
      visited_vals_.insert(gmm_op->outGamma());
      out_gamma = true;
    }

    if (gmm_op->inputs().size() == 3) {
      printer_.generateOperation(
          "fd.ops.grouped_mm",
          {gmm_op->matrix1(), gmm_op->matrix2(), gmm_op->offsets()},
          {gmm_op->out()});
    } else {
      static const auto default_args = std::make_tuple(
          KeywordArgument<decltype(gmm_op->alpha())>{"alpha", nullptr},
          KeywordArgument<decltype(gmm_op->bias())>{"bias", nullptr},
          KeywordArgument<decltype(gmm_op->beta())>{"beta", nullptr},
          KeywordArgument<DataType>{"dtype", DataType::BFloat16},
          KeywordArgument<int64_t>{"output_block_scale_size", 0},
          KeywordArgument<DataType>{
              "output_block_scale_dtype", DataType::BFloat16},
          KeywordArgument<bool>{"output_gamma", false});
      printer_.generateKwargsOperation(
          "fd.ops.grouped_mm",
          std::make_tuple(
              gmm_op->matrix1(),
              gmm_op->matrix2(),
              gmm_op->offsets(),
              gmm_op->scale1(),
              gmm_op->scale2()),
          default_args,
          std::make_tuple(
              gmm_op->alpha(),
              gmm_op->bias(),
              gmm_op->beta(),
              out_tv->dtype(),
              out_block_scale_size,
              out_block_scale_dtype,
              out_gamma),
          {gmm_op->out(), out_block_scale_tv, out_gamma_tv});
    }
  }

  // Map ScaledMmaOp to python frontend
  void handle(const ScaledMmaOp* smm_op) final {
    NVF_ERROR(smm_op != nullptr);
    TensorView* out_tv = smm_op->out();
    visited_vals_.insert(smm_op->out());

    int64_t out_block_scale_size = 0;
    PrimDataType out_block_scale_dtype = DataType::BFloat16;
    bool out_gamma = false;

    TensorView* out_block_scale_tv = smm_op->outScale();
    if (out_block_scale_tv != nullptr) {
      visited_vals_.insert(smm_op->outScale());
      const std::vector<IterDomain*>& logical =
          out_block_scale_tv->getLogicalDomain();
      Val* block_size_extent = logical.at(logical.size() - 1)->extent();
      NVF_CHECK(
          block_size_extent->isConstInt(),
          "Block size extent needs to be a constant integer");
      out_block_scale_size = block_size_extent->evaluate().as<int64_t>();
      out_block_scale_dtype =
          std::get<PrimDataType>(out_block_scale_tv->dtype().type);
    }

    TensorView* out_gamma_tv = smm_op->outGamma();
    if (out_gamma_tv != nullptr) {
      visited_vals_.insert(smm_op->outGamma());
      out_gamma = true;
    }

    static const auto default_args = std::make_tuple(
        KeywordArgument<decltype(smm_op->alpha())>{"alpha", nullptr},
        KeywordArgument<decltype(smm_op->bias())>{"bias", nullptr},
        KeywordArgument<decltype(smm_op->beta())>{"beta", nullptr},
        KeywordArgument<DataType>{"dtype", DataType::BFloat16},
        KeywordArgument<int64_t>{"output_block_scale_size", 0},
        KeywordArgument<DataType>{
            "output_block_scale_dtype", DataType::BFloat16},
        KeywordArgument<bool>{"output_gamma", false});
    printer_.generateKwargsOperation(
        "fd.ops.scaled_mm",
        std::make_tuple(
            smm_op->matrix1(),
            smm_op->matrix2(),
            smm_op->scale1(),
            smm_op->scale2()),
        default_args,
        std::make_tuple(
            smm_op->alpha(),
            smm_op->bias(),
            smm_op->beta(),
            out_tv->dtype(),
            out_block_scale_size,
            out_block_scale_dtype,
            out_gamma),
        {smm_op->out(), out_block_scale_tv, out_gamma_tv});
  }

  // Map SqueezeOp to python frontend
  void handle(const SqueezeOp* sop) final {
    NVF_ERROR(sop != nullptr);
    visited_vals_.insert(sop->out());

    const std::vector<bool>& is_squeeze_dims = sop->getSqueezeDimFlags();
    auto filter_range = std::views::iota(0UL, is_squeeze_dims.size()) |
        std::views::filter([&is_squeeze_dims](int64_t dim) {
                          return is_squeeze_dims.at(dim);
                        });
    std::vector<int64_t> squeeze_dims(filter_range.begin(), filter_range.end());

    TensorView* in_tv = sop->in()->as<TensorView>();
    NVF_ERROR(in_tv != nullptr);

    // TODO: Use std::ranges::zip_view AND std::ranges::any_of with cpp23
    bool squeeze_expanded = false;
    for (auto [squeeze_dim, id] :
         zip(is_squeeze_dims, in_tv->getLogicalDomain())) {
      if (!squeeze_dim) {
        continue;
      }
      squeeze_expanded |= (id->isBroadcast() && id->hasExpandedExtent());
    }

    static const auto default_args = std::make_tuple(
        KeywordArgument<decltype(squeeze_dims)>{"dims", std::nullopt},
        KeywordArgument<bool>{"squeeze_expanded", false});
    printer_.generateKwargsOperation(
        "fd.ops.squeeze",
        std::make_tuple(sop->in()),
        default_args,
        std::make_tuple(squeeze_dims, squeeze_expanded),
        {sop->out()});
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
      dispatch(v);
    });
    visited_vals_.insert(vop->out());
    printer_.generateKwargsOperation(
        "fd.ops.reshape",
        std::make_tuple(vop->in()),
        reshape_argument_names,
        std::make_tuple(new_shape),
        {vop->out()});
  }

  // Map ExpandOp to python frontend
  void handle(const ExpandOp* eop) final {
    NVF_ERROR(eop != nullptr);
    TensorView* in_tv = eop->in()->as<TensorView>();
    TensorView* out_tv = eop->out()->as<TensorView>();
    NVF_ERROR(in_tv->nDims() == out_tv->nDims());
    std::vector<Val*> shape = getShape(out_tv);

    static const std::vector<std::string> expand_argument_names = {"shape"};
    // Add CPP values to Fusion Definition if necessary
    std::for_each(
        shape.begin(), shape.end(), [this](const Val* v) { dispatch(v); });
    visited_vals_.insert(eop->out());
    printer_.generateKwargsOperation(
        "fd.ops.expand",
        std::make_tuple(eop->in()),
        expand_argument_names,
        std::make_tuple(shape),
        {eop->out()});
  }

  // Map SliceOp to python frontend
  void handle(const SliceOp* sop) final {
    NVF_ERROR(sop != nullptr);
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

    visited_vals_.insert(sop->out());
    // Since the normalization operations are expressed in the Fusion IR,
    // manual_normalization argument is always true and default arguments is not
    // used here.
    static const std::vector<std::string> slice_argument_names = {
        "start_indices", "end_indices", "strides", "manual_normalization"};
    printer_.generateKwargsOperation(
        "fd.ops.slice",
        std::make_tuple(sop->in()),
        slice_argument_names,
        std::make_tuple(
            start_indices,
            stop_indices,
            strides,
            /*manual_normalization=*/true),
        {sop->out()});
  }

  // Map PadOp to python frontend
  void handle(const PadOp* pad_op) final {
    NVF_ERROR(pad_op != nullptr);

    // Step 1: Get pad widths in normalized order.
    std::vector<Val*> normalized_pad_widths = pad_op->getPadWidths();
    int64_t total_size = (int64_t)normalized_pad_widths.size();

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

    visited_vals_.insert(pad_op->out());
    static const auto default_args = std::make_tuple(
        KeywordArgument<decltype(original_order_pad_widths)>{
            "pad_widths", std::nullopt},
        KeywordArgument<Val*>{"value", nullptr});
    printer_.generateKwargsOperation(
        "fd.ops.pad",
        std::make_tuple(pad_op->in()),
        default_args,
        std::make_tuple(original_order_pad_widths, pad_op->value()),
        {pad_op->out()});
  }

  // Map CatOp to python frontend
  void handle(const CatOp* cat_op) final {
    NVF_ERROR(cat_op != nullptr);

    visited_vals_.insert(cat_op->output(0));
    // Since the normalization operations are expressed in the Fusion IR,
    // manual_normalization argument is always true and default arguments is not
    // used here.
    static const std::vector<std::string> cat_argument_names = {
        "dim", "manual_padding"};
    printer_.generateKwargsOperation(
        "fd.ops.cat",
        cat_op->inputs(),
        cat_argument_names,
        std::make_tuple(cat_op->concatenatedDim(), /*manual_padding=*/true),
        {cat_op->output(0)});
  }

  // If input and output values share the same type, a LoadStoreOp will be
  // created instead of a CastOp.
  void handle(const LoadStoreOp* lsop) final {
    // TODO short-circuit: lsop is a permutation.
    if (lsop->out()->isA<TensorView>() &&
        lsop->out()->as<TensorView>()->hasRoot()) {
      return handlePermute(lsop);
    }

    NVF_ERROR(
        lsop->in()->dtype() == lsop->out()->dtype(),
        "Expected the dtype for input and output to be the same");
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

  // Map FullOp to python frontend
  void handle(const FullOp* fop) final {
    NVF_ERROR(fop != nullptr);
    TensorView* out_tv = fop->output(0)->as<TensorView>();
    visited_vals_.insert(out_tv);

    // Fill value can be dynamic so create it
    dispatch(fop->getFillValue());

    static const std::vector<std::string> argument_names = {
        "shape", "fill_value", "dtype"};
    printer_.generateKwargsOperation(
        "fd.ops.full",
        std::make_tuple(),
        argument_names,
        std::make_tuple(getShape(out_tv), fop->getFillValue(), out_tv->dtype()),
        {out_tv});
  }

  // Map IotaOp to python frontend
  void handle(const IotaOp* iop) final {
    NVF_ERROR(iop != nullptr);
    TensorView* out_tv = iop->output(0)->as<TensorView>();
    visited_vals_.insert(out_tv);

    dispatch(iop->length());
    dispatch(iop->start());
    dispatch(iop->step());

    static const auto default_args = std::make_tuple(
        KeywordArgument<decltype(iop->length())>{"length", std::nullopt},
        KeywordArgument<decltype(iop->start())>{"start", nullptr},
        KeywordArgument<decltype(iop->step())>{"step", nullptr},
        KeywordArgument<DataType>{"dtype", DataType::Int});
    printer_.generateKwargsOperation(
        "fd.ops.iota",
        std::make_tuple(),
        default_args,
        std::make_tuple(iop->length(), iop->start(), iop->step(), iop->dtype()),
        {out_tv});
  }

  // Map IndexSelectOp to IndexSelectOpRecord
  void handle(const IndexSelectOp* isop) final {
    NVF_ERROR(isop != nullptr);
    TensorView* out_tv = isop->output(0)->as<TensorView>();
    visited_vals_.insert(out_tv);
    static const std::vector<std::string> argument_names = {"dim"};
    printer_.generateKwargsOperation(
        "fd.ops.index_select",
        std::make_tuple(isop->lookupTv(), isop->indexTv()),
        argument_names,
        std::make_tuple(isop->dim()),
        {out_tv});
  }

  // Map SelectOp to IndexSelectOpRecord
  void handle(const SelectOp* sop) final {
    NVF_ERROR(sop != nullptr);
    TensorView* out_tv = sop->output(0)->as<TensorView>();
    visited_vals_.insert(out_tv);
    static const std::vector<std::string> argument_names = {"dim"};
    printer_.generateKwargsOperation(
        "fd.ops.select",
        std::make_tuple(sop->lookupTv(), sop->input(1)),
        argument_names,
        std::make_tuple(sop->dim()),
        {out_tv});
  }

  // Map ScatterOp to python frontend
  void handle(const ScatterOp* sop) final {
    NVF_ERROR(sop != nullptr);
    TensorView* out_tv = sop->output(0)->as<TensorView>();
    visited_vals_.insert(out_tv);
    static const std::vector<std::string> argument_names = {"dim"};
    printer_.generateKwargsOperation(
        "fd.ops.scatter",
        std::make_tuple(sop->selfTv(), sop->indexTv(), sop->srcTv()),
        argument_names,
        std::make_tuple(sop->dim()),
        {out_tv});
  }

  // Map GatherOp to python frontend
  void handle(const GatherOp* gop) final {
    NVF_ERROR(gop != nullptr);
    TensorView* out_tv = gop->output(0)->as<TensorView>();
    visited_vals_.insert(out_tv);
    static const std::vector<std::string> argument_names = {"dim"};
    printer_.generateKwargsOperation(
        "fd.ops.gather",
        std::make_tuple(gop->lookupTv(), gop->indexTv()),
        argument_names,
        std::make_tuple(gop->dim()),
        {out_tv});
  }

  // Map TopKOp to python frontend
  void handle(const TopKOp* topkop) final {
    NVF_ERROR(topkop != nullptr);
    visited_vals_.insert(topkop->output(0));
    visited_vals_.insert(topkop->output(1));
    static const auto default_args = std::make_tuple(
        KeywordArgument<decltype(topkop->dim())>{"dim", -1},
        KeywordArgument<bool>{"largest", true},
        KeywordArgument<bool>{"sorted", false});
    printer_.generateKwargsOperation(
        "fd.ops.topk",
        std::make_tuple(topkop->in(), topkop->k()),
        default_args,
        std::make_tuple(topkop->dim(), topkop->isLargest(), topkop->isSorted()),
        std::vector<const nvfuser::Val*>{topkop->output(0), topkop->output(1)});
  }

  // Map ArgsortOp to python frontend
  void handle(const ArgsortOp* argsortop) final {
    NVF_ERROR(argsortop != nullptr);

    TensorView* out_tv = argsortop->output(0)->as<TensorView>();
    visited_vals_.insert(out_tv);
    static const auto default_args = std::make_tuple(
        KeywordArgument<decltype(argsortop->dim())>{"dim", std::nullopt},
        KeywordArgument<bool>{"descending", false},
        KeywordArgument<bool>{"stable", false});
    printer_.generateKwargsOperation(
        "fd.ops.argsort",
        std::make_tuple(argsortop->in()),
        default_args,
        std::make_tuple(
            argsortop->dim(), argsortop->isDescending(), argsortop->isStable()),
        {out_tv});
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
