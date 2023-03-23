// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <python_frontend/python_bindings.h>

#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <instrumentation.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <ops/arith.h>
#include <ops/composite.h>
#include <python_frontend/fusion_cache.h>
#include <python_frontend/fusion_definition.h>
#include <python_frontend/fusion_record.h>
#include <python_frontend/python_bindings.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <complex>
#include <iostream>
#include <tuple>

namespace nvfuser::python_frontend {

std::vector<c10::optional<bool>> computeContiguity(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides) {
  TORCH_CHECK(
      sizes.size() == strides.size(),
      "compute_contiguity: Sizes and strides must have the same number of dimensions");
  auto not_broadcast = [&](auto i) { return strides[i] != 0 && sizes[i] != 1; };
  std::vector<c10::optional<bool>> contiguity(sizes.size(), c10::nullopt);
  if (contiguity.size() == 0) {
    return contiguity;
  }
  int64_t last = sizes.size() - 1;
  for (; last >= 0; --last) {
    if (not_broadcast(last)) {
      contiguity[last] = (strides.at(last) == 1);
      break;
    }
  }
  for (int64_t i = 0; i < last;) {
    if (not_broadcast(i)) {
      auto l = i++;
      for (; i <= last; i++) {
        if (not_broadcast(i)) {
          break;
        }
      }
      contiguity[l] = (strides[l] == strides[i] * sizes[i]);
    } else {
      i++;
    }
  }
  return contiguity;
}

void initNvFuserPythonBindings(PyObject* module) {
  auto nvfuser = py::handle(module).cast<py::module>();

  //! DataTypes supported by nvFuser in the FusionDefinition
  py::enum_<PrimDataType>(nvfuser, "DataType")
      .value("Double", DataType::Double)
      .value("Float", DataType::Float)
      .value("Half", DataType::Half)
      .value("Int", DataType::Int)
      .value("Int32", DataType::Int32)
      .value("Bool", DataType::Bool)
      .value("BFloat16", DataType::BFloat16)
      .value("ComplexFloat", DataType::ComplexFloat)
      .value("ComplexDouble", DataType::ComplexDouble)
      .value("Null", DataType::Null);

  nvfuser.def("compute_contiguity", computeContiguity);

  //! Binding the FusionCache that holds a cache of Fusions
  //! This is only bound to provide an interface to get the number of fusions
  //! that are cached.
  py::class_<FusionCache> fusion_cache(nvfuser, "FusionCache");
  fusion_cache
      .def_static(
          "get",
          &FusionCache::get,
          py::arg("max_fusions") = int(8192),
          py::return_value_policy::reference)
      .def("num_fusions", &FusionCache::numFusions)
      .def("print_stats", [](FusionCache& self) { self.print(std::cout); });

  //! These are the FusionDefinition supported object types that are either
  //! defined as inputs or the output of an operation.
  py::class_<Tensor> tensor_class(nvfuser, "Tensor");
  tensor_class.def("__repr__", [](Tensor& self) {
    std::stringstream ss;
    ss << "Tensor(index=" << self.index << ", dims=" << self.dims << ")";
    return ss.str();
  });
  tensor_class.def_property_readonly(
      "ndim", [](Tensor& self) { return self.dims; });
  tensor_class.def("_get_fusion_definition", [](Tensor& self) {
    return self.fusion_definition;
  });

  auto tensor_sizes = [](Tensor arg) -> std::vector<Scalar> {
    FUSER_PERF_SCOPE("Operators.tensor_sizes");
    auto fd = arg.fusion_definition;
    std::vector<Scalar> outputs;
    std::vector<State> output_state;
    for (const auto idx : c10::irange(arg.dims)) {
      outputs.push_back(fd->defineScalar());
      output_state.push_back(fd->recordingState(outputs[idx]()));
    }
    fd->defineRecord(
        new TensorSizesRecord({fd->recordingState(arg())}, output_state));
    return outputs;
  };
  tensor_class.def_property_readonly("shape", tensor_sizes);

  py::class_<Scalar> scalar_class(nvfuser, "Scalar");
  scalar_class.def("__repr__", [](Scalar& self) {
    std::stringstream ss;
    ss << "Scalar(index=" << self.index << ")";
    return ss.str();
  });

  //! The FusionDefinition is a context manager in Python where the user will
  //! define the set the operations and connections between operations for
  //! nvFuser to create.
  py::class_<FusionDefinition> fusion_def(nvfuser, "_FusionDefinition");
  fusion_def
      .def(
          py::init<c10::optional<size_t>, size_t>(),
          py::arg("id") = py::none(),
          py::arg("max_length") = int(1024))
      .def_readwrite("ops", &FusionDefinition::ops)
      .def_readwrite("sched", &FusionDefinition::sched)
      .def(
          "_setup_definition",
          [](FusionDefinition& self) -> FusionDefinition* {
            // Instrumentation to mark the beginning of a FusionDefinition
            inst::Trace::instance()->beginEvent(
                "FusionDefinition setupDefinition");
            return self.setupDefinition();
          })
      .def(
          "_finalize_definition",
          [](FusionDefinition& self) {
            self.finalizeDefinition();
            // Mark the end of a definition
            inst::Trace::instance()->endEvent(nullptr);
          })
      .def(
          "_setup_schedule",
          [](FusionDefinition& self, const py::iterable& iter) {
            // Instrumentation to mark the beginning of a schedule
            inst::Trace::instance()->beginEvent("FusionDefinition schedule");
            std::vector<c10::IValue> inputs;
            for (py::handle obj : iter) {
              inputs.push_back(torch::jit::toIValue(obj, c10::AnyType::get()));
            }
            self.setupSchedule(inputs);
          })
      .def(
          "_finalize_schedule",
          [](FusionDefinition& self, const py::iterable& iter) {
            std::vector<c10::IValue> inputs;
            for (py::handle obj : iter) {
              inputs.push_back(torch::jit::toIValue(obj, c10::AnyType::get()));
            }
            self.finalizeSchedule(inputs);
            // Mark the end of a schedule
            inst::Trace::instance()->endEvent(nullptr);
          })
      .def(
          "__repr__",
          [](FusionDefinition& self) {
            std::stringstream ss;
            self.print(ss);
            return ss.str();
          })
      .def("print", [](FusionDefinition& self) { self.print(std::cout); })
      .def("print_math_ir", [](FusionDefinition& self) { self.printMathIr(); })
      .def("print_ir", [](FusionDefinition& self) { self.printIr(); })
      .def(
          "_execute",
          [](FusionDefinition& self,
             const py::iterable& iter,
             bool override_user_schedule) {
            std::vector<c10::IValue> inputs;
            for (py::handle obj : iter) {
              inputs.push_back(torch::jit::toIValue(obj, c10::AnyType::get()));
            }
            return self.execute(inputs, override_user_schedule);
          },
          py::arg("inputs"),
          py::arg("override_user_schedule") = false,
          py::return_value_policy::reference)
      .def(
          "id",
          [](FusionDefinition& self) -> c10::optional<size_t> {
            return self.id();
          })
      .def(
          "add_output",
          [](FusionDefinition& self, Scalar output) {
            FUSER_PERF_SCOPE("FusionDefinition.add_output (scalar)");
            TORCH_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            self.defineRecord(
                new OutputRecord<Val>({self.recordingState(output())}));
          },
          py::arg("output"))
      .def(
          "add_output",
          [](FusionDefinition& self,
             Tensor output,
             c10::optional<Tensor> alias_input = c10::nullopt) {
            FUSER_PERF_SCOPE("FusionDefinition.add_output (tensor)");
            TORCH_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            if (alias_input.has_value()) {
              self.defineRecord(new OutputRecord<TensorView>(
                  {self.recordingState(output()),
                   self.recordingState(alias_input.value()())}));
            } else {
              self.defineRecord(new OutputRecord<TensorView>(
                  {self.recordingState(output())}));
            }
          },
          py::arg("output"),
          py::arg("alias_input") = py::none())
      .def(
          "add_output",
          [](FusionDefinition& self,
             Tensor output,
             std::vector<int64_t> stride_order) {
            FUSER_PERF_SCOPE("FusionDefinition.add_output (tensor)");
            TORCH_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            TORCH_CHECK(
                stride_order.empty() || output.dims == stride_order.size(),
                "stride_order needs to be either empty or the same length of Tensor `output`");
            int64_t duplicate_check = 0;
            for (const auto& v : stride_order) {
              TORCH_CHECK(
                  v >= 0 && v < (int64_t)stride_order.size(),
                  "stride_order elements need to be within [0, stride_order.size())");
              duplicate_check |= 1 << v;
            }
            TORCH_CHECK(
                duplicate_check == (1 << stride_order.size()) - 1,
                "duplicated elements in stride_order detected!");
            self.defineRecord(new OutputRecord<TensorView>(
                {self.recordingState(output())}, stride_order));
          },
          py::arg("output"),
          py::arg("stride_order"))
      .def(
          "define_tensor",
          [](FusionDefinition& self,
             std::vector<int64_t>& symbolic_sizes,
             std::vector<c10::optional<bool>>& contiguous,
             PrimDataType dtype = DataType::Float,
             bool is_cpu = false) -> Tensor {
            FUSER_PERF_SCOPE("FusionDefinition.define_tensor (default)");
            TORCH_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");

            for (size_t i = 0; i < symbolic_sizes.size(); ++i) {
              TORCH_CHECK(
                  symbolic_sizes[i] == -1 || symbolic_sizes[i] == 1,
                  "The value ",
                  symbolic_sizes[i],
                  " at index ",
                  i,
                  " was neither broadcast(1) or symbolic(-1).");
            }

            Tensor out = self.defineTensor(symbolic_sizes.size());
            self.defineRecord(new TensorRecord(
                {self.recordingState(out())},
                symbolic_sizes,
                contiguous,
                dtype,
                is_cpu));

            return out;
          },
          py::arg("symbolic_sizes"),
          py::arg("contiguous"),
          py::arg("dtype") = DataType::Float,
          py::arg("is_cpu") = false,
          py::return_value_policy::reference)
      .def(
          "define_tensor",
          [](FusionDefinition& self,
             std::vector<int64_t>& sizes,
             std::vector<int64_t>& strides,
             PrimDataType dtype = DataType::Float,
             bool is_cpu = false) -> Tensor {
            FUSER_PERF_SCOPE("FusionDefinition.define_tensor (integration)");
            TORCH_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            TORCH_CHECK(
                sizes.size() == strides.size(),
                "The number of sizes does not match the number of strides.",
                sizes.size(),
                strides.size());

            // TensorViewBuilder assumes any dim with a compile time constant
            // size == 1 is a "maybe broadcast" axis, symbolic sizes are
            // identified by -1, and size == 0 is not supported.

            // Translate to TensorViewBuilder's view of the world.
            std::vector<int64_t> maybe_symbolic_sizes;
            maybe_symbolic_sizes.reserve(sizes.size());
            for (const auto i : c10::irange(sizes.size())) {
              TORCH_INTERNAL_ASSERT(
                  sizes[i] >= 0,
                  "Size of ",
                  sizes[i],
                  " is not supported in nvFuser. Expected size >= 0.");
              if (sizes[i] == 1) {
                maybe_symbolic_sizes.push_back(1);
              } else {
                maybe_symbolic_sizes.push_back(-1);
              }
            }

            Tensor out = self.defineTensor(sizes.size());
            self.defineRecord(new TensorRecord(
                {self.recordingState(out())},
                std::move(maybe_symbolic_sizes),
                computeContiguity(sizes, strides),
                dtype,
                is_cpu));

            return out;
          },
          py::arg("sizes"),
          py::arg("strides"),
          py::arg("dtype") = DataType::Float,
          py::arg("is_cpu") = false,
          py::return_value_policy::reference)
      .def(
          "define_constant",
          [](FusionDefinition& self,
             double val,
             PrimDataType dtype = DataType::Double) -> Scalar {
            FUSER_PERF_SCOPE("FusionDefinition.define_constant (double)");
            TORCH_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            Scalar out = self.defineScalar();
            self.defineRecord(new ConstantRecord<Double, double>(
                {self.recordingState(out())}, val, dtype));
            return out;
          },
          py::arg("val"),
          py::arg("dtype") = DataType::Double,
          py::return_value_policy::reference)
      .def(
          "define_constant",
          [](FusionDefinition& self,
             std::complex<double> val,
             PrimDataType dtype = DataType::ComplexDouble) -> Scalar {
            FUSER_PERF_SCOPE("FusionDefinition.define_constant (complex)");
            TORCH_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            Scalar out = self.defineScalar();
            self.defineRecord(
                new ConstantRecord<ComplexDouble, std::complex<double>>(
                    {self.recordingState(out())}, val, dtype));
            return out;
          },
          py::arg("val"),
          py::arg("dtype") = DataType::ComplexDouble,
          py::return_value_policy::reference)
      .def(
          "define_constant",
          [](FusionDefinition& self,
             bool val,
             PrimDataType dtype = DataType::Bool) -> Scalar {
            FUSER_PERF_SCOPE("FusionDefinition.define_constant (bool)");
            TORCH_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            Scalar out = self.defineScalar();
            self.defineRecord(new ConstantRecord<Bool, bool>(
                {self.recordingState(out())}, val, dtype));
            return out;
          },
          py::arg("val"),
          py::arg("dtype") = DataType::Bool,
          py::return_value_policy::reference)
      .def(
          "define_constant",
          [](FusionDefinition& self,
             int64_t val,
             PrimDataType dtype = DataType::Int) -> Scalar {
            FUSER_PERF_SCOPE("FusionDefinition.define_constant (int)");
            TORCH_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            Scalar out = self.defineScalar();
            self.defineRecord(new ConstantRecord<Int, int64_t>(
                {self.recordingState(out())}, val, dtype));
            return out;
          },
          py::arg("val"),
          py::arg("dtype") = DataType::Int,
          py::return_value_policy::reference)
      .def(
          "define_scalar",
          [](FusionDefinition& self,
             PrimDataType dtype = DataType::Double) -> Scalar {
            FUSER_PERF_SCOPE("FusionDefinition.define_scalar");
            TORCH_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            Scalar out = self.defineScalar();
            self.defineRecord(
                new ScalarRecord({self.recordingState(out())}, dtype));
            return out;
          },
          py::arg("dtype") = DataType::Double,
          py::return_value_policy::reference);

  //! The Operators class is a nested class of FusionDefinition to allow the
  //! user to query the class for the list of operators.
  //!
  //! Example:
  //!   help(FusionDefinition.Operators)
  //!
  //! Additional operators are expected to be defined below as needed.  They
  //! may require defining a new RecordFunctor child class if they are unique.
  py::class_<FusionDefinition::Operators> nvf_ops(fusion_def, "Operators");
  nvf_ops.def(py::init<FusionDefinition*>());

  // ******************** INSERT OP BINDINGS BELOW HERE ********************
#define OP_PREFIX "Operators."
#define NVFUSER_PYTHON_BINDING_UNARY_OP(op_str, op_name)                       \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self, Tensor input) -> Tensor {          \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(input.dims);                          \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*>(               \
            {fd->recordingState(input())},                                     \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(TensorView*)>(op_name)));              \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self, Scalar input) -> Scalar {          \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*>(                             \
            {fd->recordingState(input())},                                     \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<Val* (*)(Val*)>(op_name)));                            \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      op_str,                                                                  \
      [](Tensor input) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = input.fusion_definition;                        \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(input.dims);                          \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*>(               \
            {fd->recordingState(input())},                                     \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(TensorView*)>(op_name)));              \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      op_str,                                                                  \
      [](Scalar input) -> Scalar {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = input.fusion_definition;                        \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*>(                             \
            {fd->recordingState(input())},                                     \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<Val* (*)(Val*)>(op_name)));                            \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_UNARY_OP("abs", abs)
  NVFUSER_PYTHON_BINDING_UNARY_OP("acos", acos)
  NVFUSER_PYTHON_BINDING_UNARY_OP("acosh", acosh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("asin", asin)
  NVFUSER_PYTHON_BINDING_UNARY_OP("asinh", asinh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("atan", atan)
  NVFUSER_PYTHON_BINDING_UNARY_OP("atanh", atanh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("ceil", ceil)
  NVFUSER_PYTHON_BINDING_UNARY_OP("cos", cos)
  NVFUSER_PYTHON_BINDING_UNARY_OP("cosh", cosh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("exp", exp)
  NVFUSER_PYTHON_BINDING_UNARY_OP("exp2", exp2)
  NVFUSER_PYTHON_BINDING_UNARY_OP("expm1", expm1)
  NVFUSER_PYTHON_BINDING_UNARY_OP("erf", erf)
  NVFUSER_PYTHON_BINDING_UNARY_OP("erfc", erfc)
  NVFUSER_PYTHON_BINDING_UNARY_OP("erfinv", erfinv)
  NVFUSER_PYTHON_BINDING_UNARY_OP("erfcinv", erfcinv)
  NVFUSER_PYTHON_BINDING_UNARY_OP("floor", floor)
  NVFUSER_PYTHON_BINDING_UNARY_OP("frac", frac)
  NVFUSER_PYTHON_BINDING_UNARY_OP("lgamma", lgamma)
  NVFUSER_PYTHON_BINDING_UNARY_OP("log", log)
  NVFUSER_PYTHON_BINDING_UNARY_OP("log10", log10)
  NVFUSER_PYTHON_BINDING_UNARY_OP("log1p", log1p)
  NVFUSER_PYTHON_BINDING_UNARY_OP("log2", log2)
  NVFUSER_PYTHON_BINDING_UNARY_OP("neg", neg)
  NVFUSER_PYTHON_BINDING_UNARY_OP("bitwise_not", bitwise_not)
  NVFUSER_PYTHON_BINDING_UNARY_OP("relu", relu)
  NVFUSER_PYTHON_BINDING_UNARY_OP("rand_like", rand_like)
  NVFUSER_PYTHON_BINDING_UNARY_OP("randn_like", randn_like)
  NVFUSER_PYTHON_BINDING_UNARY_OP("reciprocal", reciprocal)
  NVFUSER_PYTHON_BINDING_UNARY_OP("round", round)
  NVFUSER_PYTHON_BINDING_UNARY_OP("rsqrt", rsqrt)
  NVFUSER_PYTHON_BINDING_UNARY_OP("set", set)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sign", sign)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sigmoid", sigmoid)
  NVFUSER_PYTHON_BINDING_UNARY_OP("silu", silu)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sin", sin)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sinh", sinh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sqrt", sqrt)
  NVFUSER_PYTHON_BINDING_UNARY_OP("tan", tan)
  NVFUSER_PYTHON_BINDING_UNARY_OP("tanh", tanh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("trunc", trunc)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isfinite", isfinite)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isinf", isinf)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isnan", isnan)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isneginf", isneginf)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isposinf", isposinf)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isreal", isreal)
  NVFUSER_PYTHON_BINDING_UNARY_OP("real", real)
  NVFUSER_PYTHON_BINDING_UNARY_OP("imag", imag)
#undef NVFUSER_PYTHON_BINDING_UNARY_OP

#define NVFUSER_PYTHON_BINDING_UNARY_OP_SPECIAL(op_str, op_name)               \
  tensor_class.def(                                                            \
      "__" op_str "__",                                                        \
      [](Tensor input) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = input.fusion_definition;                        \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(input.dims);                          \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*>(               \
            {fd->recordingState(input())},                                     \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(TensorView*)>(op_name)));              \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      "__" op_str "__",                                                        \
      [](Scalar input) -> Scalar {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = input.fusion_definition;                        \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*>(                             \
            {fd->recordingState(input())},                                     \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<Val* (*)(Val*)>(op_name)));                            \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);
  NVFUSER_PYTHON_BINDING_UNARY_OP_SPECIAL("abs", abs)
  NVFUSER_PYTHON_BINDING_UNARY_OP_SPECIAL("neg", neg)
#undef NVFUSER_PYTHON_BINDING_UNARY_OP_SPECIAL

#define NVFUSER_PYTHON_BINDING_BINARY_OP(op_str, op_name)                      \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Tensor arg2) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, TensorView*>(  \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(TensorView*, TensorView*)>(op_name))); \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Scalar arg2) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*>(         \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(TensorView*, Val*)>(op_name)));        \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Tensor arg2) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, Val*, TensorView*>(         \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(Val*, TensorView*)>(op_name)));        \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Scalar arg2) -> Scalar {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*>(                       \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<Val* (*)(Val*, Val*)>(op_name)));                      \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      op_str,                                                                  \
      [](Tensor arg1, Tensor arg2) -> Tensor {                                 \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, TensorView*>(  \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(TensorView*, TensorView*)>(op_name))); \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      op_str,                                                                  \
      [](Tensor arg1, Scalar arg2) -> Tensor {                                 \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*>(         \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(TensorView*, Val*)>(op_name)));        \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      op_str,                                                                  \
      [](Scalar arg1, Tensor arg2) -> Tensor {                                 \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, Val*, TensorView*>(         \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(Val*, TensorView*)>(op_name)));        \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      op_str,                                                                  \
      [](Scalar arg1, Scalar arg2) -> Scalar {                                 \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*>(                       \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<Val* (*)(Val*, Val*)>(op_name)));                      \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_BINARY_OP("add", add)
  NVFUSER_PYTHON_BINDING_BINARY_OP("atan2", atan2)
  NVFUSER_PYTHON_BINDING_BINARY_OP("div", div)
  NVFUSER_PYTHON_BINDING_BINARY_OP("fmod", fmod)
  NVFUSER_PYTHON_BINDING_BINARY_OP("mul", mul)
  NVFUSER_PYTHON_BINDING_BINARY_OP("pow", pow)
  NVFUSER_PYTHON_BINDING_BINARY_OP("remainder", remainder)
  NVFUSER_PYTHON_BINDING_BINARY_OP("sub", sub)
  NVFUSER_PYTHON_BINDING_BINARY_OP("mod", mod)
  NVFUSER_PYTHON_BINDING_BINARY_OP("eq", eq)
  NVFUSER_PYTHON_BINDING_BINARY_OP("ge", ge)
  NVFUSER_PYTHON_BINDING_BINARY_OP("gt", gt)
  NVFUSER_PYTHON_BINDING_BINARY_OP("le", le)
  NVFUSER_PYTHON_BINDING_BINARY_OP("lt", lt)
  NVFUSER_PYTHON_BINDING_BINARY_OP("ne", ne)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_and", bitwise_and)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_or", bitwise_or)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_xor", bitwise_xor)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_left_shift", bitwise_left_shift)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_right_shift", bitwise_left_shift)
#undef NVFUSER_PYTHON_BINDING_BINARY_OP

#define NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL(py_op, op_str, op_name)       \
  tensor_class.def(                                                            \
      py_op,                                                                   \
      [](Tensor arg1, Tensor arg2) -> Tensor {                                 \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, TensorView*>(  \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(TensorView*, TensorView*)>(op_name))); \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      py_op,                                                                   \
      [](Tensor arg1, Scalar arg2) -> Tensor {                                 \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*>(         \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(TensorView*, Val*)>(op_name)));        \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      py_op,                                                                   \
      [](Scalar arg1, Tensor arg2) -> Tensor {                                 \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, Val*, TensorView*>(         \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(Val*, TensorView*)>(op_name)));        \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      py_op,                                                                   \
      [](Scalar arg1, Scalar arg2) -> Scalar {                                 \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg2.fusion_definition;                         \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*>(                       \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<Val* (*)(Val*, Val*)>(op_name)));                      \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__add__", "add", add)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__mul__", "mul", mul)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__pow__", "pow", pow)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__sub__", "sub", sub)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__mod__", "mod", mod)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__eq__", "eq", eq)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__ge__", "ge", ge)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__gt__", "gt", gt)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__le__", "le", le)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__lt__", "lt", lt)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__ne__", "ne", ne)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL(
      "__and__", "bitwise_and", bitwise_and)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__or__", "bitwise_or", bitwise_or)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL(
      "__xor__", "bitwise_xor", bitwise_xor)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL(
      "__lshift__", "bitwise_left_shift", bitwise_left_shift)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL(
      "__rshift__", "bitwise_right_shift", bitwise_left_shift)
  // In PyTorch, __div__ (//) and __truediv__ (/) are different.
  // When applied to integer-dtype arguments, they do as expected, returning
  // integer and float outputs, respectively. When applied to two floating-type
  // arguments, they return the floor of division for // and plain division for
  // /. When applied to mixed types, the types are promoted, so the
  // floating-point behavior is returned.
  // Our div operator matches the __truediv__ behavior, so we do not implement
  // __div__.
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__truediv__", "div", div)
#undef NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL

#define NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP(op_str, op_name)           \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Tensor arg2,                                                          \
         Scalar arg3) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, TensorView*, Val*>(         \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(TensorView*, TensorView*, Val*)>(          \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Scalar arg2,                                                          \
         Scalar arg3) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*, Val*>(   \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(op_name)));  \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Tensor arg2,                                                          \
         Scalar arg3) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, Val*, TensorView*, Val*>(   \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(Val*, TensorView*, Val*)>(op_name)));  \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Scalar arg2,                                                          \
         Scalar arg3) -> Scalar {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*, Val*>(                 \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<Val* (*)(Val*, Val*, Val*)>(op_name)));                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      op_str,                                                                  \
      [](Tensor arg1, Tensor arg2, Scalar arg3) -> Tensor {                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, TensorView*, Val*>(         \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(TensorView*, TensorView*, Val*)>(          \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      op_str,                                                                  \
      [](Tensor arg1, Scalar arg2, Scalar arg3) -> Tensor {                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*, Val*>(   \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(op_name)));  \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      op_str,                                                                  \
      [](Scalar arg1, Tensor arg2, Scalar arg3) -> Tensor {                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, Val*, TensorView*, Val*>(   \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(Val*, TensorView*, Val*)>(op_name)));  \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      op_str,                                                                  \
      [](Scalar arg1, Scalar arg2, Scalar arg3) -> Scalar {                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*, Val*>(                 \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<Val* (*)(Val*, Val*, Val*)>(op_name)));                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP("add_alpha", add_alpha)
  NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP("sub_alpha", sub_alpha)
#undef NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP

#define NVFUSER_PYTHON_BINDING_TERNARY_OP(op_str, op_name)                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Scalar arg2,                                                          \
         Scalar arg3) -> Scalar {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*, Val*>(                 \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<Val* (*)(Val*, Val*, Val*)>(op_name)));                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Tensor arg2,                                                          \
         Tensor arg3) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, TensorView*, TensorView*>(  \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                    TensorView* (*)(TensorView*, TensorView*, TensorView*)>(   \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Tensor arg2,                                                          \
         Scalar arg3) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, TensorView*, Val*>(         \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<TensorView* (*)(TensorView*, TensorView*, Val*)>(  \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Scalar arg2,                                                          \
         Tensor arg3) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, Val*, TensorView*>(         \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<TensorView* (*)(TensorView*, Val*, TensorView*)>(  \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Tensor arg2,                                                          \
         Tensor arg3) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, Val*, TensorView*, TensorView*>(         \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<TensorView* (*)(Val*, TensorView*, TensorView*)>(  \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Scalar arg2,                                                          \
         Tensor arg3) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg3.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, Val*, Val*, TensorView*>(   \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(Val*, Val*, TensorView*)>(op_name)));  \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Scalar arg2,                                                          \
         Scalar arg3) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*, Val*>(   \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(op_name)));  \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Tensor arg2,                                                          \
         Scalar arg3) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, Val*, TensorView*, Val*>(   \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(Val*, TensorView*, Val*)>(op_name)));  \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      op_str,                                                                  \
      [](Scalar arg1, Scalar arg2, Scalar arg3) -> Scalar {                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*, Val*>(                 \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<                                                       \
                                                                               \
                Val* (*)(Val*, Val*, Val*)>(op_name)));                        \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      op_str,                                                                  \
      [](Tensor arg1, Tensor arg2, Tensor arg3) -> Tensor {                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, TensorView*, TensorView*>(  \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(TensorView*, TensorView*, TensorView*)>(   \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      op_str,                                                                  \
      [](Tensor arg1, Tensor arg2, Scalar arg3) -> Tensor {                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, TensorView*, Val*>(         \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(TensorView*, TensorView*, Val*)>(          \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      op_str,                                                                  \
      [](Tensor arg1, Scalar arg2, Tensor arg3) -> Tensor {                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, Val*, TensorView*>(         \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(TensorView*, Val*, TensorView*)>(          \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      op_str,                                                                  \
      [](Scalar arg1, Tensor arg2, Tensor arg3) -> Tensor {                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, Val*, TensorView*, TensorView*>(         \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(Val*, TensorView*, TensorView*)>(          \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      op_str,                                                                  \
      [](Scalar arg1, Scalar arg2, Tensor arg3) -> Tensor {                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg3.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, Val*, Val*, TensorView*>(   \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<                                                       \
                                                                               \
                TensorView* (*)(Val*, Val*, TensorView*)>(op_name)));          \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      op_str,                                                                  \
      [](Tensor arg1, Scalar arg2, Scalar arg3) -> Tensor {                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*, Val*>(   \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<                                                       \
                                                                               \
                TensorView* (*)(TensorView*, Val*, Val*)>(op_name)));          \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      op_str,                                                                  \
      [](Scalar arg1, Tensor arg2, Scalar arg3) -> Tensor {                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, Val*, TensorView*, Val*>(   \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<                                                       \
                                                                               \
                TensorView* (*)(Val*, TensorView*, Val*)>(op_name)));          \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_TERNARY_OP("lerp", lerp)
  NVFUSER_PYTHON_BINDING_TERNARY_OP("where", where)
#undef NVFUSER_PYTHON_BINDING_TERNARY_OP

#define NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP(op_str, op_name)              \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Scalar arg2,                                                          \
         Scalar arg3) -> Scalar {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            !self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                         \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*, Val*>(                 \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<                                                       \
                                                                               \
                Val* (*)(Val*, Val*, Val*)>(op_name)));                        \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Scalar arg2,                                                          \
         Scalar arg3) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            !self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*, Val*>(   \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<                                                       \
                                                                               \
                TensorView* (*)(TensorView*, Val*, Val*)>(op_name)));          \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      op_str,                                                                  \
      [](Scalar arg1, Scalar arg2, Scalar arg3) -> Scalar {                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*, Val*>(                 \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<                                                       \
                                                                               \
                Val* (*)(Val*, Val*, Val*)>(op_name)));                        \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      op_str,                                                                  \
      [](Tensor arg1, Scalar arg2, Scalar arg3) -> Tensor {                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*, Val*>(   \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<                                                       \
                                                                               \
                TensorView* (*)(TensorView*, Val*, Val*)>(op_name)));          \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP("clamp", clamp)
  NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP("threshold", threshold)
#undef NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP

#define NVFUSER_PYTHON_BINDING_TERNARY_WITH_ALPHA_OP(op_str, op_name)          \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Scalar arg2,                                                          \
         Scalar arg3,                                                          \
         Scalar arg4) -> Scalar {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*, Val*, Val*>(           \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3()),                                       \
             fd->recordingState(arg4())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<                                                       \
                                                                               \
                Val* (*)(Val*, Val*, Val*, Val*)>(op_name)));                  \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Tensor arg2,                                                          \
         Tensor arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<                                         \
                         TensorView*,                                          \
                         TensorView*,                                          \
                         TensorView*,                                          \
                         TensorView*,                                          \
                         Val*>(                                                \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3()),                                       \
             fd->recordingState(arg4())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<                                                       \
                                                                               \
                TensorView* (*)(TensorView*, TensorView*, TensorView*, Val*)>( \
                op_name)));                                                    \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Tensor arg2,                                                          \
         Scalar arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, TensorView*, Val*, Val*>(   \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(TensorView*, TensorView*, Val*, Val*)>(    \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Scalar arg2,                                                          \
         Tensor arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, Val*, TensorView*, Val*>(   \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(TensorView*, Val*, TensorView*, Val*)>(    \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Tensor arg2,                                                          \
         Tensor arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, Val*, TensorView*, TensorView*, Val*>(   \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(Val*, TensorView*, TensorView*, Val*)>(    \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Scalar arg2,                                                          \
         Tensor arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg3.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, Val*, Val*, TensorView*, Val*>(          \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(Val*, Val*, TensorView*, Val*)>(           \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Scalar arg2,                                                          \
         Scalar arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, Val*, Val*, Val*>(          \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(TensorView*, Val*, Val*, Val*)>(           \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Tensor arg2,                                                          \
         Scalar arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, Val*, TensorView*, Val*, Val*>(          \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(Val*, TensorView*, Val*, Val*)>(           \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      op_str,                                                                  \
      [](Scalar arg1, Scalar arg2, Scalar arg3, Scalar arg4) -> Scalar {       \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*, Val*, Val*>(           \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3()),                                       \
             fd->recordingState(arg4())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<                                                       \
                                                                               \
                Val* (*)(Val*, Val*, Val*, Val*)>(op_name)));                  \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      op_str,                                                                  \
      [](Tensor arg1, Tensor arg2, Tensor arg3, Scalar arg4) -> Tensor {       \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<                                         \
                         TensorView*,                                          \
                         TensorView*,                                          \
                         TensorView*,                                          \
                         TensorView*,                                          \
                         Val*>(                                                \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3()),                                       \
             fd->recordingState(arg4())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<                                                       \
                                                                               \
                TensorView* (*)(TensorView*, TensorView*, TensorView*, Val*)>( \
                op_name)));                                                    \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      op_str,                                                                  \
      [](Tensor arg1, Tensor arg2, Scalar arg3, Scalar arg4) -> Tensor {       \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, TensorView*, Val*, Val*>(   \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(TensorView*, TensorView*, Val*, Val*)>(    \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      op_str,                                                                  \
      [](Tensor arg1, Scalar arg2, Tensor arg3, Scalar arg4) -> Tensor {       \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, Val*, TensorView*, Val*>(   \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(TensorView*, Val*, TensorView*, Val*)>(    \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      op_str,                                                                  \
      [](Scalar arg1, Tensor arg2, Tensor arg3, Scalar arg4) -> Tensor {       \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, Val*, TensorView*, TensorView*, Val*>(   \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(Val*, TensorView*, TensorView*, Val*)>(    \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      op_str,                                                                  \
      [](Scalar arg1, Scalar arg2, Tensor arg3, Scalar arg4) -> Tensor {       \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg3.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, Val*, Val*, TensorView*, Val*>(          \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(Val*, Val*, TensorView*, Val*)>(           \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      op_str,                                                                  \
      [](Tensor arg1, Scalar arg2, Scalar arg3, Scalar arg4) -> Tensor {       \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, Val*, Val*, Val*>(          \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(TensorView*, Val*, Val*, Val*)>(           \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      op_str,                                                                  \
      [](Scalar arg1, Tensor arg2, Scalar arg3, Scalar arg4) -> Tensor {       \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, Val*, TensorView*, Val*, Val*>(          \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<                                                   \
                                                                               \
                    TensorView* (*)(Val*, TensorView*, Val*, Val*)>(           \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_TERNARY_WITH_ALPHA_OP("addcmul", addcmul)
#undef NVFUSER_PYTHON_BINDING_TERNARY_WITH_ALPHA_OP

#define NVFUSER_PYTHON_BINDING_REDUCTION_OP(op_str, op_name)                            \
  nvf_ops.def(                                                                          \
      op_str,                                                                           \
      [](FusionDefinition::Operators& self,                                             \
         Tensor arg,                                                                    \
         PrimDataType dtype) -> Tensor {                                                \
        FUSER_PERF_SCOPE("Operators." op_str);                                          \
        TORCH_CHECK(                                                                    \
            self.validUse(), "Attempting to add to a completed definition!");           \
        FusionDefinition* fd = self.fusion_definition;                                  \
        size_t ndims = 0;                                                               \
        std::vector<int> axes(arg.dims);                                                \
        std::iota(axes.begin(), axes.end(), 0);                                         \
        Tensor output = fd->defineTensor(ndims);                                        \
        fd->defineRecord(new ReductionOpRecord(                                         \
            {fd->recordingState(arg())},                                                \
            {fd->recordingState(output())},                                             \
            ("ops." op_str),                                                            \
            static_cast<                                                                \
                TensorView* (*)(TensorView*, const std::vector<int>&, bool, DataType)>( \
                op_name),                                                               \
            axes,                                                                       \
            false,                                                                      \
            dtype));                                                                    \
        return output;                                                                  \
      },                                                                                \
      py::arg("arg"),                                                                   \
      py::arg("dtype") = DataType::Null,                                                \
      py::return_value_policy::reference);                                              \
  nvf_ops.def(                                                                          \
      op_str,                                                                           \
      [](FusionDefinition::Operators& self,                                             \
         Tensor arg,                                                                    \
         int axis,                                                                      \
         bool keepdim,                                                                  \
         PrimDataType dtype) -> Tensor {                                                \
        FUSER_PERF_SCOPE("Operators." op_str);                                          \
        TORCH_CHECK(                                                                    \
            self.validUse(), "Attempting to add to a completed definition!");           \
        FusionDefinition* fd = self.fusion_definition;                                  \
        size_t ndims = keepdim ? arg.dims : (arg.dims - 1);                             \
        Tensor output = fd->defineTensor(ndims);                                        \
        fd->defineRecord(new ReductionOpRecord(                                         \
            {fd->recordingState(arg())},                                                \
            {fd->recordingState(output())},                                             \
            ("ops." op_str),                                                            \
            static_cast<                                                                \
                TensorView* (*)(TensorView*, const std::vector<int>&, bool, DataType)>( \
                op_name),                                                               \
            {axis},                                                                     \
            keepdim,                                                                    \
            dtype));                                                                    \
        return output;                                                                  \
      },                                                                                \
      py::arg("arg"),                                                                   \
      py::arg("axis"),                                                                  \
      py::arg("keepdim") = false,                                                       \
      py::arg("dtype") = DataType::Null,                                                \
      py::return_value_policy::reference);                                              \
  nvf_ops.def(                                                                          \
      op_str,                                                                           \
      [](FusionDefinition::Operators& self,                                             \
         Tensor arg,                                                                    \
         const std::vector<int>& axes,                                                  \
         bool keepdim,                                                                  \
         PrimDataType dtype) -> Tensor {                                                \
        FUSER_PERF_SCOPE("Operators." op_str);                                          \
        TORCH_CHECK(                                                                    \
            self.validUse(), "Attempting to add to a completed definition!");           \
        FusionDefinition* fd = self.fusion_definition;                                  \
        size_t ndims = keepdim ? arg.dims : (arg.dims - axes.size());                   \
        Tensor output = fd->defineTensor(ndims);                                        \
        fd->defineRecord(new ReductionOpRecord(                                         \
            {fd->recordingState(arg())},                                                \
            {fd->recordingState(output())},                                             \
            ("ops." op_str),                                                            \
            static_cast<                                                                \
                                                                                        \
                TensorView* (*)(TensorView*, const std::vector<int>&, bool, DataType)>( \
                op_name),                                                               \
            axes,                                                                       \
            keepdim,                                                                    \
            dtype));                                                                    \
        return output;                                                                  \
      },                                                                                \
      py::arg("arg"),                                                                   \
      py::arg("axes"),                                                                  \
      py::arg("keepdim") = false,                                                       \
      py::arg("dtype") = DataType::Null,                                                \
      py::return_value_policy::reference);                                              \
  tensor_class.def(                                                                     \
      op_str,                                                                           \
      [](Tensor arg,                                                                    \
         const std::vector<int>& axes,                                                  \
         bool keepdim,                                                                  \
         PrimDataType dtype) -> Tensor {                                                \
        FUSER_PERF_SCOPE("Operators." op_str);                                          \
        FusionDefinition* fd = arg.fusion_definition;                                   \
        size_t ndims = keepdim ? arg.dims : (arg.dims - axes.size());                   \
        Tensor output = fd->defineTensor(ndims);                                        \
        fd->defineRecord(new ReductionOpRecord(                                         \
            {fd->recordingState(arg())},                                                \
            {fd->recordingState(output())},                                             \
            ("ops." op_str),                                                            \
            static_cast<                                                                \
                TensorView* (*)(TensorView*, const std::vector<int>&, bool, DataType)>( \
                op_name),                                                               \
            axes,                                                                       \
            keepdim,                                                                    \
            dtype));                                                                    \
        return output;                                                                  \
      },                                                                                \
      py::arg("axes"),                                                                  \
      py::arg("keepdim") = false,                                                       \
      py::arg("dtype") = DataType::Null,                                                \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_REDUCTION_OP("sum", sum)
  NVFUSER_PYTHON_BINDING_REDUCTION_OP("prod", prod)
  NVFUSER_PYTHON_BINDING_REDUCTION_OP("max", max)
  NVFUSER_PYTHON_BINDING_REDUCTION_OP("min", min)
#undef NVFUSER_PYTHON_BINDING_REDUCTION_OP

#define NVFUSER_PYTHON_BINDING_CAST_OP(op_str, op_name)                        \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg,                                                           \
         PrimDataType dtype) -> Tensor {                                       \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg.dims);                            \
        fd->defineRecord(new CastOpRecord<TensorView*, TensorView*>(           \
            {fd->recordingState(arg())},                                       \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<TensorView* (*)(DataType, TensorView*)>(op_name),      \
            dtype));                                                           \
        return output;                                                         \
      },                                                                       \
      py::arg("arg"),                                                          \
      py::arg("dtype"),                                                        \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg,                                                           \
         PrimDataType dtype) -> Scalar {                                       \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        TORCH_CHECK(                                                           \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new CastOpRecord<Val*, Val*>(                         \
            {fd->recordingState(arg())},                                       \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<Val* (*)(DataType, Val*)>(op_name),                    \
            dtype));                                                           \
        return output;                                                         \
      },                                                                       \
      py::arg("arg"),                                                          \
      py::arg("dtype"),                                                        \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      op_str,                                                                  \
      [](Tensor arg, PrimDataType dtype) -> Tensor {                           \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg.fusion_definition;                          \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(arg.dims);                            \
        fd->defineRecord(new CastOpRecord<TensorView*, TensorView*>(           \
            {fd->recordingState(arg())},                                       \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<                                                       \
                                                                               \
                TensorView* (*)(DataType, TensorView*)>(op_name),              \
            dtype));                                                           \
        return output;                                                         \
      },                                                                       \
      py::arg("dtype"),                                                        \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      op_str,                                                                  \
      [](Scalar arg, PrimDataType dtype) -> Scalar {                           \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg.fusion_definition;                          \
        TORCH_CHECK(                                                           \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new CastOpRecord<Val*, Val*>(                         \
            {fd->recordingState(arg())},                                       \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<Val* (*)(DataType, Val*)>(op_name),                    \
            dtype));                                                           \
        return output;                                                         \
      },                                                                       \
      py::arg("dtype"),                                                        \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_CAST_OP("cast", castOp)
#undef NVFUSER_PYTHON_BINDING_CAST_OP

  nvf_ops.def(
      "batch_norm",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         c10::optional<Tensor> weight,
         c10::optional<Tensor> bias,
         c10::optional<Tensor> running_mean,
         c10::optional<Tensor> running_var,
         Scalar momentum,
         Scalar eps,
         bool training,
         bool channels_last) -> decltype(auto) {
        FUSER_PERF_SCOPE("Operators.batch_norm");
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        Tensor mean = fd->defineTensor(1);
        Tensor invstd = fd->defineTensor(1);
        auto weight_state = weight.has_value()
            ? fd->recordingState(weight.value()())
            : State(0, StateType::None);
        auto bias_state = bias.has_value() ? fd->recordingState(bias.value()())
                                           : State(0, StateType::None);
        auto running_mean_state = running_mean.has_value()
            ? fd->recordingState(running_mean.value()())
            : State(0, StateType::None);
        auto running_var_state = running_var.has_value()
            ? fd->recordingState(running_var.value()())
            : State(0, StateType::None);
        fd->defineRecord(new BatchNormOpRecord(
            {fd->recordingState(arg()),
             weight_state,
             bias_state,
             running_mean_state,
             running_var_state,
             fd->recordingState(momentum()),
             fd->recordingState(eps())},
            {fd->recordingState(output()),
             fd->recordingState(mean()),
             fd->recordingState(invstd())},
            training,
            channels_last));
        return std::make_tuple(output, mean, invstd);
      },
      py::arg("arg"),
      py::arg("weight").none(true),
      py::arg("bias").none(true),
      py::arg("running_mean").none(true),
      py::arg("running_var").none(true),
      py::arg("momentum"),
      py::arg("eps"),
      py::arg("training"),
      py::arg("channels_last") = false,
      py::return_value_policy::reference);
  tensor_class.def(
      "batch_norm",
      [](Tensor arg,
         c10::optional<Tensor> weight,
         c10::optional<Tensor> bias,
         c10::optional<Tensor> running_mean,
         c10::optional<Tensor> running_var,
         Scalar momentum,
         Scalar eps,
         bool training,
         bool channels_last) -> decltype(auto) {
        FUSER_PERF_SCOPE("Operators.batch_norm");
        FusionDefinition* fd = arg.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        Tensor mean = fd->defineTensor(1);
        Tensor invstd = fd->defineTensor(1);
        auto weight_state = weight.has_value()
            ? fd->recordingState(weight.value()())
            : State(0, StateType::None);
        auto bias_state = bias.has_value() ? fd->recordingState(bias.value()())
                                           : State(0, StateType::None);
        auto running_mean_state = running_mean.has_value()
            ? fd->recordingState(running_mean.value()())
            : State(0, StateType::None);
        auto running_var_state = running_var.has_value()
            ? fd->recordingState(running_var.value()())
            : State(0, StateType::None);
        fd->defineRecord(new BatchNormOpRecord(
            {fd->recordingState(arg()),
             weight_state,
             bias_state,
             running_mean_state,
             running_var_state,
             fd->recordingState(momentum()),
             fd->recordingState(eps())},
            {fd->recordingState(output()),
             fd->recordingState(mean()),
             fd->recordingState(invstd())},
            training,
            channels_last));
        return std::make_tuple(output, mean, invstd);
      },
      py::arg("weight").none(true),
      py::arg("bias").none(true),
      py::arg("running_mean").none(true),
      py::arg("running_var").none(true),
      py::arg("momentum"),
      py::arg("eps"),
      py::arg("training"),
      py::arg("channels_last") = false,
      py::return_value_policy::reference);
  // Concreate Output Shape Overload
  nvf_ops.def(
      "broadcast_in_dim",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int64_t>& output_shape,
         std::vector<int64_t>& broadcast_dims) -> Tensor {
        FUSER_PERF_SCOPE("Operators.broadcast_in_dim");
        FusionDefinition* fd = self.fusion_definition;
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        TORCH_CHECK(
            output_shape.size() >= broadcast_dims.size(),
            "broadcast_dims vector size is too big for output shape!");
        Tensor output = fd->defineTensor(output_shape.size());
        fd->defineRecord(new BroadcastInDimOpRecord<int64_t>(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            "ops.broadcast_in_dim",
            output_shape,
            broadcast_dims));
        return output;
      },
      py::arg("arg"),
      py::arg("output_shape"),
      py::arg("broadcast_dims"),
      py::return_value_policy::reference);
  tensor_class.def(
      "broadcast_in_dim",
      [](Tensor arg,
         std::vector<int64_t>& output_shape,
         std::vector<int64_t>& broadcast_dims) -> Tensor {
        FUSER_PERF_SCOPE("Operators.broadcast_in_dim");
        FusionDefinition* fd = arg.fusion_definition;
        TORCH_CHECK(
            output_shape.size() >= broadcast_dims.size(),
            "broadcast_dims vector size is too big for output shape!");
        Tensor output = fd->defineTensor(output_shape.size());
        fd->defineRecord(new BroadcastInDimOpRecord<int64_t>(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            "ops.broadcast_in_dim",
            output_shape,
            broadcast_dims));
        return output;
      },
      py::arg("output_shape"),
      py::arg("broadcast_dims"),
      py::return_value_policy::reference);
  // Symbolic Output Shape Overload
  nvf_ops.def(
      "broadcast_in_dim",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<Scalar>& output_shape,
         std::vector<int64_t>& broadcast_dims) -> Tensor {
        FUSER_PERF_SCOPE("Operators.broadcast_in_dim");
        FusionDefinition* fd = self.fusion_definition;
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        TORCH_CHECK(
            output_shape.size() >= broadcast_dims.size(),
            "broadcast_dims vector size is too big for output shape!");
        Tensor output = fd->defineTensor(output_shape.size());
        std::vector<State> output_shape_states(
            output_shape.size(), State(0, StateType::Scalar));
        std::transform(
            output_shape.begin(),
            output_shape.end(),
            output_shape_states.begin(),
            [&fd](const Scalar& s) { return fd->recordingState(s()); });
        fd->defineRecord(new BroadcastInDimOpRecord<State>(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            "ops.broadcast_in_dim",
            output_shape_states,
            broadcast_dims));
        return output;
      },
      py::arg("arg"),
      py::arg("output_shape"),
      py::arg("broadcast_dims"),
      py::return_value_policy::reference);
  tensor_class.def(
      "broadcast_in_dim",
      [](Tensor arg,
         std::vector<Scalar>& output_shape,
         std::vector<int64_t>& broadcast_dims) -> Tensor {
        FUSER_PERF_SCOPE("Operators.broadcast_in_dim");
        FusionDefinition* fd = arg.fusion_definition;
        TORCH_CHECK(
            output_shape.size() >= broadcast_dims.size(),
            "broadcast_dims vector size is too big for output shape!");
        Tensor output = fd->defineTensor(output_shape.size());
        std::vector<State> output_shape_states(
            output_shape.size(), State(0, StateType::Scalar));
        std::transform(
            output_shape.begin(),
            output_shape.end(),
            output_shape_states.begin(),
            [&fd](const Scalar& s) { return fd->recordingState(s()); });
        fd->defineRecord(new BroadcastInDimOpRecord<State>(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            "ops.broadcast_in_dim",
            output_shape_states,
            broadcast_dims));
        return output;
      },
      py::arg("output_shape"),
      py::arg("broadcast_dims"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "broadcast",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<bool>& is_broadcast_dim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.broadcast");
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        fd->defineRecord(new BroadcastOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            "ops.broadcast",
            is_broadcast_dim));
        return output;
      },
      py::arg("arg"),
      py::arg("is_broadcast_dim"),
      py::return_value_policy::reference);
  tensor_class.def(
      "broadcast",
      [](Tensor arg, std::vector<bool>& is_broadcast_dim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.broadcast");
        FusionDefinition* fd = arg.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        fd->defineRecord(new BroadcastOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            "ops.broadcast",
            is_broadcast_dim));
        return output;
      },
      py::arg("is_broadcast_dim"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "cat",
      [](FusionDefinition::Operators& self,
         std::vector<Tensor> tensors,
         int64_t dim) -> Tensor {
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        TORCH_CHECK(
            tensors.size() > 0,
            "Attempting to concatenate empty list of tensors")
        Tensor output = fd->defineTensor(tensors[0].dims);
        std::vector<State> tensor_states;
        for (auto& t : tensors) {
          tensor_states.push_back(fd->recordingState(t()));
        }
        self.fusion_definition->defineRecord(new CatOpRecord(
            tensor_states, {fd->recordingState(output())}, dim));
        return output;
      },
      py::arg("tensors"),
      py::arg("dim"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "index_select",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         Tensor index,
         int64_t dim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.index_select");
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        fd->defineRecord(new IndexSelectOpRecord(
            {
                fd->recordingState(arg()),
                fd->recordingState(index()),
            },
            {fd->recordingState(output())},
            dim));
        return output;
      },
      py::arg("arg"),
      py::arg("index"),
      py::arg("dim"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "gather",
      [](FusionDefinition::Operators& self,
         Tensor arg1,
         Tensor index,
         int64_t dim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.gather");
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg1.dims);
        fd->defineRecord(new TorchGatherOpRecord(
            {
                fd->recordingState(arg1()),
                fd->recordingState(index()),
            },
            {fd->recordingState(output())},
            dim));
        return output;
      },
      py::arg("arg1"),
      py::arg("index"),
      py::arg("dim"),
      py::return_value_policy::reference);
  tensor_class.def(
      "gather",
      [](Tensor arg1, Tensor index, int64_t dim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.gather");
        FusionDefinition* fd = arg1.fusion_definition;
        Tensor output = fd->defineTensor(arg1.dims);
        fd->defineRecord(new TorchGatherOpRecord(
            {
                fd->recordingState(arg1()),
                fd->recordingState(index()),
            },
            {fd->recordingState(output())},
            dim));
        return output;
      },
      py::arg("index"),
      py::arg("dim"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "permute",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int64_t>& dims) -> Tensor {
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        self.fusion_definition->defineRecord(new PermuteOpRecord(
            {fd->recordingState(arg())}, {fd->recordingState(output())}, dims));
        return output;
      },
      py::arg("arg"),
      py::arg("dims"),
      py::return_value_policy::reference);
  tensor_class.def(
      "permute",
      [](Tensor arg, std::vector<int64_t>& dims) -> Tensor {
        FusionDefinition* fd = arg.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        fd->defineRecord(new PermuteOpRecord(
            {fd->recordingState(arg())}, {fd->recordingState(output())}, dims));
        return output;
      },
      py::arg("dims"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "slice",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int64_t>& start_indices,
         std::vector<int64_t>& end_indices,
         std::vector<int64_t>& strides) -> Tensor {
        FUSER_PERF_SCOPE("Operators.slice");
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        TORCH_CHECK(
            arg.dims == start_indices.size(),
            "Number of tensor dimensions does not match slice dimensions! Tensor-dims: ",
            arg.dims,
            " Slice-dims: ",
            start_indices.size());
        TORCH_CHECK(
            (start_indices.size() == end_indices.size()) &&
                (end_indices.size() == strides.size()),
            "Slice indexing attribute dimensions don't match! Start Indices: ",
            start_indices.size(),
            " End Indices: ",
            end_indices.size(),
            " Strides: ",
            strides.size());
        for (const auto i : c10::irange(arg.dims)) {
          auto start_idx = start_indices[i];
          auto end_idx = end_indices[i];
          auto stride = strides[i];
          TORCH_CHECK(
              start_idx >= 0,
              "Slice operation start_indices must be greater-than-or-equal-to 0. Start Indices: ",
              start_indices,
              " End Indices: ",
              end_indices,
              " Strides: ",
              strides);
          TORCH_CHECK(
              end_idx >= start_idx,
              "Slice operation end_indices must be greater-than-or-equal-to start_indices. Start Indices: ",
              start_indices,
              " End Indices: ",
              end_indices,
              " Strides: ",
              strides);
          TORCH_CHECK(
              stride == 1,
              "nvFuser Limitation: All slice operation strides must be of size 1. Start Indices: ",
              start_indices,
              " End Indices: ",
              end_indices,
              " Strides: ",
              strides);
        }
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        fd->defineRecord(new SliceOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            start_indices,
            end_indices,
            strides));
        return output;
      },
      py::arg("arg"),
      py::arg("start_indices"),
      py::arg("end_indices"),
      py::arg("strides"),
      py::return_value_policy::reference);
  tensor_class.def(
      "slice",
      [](Tensor arg,
         std::vector<int64_t>& start_indices,
         std::vector<int64_t>& end_indices,
         std::vector<int64_t>& strides) -> Tensor {
        FUSER_PERF_SCOPE("Operators.slice");
        FusionDefinition* fd = arg.fusion_definition;
        TORCH_CHECK(
            fd->ops.validUse(), "Attempting to add to a completed definition!");
        TORCH_CHECK(
            arg.dims == start_indices.size(),
            "Number of tensor dimensions does not match slice dimensions! Tensor-dims: ",
            arg.dims,
            " Slice-dims: ",
            start_indices.size());
        TORCH_CHECK(
            (start_indices.size() == end_indices.size()) &&
                (end_indices.size() == strides.size()),
            "Slice indexing attribute dimensions don't match! Start Indices: ",
            start_indices.size(),
            " End Indices: ",
            end_indices.size(),
            " Strides: ",
            strides.size());
        for (const auto i : c10::irange(arg.dims)) {
          auto start_idx = start_indices[i];
          auto end_idx = end_indices[i];
          auto stride = strides[i];
          TORCH_CHECK(
              start_idx >= 0,
              "Slice operation start_indices must be greater-than-or-equal-to 0. Start Indices: ",
              start_indices,
              " End Indices: ",
              end_indices,
              " Strides: ",
              strides);
          TORCH_CHECK(
              end_idx >= start_idx,
              "Slice operation end_indices must be greater-than-or-equal-to start_indices. Start Indices: ",
              start_indices,
              " End Indices: ",
              end_indices,
              " Strides: ",
              strides);
          TORCH_CHECK(
              stride == 1,
              "nvFuser Limitation: All slice operation strides must be of size 1. Start Indices: ",
              start_indices,
              " End Indices: ",
              end_indices,
              " Strides: ",
              strides);
        }
        Tensor output = fd->defineTensor(arg.dims);
        fd->defineRecord(new SliceOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            start_indices,
            end_indices,
            strides));
        return output;
      },
      py::arg("start_indices"),
      py::arg("end_indices"),
      py::arg("strides"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "squeeze",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int64_t>& original_shape,
         std::vector<int64_t>& dims) -> Tensor {
        FUSER_PERF_SCOPE("Operators.squeeze");
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims - 1);
        fd->defineRecord(new SqueezeOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            original_shape,
            dims));
        return output;
      },
      py::arg("arg"),
      py::arg("original_shape"),
      py::arg("dims"),
      py::return_value_policy::reference);
  tensor_class.def(
      "squeeze",
      [](Tensor arg,
         std::vector<int64_t>& original_shape,
         std::vector<int64_t>& dims) -> Tensor {
        FUSER_PERF_SCOPE("Operators.squeeze");
        FusionDefinition* fd = arg.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims - 1);
        fd->defineRecord(new SqueezeOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            original_shape,
            dims));
        return output;
      },
      py::arg("original_shape"),
      py::arg("dims"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "tensor_sizes",
      [](FusionDefinition::Operators& self, Tensor arg) -> std::vector<Scalar> {
        FUSER_PERF_SCOPE("Operators.tensor_sizes");
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        std::vector<Scalar> outputs;
        std::vector<State> output_state;
        for (const auto idx : c10::irange(arg.dims)) {
          outputs.push_back(fd->defineScalar());
          output_state.push_back(fd->recordingState(outputs[idx]()));
        }
        fd->defineRecord(
            new TensorSizesRecord({fd->recordingState(arg())}, output_state));
        return outputs;
      },
      py::arg("arg"),
      py::return_value_policy::reference);
  tensor_class.def(
      "tensor_sizes",
      [](Tensor arg) -> std::vector<Scalar> {
        FUSER_PERF_SCOPE("Operators.tensor_sizes");
        FusionDefinition* fd = arg.fusion_definition;
        std::vector<Scalar> outputs;
        std::vector<State> output_state;
        for (const auto idx : c10::irange(arg.dims)) {
          outputs.push_back(fd->defineScalar());
          output_state.push_back(fd->recordingState(outputs[idx]()));
        }
        fd->defineRecord(
            new TensorSizesRecord({fd->recordingState(arg())}, output_state));
        return outputs;
      },
      py::return_value_policy::reference);
  nvf_ops.def(
      "reshape",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int64_t>& original_shape,
         std::vector<int64_t>& new_shape) -> Tensor {
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(new_shape.size());
        self.fusion_definition->defineRecord(new ReshapeOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            original_shape,
            new_shape));
        return output;
      },
      py::arg("arg"),
      py::arg("original_shape"),
      py::arg("new_shape"),
      py::return_value_policy::reference);
  tensor_class.def(
      "reshape",
      [](Tensor arg,
         std::vector<int64_t>& original_shape,
         std::vector<int64_t>& new_shape) -> Tensor {
        FusionDefinition* fd = arg.fusion_definition;
        Tensor output = fd->defineTensor(new_shape.size());
        fd->defineRecord(new ReshapeOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            original_shape,
            new_shape));
        return output;
      },
      py::arg("original_shape"),
      py::arg("new_shape"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "full",
      [](FusionDefinition::Operators& self,
         std::vector<int64_t>& size,
         Scalar arg,
         PrimDataType dtype) -> Tensor {
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(size.size());
        fd->defineRecord(new FullOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            size,
            dtype));
        return output;
      },
      py::arg("size"),
      py::arg("arg"),
      py::arg("dtype"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "iota",
      [](FusionDefinition::Operators& self,
         Scalar length,
         c10::optional<Scalar> start,
         c10::optional<Scalar> step,
         PrimDataType dtype) -> Tensor {
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(1);
        auto start_state = start.has_value()
            ? fd->recordingState(start.value()())
            : State(0, StateType::None);
        auto step_state = step.has_value() ? fd->recordingState(step.value()())
                                           : State(0, StateType::None);
        fd->defineRecord(new IotaOpRecord(
            {fd->recordingState(length()), start_state, step_state},
            {fd->recordingState(output())},
            dtype));
        return output;
      },
      py::arg("length"),
      py::arg("start").none(true),
      py::arg("step").none(true),
      py::arg("dtype") = DataType::Int,
      py::return_value_policy::reference);
  nvf_ops.def(
      "var",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int>& axes,
         int64_t correction,
         bool keepdim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.var");
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        size_t ndims = keepdim ? arg.dims : (arg.dims - axes.size());
        Tensor output = fd->defineTensor(ndims);
        fd->defineRecord(new VarianceOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            axes,
            correction,
            keepdim));
        return output;
      },
      py::arg("arg"),
      py::arg("axes"),
      py::arg("correction"),
      py::arg("keepdim") = false,
      py::return_value_policy::reference);
  tensor_class.def(
      "var",
      [](Tensor arg, std::vector<int>& axes, int64_t correction, bool keepdim)
          -> Tensor {
        FUSER_PERF_SCOPE("Operators.var");
        FusionDefinition* fd = arg.fusion_definition;
        size_t ndims = keepdim ? arg.dims : (arg.dims - axes.size());
        Tensor output = fd->defineTensor(ndims);
        fd->defineRecord(new VarianceOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            axes,
            correction,
            keepdim));
        return output;
      },
      py::arg("axes"),
      py::arg("correction"),
      py::arg("keepdim") = false,
      py::return_value_policy::reference);
  nvf_ops.def(
      "var_mean",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int>& axes,
         int64_t correction,
         bool keepdim) -> decltype(auto) {
        FUSER_PERF_SCOPE("Operators.var_mean");
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        size_t ndims = keepdim ? arg.dims : (arg.dims - axes.size());
        Tensor var = fd->defineTensor(ndims);
        Tensor mean = fd->defineTensor(ndims);
        fd->defineRecord(new VarianceMeanOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(var()), fd->recordingState(mean())},
            axes,
            correction,
            keepdim));
        return std::make_tuple(var, mean);
      },
      py::arg("arg"),
      py::arg("axes"),
      py::arg("correction"),
      py::arg("keepdim") = false,
      py::return_value_policy::reference);
  tensor_class.def(
      "var_mean",
      [](Tensor arg, std::vector<int>& axes, int64_t correction, bool keepdim)
          -> decltype(auto) {
        FUSER_PERF_SCOPE("Operators.var_mean");
        FusionDefinition* fd = arg.fusion_definition;
        TORCH_CHECK(
            !fd->completed(),
            "Attempting to use a SchedOperators Op prior to definition!");
        size_t ndims = keepdim ? arg.dims : (arg.dims - axes.size());
        Tensor var = fd->defineTensor(ndims);
        Tensor mean = fd->defineTensor(ndims);
        fd->defineRecord(new VarianceMeanOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(var()), fd->recordingState(mean())},
            axes,
            correction,
            keepdim));
        return std::make_tuple(var, mean);
      },
      py::arg("axes"),
      py::arg("correction"),
      py::arg("keepdim") = false,
      py::return_value_policy::reference);
  nvf_ops.def(
      "uniform",
      [](FusionDefinition::Operators& self,
         Scalar minval,
         Scalar maxval,
         std::vector<Scalar>& shape,
         PrimDataType dtype) -> Tensor {
        FUSER_PERF_SCOPE("Operators.uniform");
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(shape.size());
        std::vector<State> output_shape_states(
            shape.size(), State(0, StateType::Scalar));
        std::transform(
            shape.begin(),
            shape.end(),
            output_shape_states.begin(),
            [&fd](const Scalar& s) { return fd->recordingState(s()); });
        fd->defineRecord(new RandomOpRecord(
            {
                fd->recordingState(minval()),
                fd->recordingState(maxval()),
            },
            {fd->recordingState(output())},
            output_shape_states,
            "ops.uniform",
            dtype));
        return output;
      },
      py::arg("minval"),
      py::arg("maxval"),
      py::arg("shape"),
      py::arg("dtype") = DataType::Float,
      py::return_value_policy::reference);
  nvf_ops.def(
      "normal",
      [](FusionDefinition::Operators& self,
         Scalar mean,
         Scalar std,
         std::vector<Scalar>& shape,
         PrimDataType dtype) -> Tensor {
        FUSER_PERF_SCOPE("Operators.normal");
        TORCH_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(shape.size());
        std::vector<State> output_shape_states(
            shape.size(), State(0, StateType::Scalar));
        std::transform(
            shape.begin(),
            shape.end(),
            output_shape_states.begin(),
            [&fd](const Scalar& s) { return fd->recordingState(s()); });
        fd->defineRecord(new RandomOpRecord(
            {
                fd->recordingState(mean()),
                fd->recordingState(std()),
            },
            {fd->recordingState(output())},
            output_shape_states,
            "ops.normal",
            dtype));
        return output;
      },
      py::arg("mean"),
      py::arg("std"),
      py::arg("shape"),
      py::arg("dtype") = DataType::Float,
      py::return_value_policy::reference);
  //! The ScedOperators class is a nested class of FusionDefinition to allow the
  //! user to query the class for the list of schedule operators.
  //!
  //! Example:
  //!   help(FusionDefinition.SchedOperators)
  //!
  //! Additional operators are expected to be defined below as needed.
  py::class_<FusionDefinition::SchedOperators> nvf_sched(
      fusion_def, "SchedOperators");
  nvf_sched.def(py::init<FusionDefinition*>());
  nvf_sched.def(
      "merge",
      [](FusionDefinition::SchedOperators& self, Tensor arg, int dim) {
        FUSER_PERF_SCOPE("SchedOperators.merge");
        TORCH_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        FusionDefinition* fd = self.fusion_definition;
        auto input_tv =
            fd->getFusionState(arg.index)->template as<TensorView>();
        input_tv->merge(dim);
      },
      py::arg("arg"),
      py::arg("dim"));
  nvf_sched.def(
      "split",
      [](FusionDefinition::SchedOperators& self,
         Tensor arg,
         int dim,
         unsigned int factor,
         bool inner_split,
         bool trim_out_of_bounds) {
        FUSER_PERF_SCOPE("SchedOperators.split");
        TORCH_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        FusionDefinition* fd = self.fusion_definition;
        auto input_tv =
            fd->getFusionState(arg.index)->template as<TensorView>();
        input_tv->split(dim, factor, inner_split, trim_out_of_bounds);
      },
      py::arg("arg"),
      py::arg("dim"),
      py::arg("factor"),
      py::arg("inner_split") = true,
      py::arg("trim_out_of_bounds") = false);
}

} // namespace nvfuser::python_frontend
