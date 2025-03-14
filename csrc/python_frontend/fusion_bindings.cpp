// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/base_nodes.h>
#include <ir/container.h>
#include <python_frontend/python_bindings.h>

namespace nvfuser::python_frontend {

namespace {
void bindIrContainer(py::module& nvfuser) {
  // Statement
  py::class_<nvfuser::Statement>(nvfuser, "Statement")
      .def(
          "name",
          &nvfuser::Statement::name,
          "Return the int that represents its name")
      .def(
          "is_val",
          &nvfuser::Statement::isVal,
          "Short cut to figure out if it is a value")
      .def(
          "is_expr",
          &nvfuser::Statement::isExpr,
          "Short cut to figure out if it is an expression")
      .def(
          "fusion",
          &nvfuser::Statement::fusion,
          "Return the fusion this statement belongs to")
      .def(
          "container",
          &nvfuser::Statement::container,
          "Return the container this statement belongs to")
      .def(
          "same_type",
          &nvfuser::Statement::sameType,
          "Return if this statement is the same type as another statement")
      .def(
          "same_as",
          &nvfuser::Statement::sameAs,
          "Return if this statement is the same as another statement")
      .def(
          "to_string",
          &nvfuser::Statement::toString,
          "Return the string representation of the statement");

  // Val
  py::class_<nvfuser::Val, nvfuser::Statement>(nvfuser, "Val")
      .def("vtype", &nvfuser::Val::vtype, "Return the ValType of the value")
      .def("dtype", &nvfuser::Val::dtype, "Return the DataType of the value")
      .def(
          "is_symbolic",
          &nvfuser::Val::isSymbolic,
          "Returns if the value is symbolic")
      .def(
          "is_scalar",
          &nvfuser::Val::isScalar,
          "Returns if the Val is a scalar")
      .def(
          "is_const_scalar",
          &nvfuser::Val::isConstScalar,
          "Returns if all dependencies are constant scalars")
      .def(
          "is_const_int",
          &nvfuser::Val::isConstInt,
          "Returns if all dependencies are constant integers")
      .def(
          "is_integral_scalar",
          &nvfuser::Val::isIntegralScalar,
          "Returns if it is an integral scalar")
      .def(
          "is_floating_point_scalar",
          &nvfuser::Val::isFloatingPointScalar,
          "Returns if it is a floating point scalar")
      .def("is_a_bool", &nvfuser::Val::isABool, "Returns if it is a boolean")
      .def(
          "evaluate",
          &nvfuser::Val::evaluate,
          "If this Val's history is comprised only of constant values, will return a PolymorphicValue.")
      .def(
          "is_const",
          &nvfuser::Val::isConst,
          "Returns if no dependencies and is a constant scalar.")
      .def("is_zero", &nvfuser::Val::isZero, "Returns if the value is zero")
      .def(
          "is_zero_int",
          &nvfuser::Val::isZeroInt,
          "Returns if the value is zero integer")
      .def("is_one", &nvfuser::Val::isOne, "Returns if the value is one")
      .def(
          "is_one_int",
          &nvfuser::Val::isOneInt,
          "Returns if the value is one integer")
      .def("is_true", &nvfuser::Val::isTrue, "Returns if the value is true")
      .def("is_false", &nvfuser::Val::isFalse, "Returns if the value is false")
      .def(
          "definition",
          &nvfuser::Val::definition,
          "Returns the Expr that this value is an output of, returns nullptr if none was found")
      .def(
          "uses",
          &nvfuser::Val::uses,
          "Returns the Exprs for which this is an input.")
      .def(
          "is_fusion_input",
          &nvfuser::Val::isFusionInput,
          "Returns if the value is a fusion input")
      .def(
          "is_fusion_output",
          &nvfuser::Val::isFusionOutput,
          "Returns if the value is a fusion output");

  // Expr
  py::class_<nvfuser::Expr, nvfuser::Statement>(nvfuser, "Expr")
      .def(
          "input",
          &nvfuser::Expr::input,
          py::arg("index"),
          "Returns the input at the given index.\n"
          "Args:\n"
          "    index (int): The index of the input to retrieve.")
      .def(
          "output",
          &nvfuser::Expr::output,
          py::arg("index"),
          "Returns the output at the given index.\n"
          "Args:\n"
          "    index (int): The index of the output to retrieve.")
      .def(
          "attribute_val",
          &nvfuser::Expr::attributeVal,
          "Returns the attribute value at the given index")
      .def(
          "same_op",
          &nvfuser::Expr::sameOp,
          "Check that if this and other are the same operator.")
      .def(
          "same_as",
          &nvfuser::Expr::sameAs,
          "Return if this and other are the same")
      .def(
          "get_op_string",
          &nvfuser::Expr::getOpString,
          "Get the name of an expression");

  py::class_<nvfuser::IrContainer>(nvfuser, "IrContainer")
      .def(py::init<>(), "Constructor for IrContainer")
      .def(
          "in_container",
          &nvfuser::IrContainer::inContainer,
          "Check if the statement is in the container.")
      .def(
          "deterministic_vals",
          &nvfuser::IrContainer::deterministic_vals,
          "Return values in insertion order.")
      .def(
          "deterministic_exprs",
          &nvfuser::IrContainer::deterministic_exprs,
          "Return expressions in insertion order.")
      .def(
          "deterministic_vals_map",
          &nvfuser::IrContainer::deterministic_vals_map,
          "Return mapping from value to integer ID.")
      .def(
          "deterministic_exprs_map",
          &nvfuser::IrContainer::deterministic_exprs_map,
          "Return mapping from expression to integer ID.")
      .def(
          "zero_val",
          (nvfuser::Val * (nvfuser::IrContainer::*)()) &
              nvfuser::IrContainer::zeroVal,
          "Return the zero value.")
      .def(
          "zero_val",
          (nvfuser::Val * (nvfuser::IrContainer::*)(nvfuser::DataType)) &
              nvfuser::IrContainer::zeroVal,
          "Return the zero value with the specified data type.")
      .def(
          "one_val",
          (nvfuser::Val * (nvfuser::IrContainer::*)()) &
              nvfuser::IrContainer::oneVal,
          "Return the one value.")
      .def(
          "one_val",
          (nvfuser::Val * (nvfuser::IrContainer::*)(nvfuser::DataType)) &
              nvfuser::IrContainer::oneVal,
          "Return the one value with the specified data type.")
      .def(
          "false_val",
          &nvfuser::IrContainer::falseVal,
          "Return the false value.")
      .def("true_val", &nvfuser::IrContainer::trueVal, "Return the true value.")
      .def(
          "magic_zero_val",
          &nvfuser::IrContainer::magicZeroVal,
          "Return the magic zero value.")
      .def(
          "metadata_of",
          &nvfuser::IrContainer::metadataOf,
          "Return the metadata of the value.")
      .def("axioms", &nvfuser::IrContainer::axioms, "Return the axioms.")
      .def(
          "assume_positive",
          &nvfuser::IrContainer::assumePositive,
          "Assume the value is positive.")
      .def(
          "assume_non_negative",
          &nvfuser::IrContainer::assumeNonNegative,
          "Assume the value is non-negative.")
      .def(
          "unordered_exprs",
          &nvfuser::IrContainer::unordered_exprs,
          "Return the set of unordered expressions.")
      .def("vals", &nvfuser::IrContainer::vals, "Return the set of values.");
}

} // namespace

void bindFusion(py::module& nvfuser) {
  bindIrContainer(nvfuser);
}

} // namespace nvfuser::python_frontend
