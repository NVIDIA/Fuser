// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/container.h>
#include <python_frontend/python_bindings.h>

namespace nvfuser::python_frontend {

namespace {
void bindIrContainer(py::module& nvfuser) {
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
