// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <ops/all_ops.h>

namespace nvfuser::python {

namespace {

#define NVFUSER_DIRECT_BINDING_BINARY_OP(NAME, OP_NAME, DOCSTRING)             \
  ops.def(NAME, [](Val* lhs, Val* rhs) -> Val* {                               \
    return static_cast<Val* (*)(Val*, Val*)>(OP_NAME)(lhs, rhs);               \
  });                                                                          \
  ops.def(NAME, [](TensorView* lhs, Val* rhs) -> TensorView* {                 \
    return static_cast<TensorView* (*)(TensorView*, Val*)>(OP_NAME)(lhs, rhs); \
  });                                                                          \
  ops.def(NAME, [](Val* lhs, TensorView* rhs) -> TensorView* {                 \
    return static_cast<TensorView* (*)(Val*, TensorView*)>(OP_NAME)(lhs, rhs); \
  });                                                                          \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](TensorView* lhs, TensorView* rhs) -> TensorView* {                    \
        return static_cast<TensorView* (*)(TensorView*, TensorView*)>(         \
            OP_NAME)(lhs, rhs);                                                \
      },                                                                       \
      DOCSTRING);

void bindBinaryOps(py::module_& ops) {
  // Use the macro for add operation
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "add",
      add,
      R"(
Element-wise addition of two operands.

Parameters
----------
lhs : Val or TensorView
    The left-hand side operand.
rhs : Val or TensorView
    The right-hand side operand.

Returns
-------
Val or TensorView
    The sum of the inputs.
)")
};

} // namespace

void bindOperations(py::module& nvfuser) {
  py::module_ nvf_ops = nvfuser.def_submodule(
      "ops", "This submodule contains all operations for NvFuser.");
  bindBinaryOps(nvf_ops);
}

} // namespace nvfuser::python
