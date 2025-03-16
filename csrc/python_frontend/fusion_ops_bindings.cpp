// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ops/all_ops.h>
#include <python_frontend/python_bindings.h>

namespace nvfuser::python_frontend {

#define NVFUSER_DIRECT_BINDING_BINARY_OP(NAME, OP_NAME, DOCSTRING)            \
  ops.def(                                                                    \
      NAME,                                                                   \
      static_cast<nvfuser::Val* (*)(nvfuser::Val*, nvfuser::Val*)>(           \
          nvfuser::OP_NAME),                                                  \
      DOCSTRING);                                                             \
  ops.def(                                                                    \
      NAME,                                                                   \
      static_cast<nvfuser::TensorView* (*)(nvfuser::TensorView*,              \
                                           nvfuser::Val*)>(nvfuser::OP_NAME), \
      DOCSTRING);                                                             \
  ops.def(                                                                    \
      NAME,                                                                   \
      static_cast<nvfuser::TensorView* (*)(nvfuser::Val*,                     \
                                           nvfuser::TensorView*)>(            \
          nvfuser::OP_NAME),                                                  \
      DOCSTRING);                                                             \
  ops.def(                                                                    \
      NAME,                                                                   \
      static_cast<nvfuser::TensorView* (*)(nvfuser::TensorView*,              \
                                           nvfuser::TensorView*)>(            \
          nvfuser::OP_NAME),                                                  \
      DOCSTRING)

void bindOperations(py::module& fusion) {
  py::module ops = fusion.def_submodule("ops", "CPP Fusion Operations");

  // Use the macro for add operation
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "add",
      add,
      R"(
Element-wise addition of two operands.

This operation supports the following type combinations:
- scalar + scalar -> scalar
- tensor + scalar -> tensor
- scalar + tensor -> tensor
- tensor + tensor -> tensor

Parameters
----------
lhs : Val or TensorView
    The left-hand side operand.
rhs : Val or TensorView
    The right-hand side operand.

Returns
-------
Val or TensorView
    If both inputs are scalars (Val), returns a scalar.
    If either input is a tensor, returns a tensor.

Notes
-----
When using tensors, broadcasting is supported following NumPy rules.
)");
}

} // namespace nvfuser::python_frontend
