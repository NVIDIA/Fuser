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
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "add",
      add,
      R"(
Element-wise addition of two operands.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    The sum of the inputs.
)")
      NVFUSER_DIRECT_BINDING_BINARY_OP(
          "atan2",
          atan2,
          R"(
Element-wise arctangent of lhs/rhs choosing the quadrant.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    The angles in radians between the positive x-axis and a line to the (x, y) point.
)")
          NVFUSER_DIRECT_BINDING_BINARY_OP(
              "div",
              div,
              R"(
Element-wise division.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    The quotient of the division, truncated towards zero as per C++'s operator /.
)")
              NVFUSER_DIRECT_BINDING_BINARY_OP(
                  "truediv",
                  truediv,
                  R"(
Element-wise true (floating point) division.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    The floating point quotient.
)")
                  NVFUSER_DIRECT_BINDING_BINARY_OP(
                      "fmod",
                      fmod,
                      R"(
Element-wise floating-point mod.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    The floating-point mod.
)")
                      NVFUSER_DIRECT_BINDING_BINARY_OP(
                          "mul",
                          mul,
                          R"(
Element-wise multiplication.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    The product of the inputs.
)")
                          NVFUSER_DIRECT_BINDING_BINARY_OP(
                              "nextafter",
                              nextafter,
                              R"(
Return the next floating-point value after lhs towards rhs.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    The next representable values after lhs in the direction of rhs.
)")
                              NVFUSER_DIRECT_BINDING_BINARY_OP(
                                  "pow",
                                  pow,
                                  R"(
Element-wise power function.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    The bases raised to the exponents.
)")
                                  NVFUSER_DIRECT_BINDING_BINARY_OP(
                                      "remainder",
                                      remainder,
                                      R"(
Element-wise IEEE remainder.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    The IEEE remainder of the division.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("sub",
                                     sub,
                                     R"(
Element-wise subtraction.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    The difference of the inputs.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("minimum",
                                     minimum,
                                     R"(
Element-wise minimum.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    The smaller of each pair of elements.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("maximum",
                                     maximum,
                                     R"(
Element-wise maximum.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    The larger of each pair of elements.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("mod",
                                     mod,
                                     R"(
Element-wise modulo operation.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    The remainder after division.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("eq",
                                     eq,
                                     R"(
Element-wise equality comparison.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    True where elements are equal, False otherwise.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("ge",
                                     ge,
                                     R"(
Element-wise greater than or equal comparison.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    True where lhs >= rhs, False otherwise.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("gt",
                                     gt,
                                     R"(
Element-wise greater than comparison.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    True where lhs > rhs, False otherwise.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("le",
                                     le,
                                     R"(
Element-wise less than or equal comparison.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    True where lhs <= rhs, False otherwise.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("lt",
                                     lt,
                                     R"(
Element-wise less than comparison.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    True where lhs < rhs, False otherwise.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("ne",
                                     ne,
                                     R"(
Element-wise not equal comparison.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    True where elements are not equal, False otherwise.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("logical_and",
                                     logical_and,
                                     R"(
Element-wise logical AND.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    True where both inputs are True, False otherwise.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("logical_or",
                                     logical_or,
                                     R"(
Element-wise logical OR.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    True where either input is True, False otherwise.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("bitwise_and",
                                     bitwise_and,
                                     R"(
Element-wise bitwise AND.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    Bitwise AND of the inputs.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("bitwise_or",
                                     bitwise_or,
                                     R"(
Element-wise bitwise OR.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    Bitwise OR of the inputs.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("bitwise_xor",
                                     bitwise_xor,
                                     R"(
Element-wise bitwise XOR.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    Bitwise XOR of the inputs.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("bitwise_left_shift",
                                     bitwise_left_shift,
                                     R"(
Element-wise bitwise left shift.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    Values shifted left by specified amounts.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("bitwise_right_shift",
                                     bitwise_right_shift,
                                     R"(
Element-wise bitwise right shift.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    Values shifted right by specified amounts.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("logical_right_shift",
                                     logical_right_shift,
                                     R"(
Element-wise logical right shift.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    Values logically shifted right by specified amounts.
)") NVFUSER_DIRECT_BINDING_BINARY_OP("gcd",
                                     gcd,
                                     R"(
Element-wise greatest common divisor.

Parameters
----------
lhs : Val or TensorView
rhs : Val or TensorView

Returns
-------
Val or TensorView
    Greatest common divisor of each pair of elements.
)")
};

} // namespace

void bindOperations(py::module& nvfuser) {
  py::module_ nvf_ops = nvfuser.def_submodule(
      "ops", "This submodule contains all operations for NvFuser.");
  bindBinaryOps(nvf_ops);
}

} // namespace nvfuser::python
