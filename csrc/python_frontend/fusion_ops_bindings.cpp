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

#define NVFUSER_DIRECT_BINDING_UNARY_OP(NAME, OP_NAME)                     \
  ops.def(                                                                    \
      NAME,                                                                   \
      static_cast<nvfuser::Val* (*)(nvfuser::Val*)>(nvfuser::OP_NAME));      \
  ops.def(                                                                    \
      NAME,                                                                   \
      static_cast<nvfuser::TensorView* (*)(nvfuser::TensorView*)>(           \
          nvfuser::OP_NAME))

void bindUnaryOps(py::module& ops) {
  NVFUSER_DIRECT_BINDING_UNARY_OP("abs", abs);
  NVFUSER_DIRECT_BINDING_UNARY_OP("acos", acos);
  NVFUSER_DIRECT_BINDING_UNARY_OP("acosh", acosh);
  NVFUSER_DIRECT_BINDING_UNARY_OP("asin", asin);
  NVFUSER_DIRECT_BINDING_UNARY_OP("asinh", asinh);
  NVFUSER_DIRECT_BINDING_UNARY_OP("atan", atan);
  NVFUSER_DIRECT_BINDING_UNARY_OP("atanh", atanh);
  NVFUSER_DIRECT_BINDING_UNARY_OP("ceil", ceil);
  NVFUSER_DIRECT_BINDING_UNARY_OP("cos", cos);
  NVFUSER_DIRECT_BINDING_UNARY_OP("cosh", cosh);
  NVFUSER_DIRECT_BINDING_UNARY_OP("exp", exp);
  NVFUSER_DIRECT_BINDING_UNARY_OP("exp2", exp2);
  NVFUSER_DIRECT_BINDING_UNARY_OP("expm1", expm1);
  NVFUSER_DIRECT_BINDING_UNARY_OP("erf", erf);
  NVFUSER_DIRECT_BINDING_UNARY_OP("erfc", erfc);
  NVFUSER_DIRECT_BINDING_UNARY_OP("erfinv", erfinv);
  NVFUSER_DIRECT_BINDING_UNARY_OP("erfcinv", erfcinv);
  NVFUSER_DIRECT_BINDING_UNARY_OP("floor", floor);
  NVFUSER_DIRECT_BINDING_UNARY_OP("frac", frac);
  NVFUSER_DIRECT_BINDING_UNARY_OP("lgamma", lgamma);
  NVFUSER_DIRECT_BINDING_UNARY_OP("log", log);
  NVFUSER_DIRECT_BINDING_UNARY_OP("log10", log10);
  NVFUSER_DIRECT_BINDING_UNARY_OP("log1p", log1p);
  NVFUSER_DIRECT_BINDING_UNARY_OP("log2", log2);
  NVFUSER_DIRECT_BINDING_UNARY_OP("neg", neg);
  NVFUSER_DIRECT_BINDING_UNARY_OP("logical_not", logical_not);
  NVFUSER_DIRECT_BINDING_UNARY_OP("bitwise_not", bitwise_not);
  NVFUSER_DIRECT_BINDING_UNARY_OP("relu", relu);
  NVFUSER_DIRECT_BINDING_UNARY_OP("rand_like", rand_like);
  NVFUSER_DIRECT_BINDING_UNARY_OP("randn_like", randn_like);
  NVFUSER_DIRECT_BINDING_UNARY_OP("reciprocal", reciprocal);
  NVFUSER_DIRECT_BINDING_UNARY_OP("round", round);
  NVFUSER_DIRECT_BINDING_UNARY_OP("rsqrt", rsqrt);
  NVFUSER_DIRECT_BINDING_UNARY_OP("set", set);
  NVFUSER_DIRECT_BINDING_UNARY_OP("segment_set", segment_set);
  NVFUSER_DIRECT_BINDING_UNARY_OP("sign", sign);
  NVFUSER_DIRECT_BINDING_UNARY_OP("sigmoid", sigmoid);
  NVFUSER_DIRECT_BINDING_UNARY_OP("signbit", signbit);
  NVFUSER_PYTHON_BINDING_UNARY_OP("silu", silu)
  NVFUSER_DIRECT_BINDING_UNARY_OP("sin", sin);
  NVFUSER_DIRECT_BINDING_UNARY_OP("sinh", sinh);
  NVFUSER_DIRECT_BINDING_UNARY_OP("sqrt", sqrt);
  NVFUSER_DIRECT_BINDING_UNARY_OP("tan", tan);
  NVFUSER_DIRECT_BINDING_UNARY_OP("tanh", tanh);
  NVFUSER_DIRECT_BINDING_UNARY_OP("trunc", trunc);
  NVFUSER_DIRECT_BINDING_UNARY_OP("isfinite", isfinite);
  NVFUSER_DIRECT_BINDING_UNARY_OP("isinf", isinf);
  NVFUSER_DIRECT_BINDING_UNARY_OP("isnan", isnan);
  NVFUSER_DIRECT_BINDING_UNARY_OP("isneginf", isneginf);
  NVFUSER_DIRECT_BINDING_UNARY_OP("isposinf", isposinf);
  NVFUSER_DIRECT_BINDING_UNARY_OP("isreal", isreal);
  NVFUSER_DIRECT_BINDING_UNARY_OP("real", real);
  NVFUSER_DIRECT_BINDING_UNARY_OP("imag", imag);
}

void bindBinaryOps(py::module& ops) {
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
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "atan2",
      atan2,
      R"(
Element-wise arctangent of lhs/rhs choosing the quadrant.

Parameters
----------
lhs : Val or TensorView
    The y-coordinates.
rhs : Val or TensorView
    The x-coordinates.

Returns
-------
Val or TensorView
    The angles in radians between the positive x-axis and the rays to the points (x,y).
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "div",
      div,
      R"(
Element-wise integer division.

Parameters
----------
lhs : Val or TensorView
    The dividend.
rhs : Val or TensorView
    The divisor.

Returns
-------
Val or TensorView
    The integer quotient.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "truediv",
      truediv,
      R"(
Element-wise true (floating point) division.

Parameters
----------
lhs : Val or TensorView
    The dividend.
rhs : Val or TensorView
    The divisor.

Returns
-------
Val or TensorView
    The floating point quotient.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "fmod",
      fmod,
      R"(
Element-wise floating-point remainder.

Parameters
----------
lhs : Val or TensorView
    The dividend.
rhs : Val or TensorView
    The divisor.

Returns
-------
Val or TensorView
    The floating-point remainder of the division.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "mul",
      mul,
      R"(
Element-wise multiplication.

Parameters
----------
lhs : Val or TensorView
    The first factor.
rhs : Val or TensorView
    The second factor.

Returns
-------
Val or TensorView
    The product of the inputs.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "nextafter",
      nextafter,
      R"(
Return the next floating-point value after lhs towards rhs.

Parameters
----------
lhs : Val or TensorView
    The starting values.
rhs : Val or TensorView
    The direction values.

Returns
-------
Val or TensorView
    The next representable values after lhs in the direction of rhs.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "pow",
      pow,
      R"(
Element-wise power function.

Parameters
----------
lhs : Val or TensorView
    The base values.
rhs : Val or TensorView
    The exponent values.

Returns
-------
Val or TensorView
    The bases raised to the exponents.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "remainder",
      remainder,
      R"(
Element-wise IEEE remainder.

Parameters
----------
lhs : Val or TensorView
    The dividend.
rhs : Val or TensorView
    The divisor.

Returns
-------
Val or TensorView
    The IEEE remainder of the division.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "sub",
      sub,
      R"(
Element-wise subtraction.

Parameters
----------
lhs : Val or TensorView
    The minuend.
rhs : Val or TensorView
    The subtrahend.

Returns
-------
Val or TensorView
    The difference of the inputs.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "minimum",
      minimum,
      R"(
Element-wise minimum.

Parameters
----------
lhs : Val or TensorView
    First input.
rhs : Val or TensorView
    Second input.

Returns
-------
Val or TensorView
    The smaller of each pair of elements.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "maximum",
      maximum,
      R"(
Element-wise maximum.

Parameters
----------
lhs : Val or TensorView
    First input.
rhs : Val or TensorView
    Second input.

Returns
-------
Val or TensorView
    The larger of each pair of elements.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "mod",
      mod,
      R"(
Element-wise modulo operation.

Parameters
----------
lhs : Val or TensorView
    The dividend.
rhs : Val or TensorView
    The divisor.

Returns
-------
Val or TensorView
    The remainder after division.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "eq",
      eq,
      R"(
Element-wise equality comparison.

Parameters
----------
lhs : Val or TensorView
    First input.
rhs : Val or TensorView
    Second input.

Returns
-------
Val or TensorView
    True where elements are equal, False otherwise.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "ge",
      ge,
      R"(
Element-wise greater than or equal comparison.

Parameters
----------
lhs : Val or TensorView
    First input.
rhs : Val or TensorView
    Second input.

Returns
-------
Val or TensorView
    True where lhs >= rhs, False otherwise.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "gt",
      gt,
      R"(
Element-wise greater than comparison.

Parameters
----------
lhs : Val or TensorView
    First input.
rhs : Val or TensorView
    Second input.

Returns
-------
Val or TensorView
    True where lhs > rhs, False otherwise.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "le",
      le,
      R"(
Element-wise less than or equal comparison.

Parameters
----------
lhs : Val or TensorView
    First input.
rhs : Val or TensorView
    Second input.

Returns
-------
Val or TensorView
    True where lhs <= rhs, False otherwise.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "lt",
      lt,
      R"(
Element-wise less than comparison.

Parameters
----------
lhs : Val or TensorView
    First input.
rhs : Val or TensorView
    Second input.

Returns
-------
Val or TensorView
    True where lhs < rhs, False otherwise.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "ne",
      ne,
      R"(
Element-wise not equal comparison.

Parameters
----------
lhs : Val or TensorView
    First input.
rhs : Val or TensorView
    Second input.

Returns
-------
Val or TensorView
    True where elements are not equal, False otherwise.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "logical_and",
      logical_and,
      R"(
Element-wise logical AND.

Parameters
----------
lhs : Val or TensorView
    First input.
rhs : Val or TensorView
    Second input.

Returns
-------
Val or TensorView
    True where both inputs are True, False otherwise.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "logical_or",
      logical_or,
      R"(
Element-wise logical OR.

Parameters
----------
lhs : Val or TensorView
    First input.
rhs : Val or TensorView
    Second input.

Returns
-------
Val or TensorView
    True where either input is True, False otherwise.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "bitwise_and",
      bitwise_and,
      R"(
Element-wise bitwise AND.

Parameters
----------
lhs : Val or TensorView
    First input.
rhs : Val or TensorView
    Second input.

Returns
-------
Val or TensorView
    Bitwise AND of the inputs.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "bitwise_or",
      bitwise_or,
      R"(
Element-wise bitwise OR.

Parameters
----------
lhs : Val or TensorView
    First input.
rhs : Val or TensorView
    Second input.

Returns
-------
Val or TensorView
    Bitwise OR of the inputs.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "bitwise_xor",
      bitwise_xor,
      R"(
Element-wise bitwise XOR.

Parameters
----------
lhs : Val or TensorView
    First input.
rhs : Val or TensorView
    Second input.

Returns
-------
Val or TensorView
    Bitwise XOR of the inputs.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "bitwise_left_shift",
      bitwise_left_shift,
      R"(
Element-wise bitwise left shift.

Parameters
----------
lhs : Val or TensorView
    Values to shift.
rhs : Val or TensorView
    Number of positions to shift by.

Returns
-------
Val or TensorView
    Values shifted left by specified amounts.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "bitwise_right_shift",
      bitwise_right_shift,
      R"(
Element-wise bitwise right shift.

Parameters
----------
lhs : Val or TensorView
    Values to shift.
rhs : Val or TensorView
    Number of positions to shift by.

Returns
-------
Val or TensorView
    Values shifted right by specified amounts.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "logical_right_shift",
      logical_right_shift,
      R"(
Element-wise logical right shift.

Parameters
----------
lhs : Val or TensorView
    Values to shift.
rhs : Val or TensorView
    Number of positions to shift by.

Returns
-------
Val or TensorView
    Values logically shifted right by specified amounts.
)");
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "gcd",
      gcd,
      R"(
Element-wise greatest common divisor.

Parameters
----------
lhs : Val or TensorView
    First input.
rhs : Val or TensorView
    Second input.

Returns
-------
Val or TensorView
    Greatest common divisor of each pair of elements.
)");
}

void bindOperations(py::module& fusion) {
    py::module ops = fusion.def_submodule("ops", "CPP Fusion Operations");
    bindUnaryOps(ops);
    bindBinaryOps(ops);
}

} // namespace nvfuser::python_frontend
