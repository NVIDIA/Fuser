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

#define NVFUSER_DIRECT_BINDING_UNARY_OP(NAME, OP_NAME, DOCSTRING)      \
  ops.def(NAME, [](Val* v) -> Val* {                                   \
    return static_cast<Val* (*)(Val*)>(OP_NAME)(v);                    \
  });                                                                  \
  ops.def(                                                             \
      NAME,                                                            \
      [](TensorView* tv) -> TensorView* {                              \
        return static_cast<TensorView* (*)(TensorView*)>(OP_NAME)(tv); \
      },                                                               \
      DOCSTRING,                                                       \
      py::return_value_policy::reference);

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
      DOCSTRING,                                                               \
      py::return_value_policy::reference);

#define NVFUSER_DIRECT_BINDING_TERNARY_OP(NAME, OP_NAME, DOCSTRING)            \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](Val* arg1, Val* arg2, Val* arg3) -> Val* {                            \
        return static_cast<Val* (*)(Val*, Val*, Val*)>(OP_NAME)(               \
            arg1, arg2, arg3);                                                 \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](TensorView* arg1,                                                     \
         TensorView* arg2,                                                     \
         TensorView* arg3) -> TensorView* {                                    \
        return static_cast<                                                    \
            TensorView* (*)(TensorView*, TensorView*, TensorView*)>(OP_NAME)(  \
            arg1, arg2, arg3);                                                 \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](TensorView* arg1, TensorView* arg2, Val* arg3) -> TensorView* {       \
        return static_cast<TensorView* (*)(TensorView*, TensorView*, Val*)>(   \
            OP_NAME)(arg1, arg2, arg3);                                        \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](TensorView* arg1, Val* arg2, TensorView* arg3) -> TensorView* {       \
        return static_cast<TensorView* (*)(TensorView*, Val*, TensorView*)>(   \
            OP_NAME)(arg1, arg2, arg3);                                        \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](Val* arg1, TensorView* arg2, TensorView* arg3) -> TensorView* {       \
        return static_cast<TensorView* (*)(Val*, TensorView*, TensorView*)>(   \
            OP_NAME)(arg1, arg2, arg3);                                        \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](Val* arg1, Val* arg2, TensorView* arg3) -> TensorView* {              \
        return static_cast<TensorView* (*)(Val*, Val*, TensorView*)>(OP_NAME)( \
            arg1, arg2, arg3);                                                 \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](TensorView* arg1, Val* arg2, Val* arg3) -> TensorView* {              \
        return static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(OP_NAME)( \
            arg1, arg2, arg3);                                                 \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](Val* arg1, TensorView* arg2, Val* arg3) -> TensorView* {              \
        return static_cast<TensorView* (*)(Val*, TensorView*, Val*)>(OP_NAME)( \
            arg1, arg2, arg3);                                                 \
      },                                                                       \
      DOCSTRING,                                                               \
      py::return_value_policy::reference);

#define NVFUSER_DIRECT_BINDING_THRESHOLD_LIKE_OP(NAME, OP_NAME, DOCSTRING)     \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](Val* arg1, Val* arg2, Val* arg3) -> Val* {                            \
        return static_cast<Val* (*)(Val*, Val*, Val*)>(OP_NAME)(               \
            arg1, arg2, arg3);                                                 \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](TensorView* arg1, Val* arg2, Val* arg3) -> TensorView* {              \
        return static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(OP_NAME)( \
            arg1, arg2, arg3);                                                 \
      },                                                                       \
      DOCSTRING,                                                               \
      py::return_value_policy::reference);

#define NVFUSER_DIRECT_BINDING_TERNARY_WITH_ALPHA_OP(NAME, OP_NAME, DOCSTRING) \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](Val* arg1, Val* arg2, Val* arg3, Val* arg4) -> Val* {                 \
        return static_cast<Val* (*)(Val*, Val*, Val*, Val*)>(OP_NAME)(         \
            arg1, arg2, arg3, arg4);                                           \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](TensorView* arg1, TensorView* arg2, TensorView* arg3, Val* arg4)      \
          -> TensorView* {                                                     \
        return static_cast<                                                    \
            TensorView* (*)(TensorView*, TensorView*, TensorView*, Val*)>(     \
            OP_NAME)(arg1, arg2, arg3, arg4);                                  \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](TensorView* arg1, TensorView* arg2, Val* arg3, Val* arg4)             \
          -> TensorView* {                                                     \
        return static_cast<                                                    \
            TensorView* (*)(TensorView*, TensorView*, Val*, Val*)>(OP_NAME)(   \
            arg1, arg2, arg3, arg4);                                           \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](TensorView* arg1, Val* arg2, TensorView* arg3, Val* arg4)             \
          -> TensorView* {                                                     \
        return static_cast<TensorView* (*)(TensorView*, Val*, Val*, Val*)>(    \
            OP_NAME)(arg1, arg2, arg3, arg4);                                  \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](Val* arg1, TensorView* arg2, TensorView* arg3, Val* arg4)             \
          -> TensorView* {                                                     \
        return static_cast<                                                    \
            TensorView* (*)(Val*, TensorView*, TensorView*, Val*)>(OP_NAME)(   \
            arg1, arg2, arg3, arg4);                                           \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](Val* arg1, Val* arg2, TensorView* arg3, Val* arg4) -> TensorView* {   \
        return static_cast<TensorView* (*)(Val*, Val*, TensorView*, Val*)>(    \
            OP_NAME)(arg1, arg2, arg3, arg4);                                  \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](TensorView* arg1, Val* arg2, Val* arg3, Val* arg4) -> TensorView* {   \
        return static_cast<TensorView* (*)(TensorView*, Val*, Val*, Val*)>(    \
            OP_NAME)(arg1, arg2, arg3, arg4);                                  \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  ops.def(                                                                     \
      NAME,                                                                    \
      [](Val* arg1, TensorView* arg2, Val* arg3, Val* arg4) -> TensorView* {   \
        return static_cast<TensorView* (*)(Val*, TensorView*, Val*, Val*)>(    \
            OP_NAME)(arg1, arg2, arg3, arg4);                                  \
      },                                                                       \
      DOCSTRING,                                                               \
      py::return_value_policy::reference);

#define NVFUSER_DIRECT_BINDING_REDUCTION_OP(NAME, OP_NAME, DOCSTRING)   \
  ops.def(                                                              \
      NAME,                                                             \
      [](TensorView* arg, PrimDataType dtype) -> TensorView* {          \
        std::vector<int64_t> dims(arg->nDims());                        \
        std::iota(dims.begin(), dims.end(), 0);                         \
        return static_cast<TensorView* (*)(TensorView*,                 \
                                           const std::vector<int64_t>&, \
                                           bool,                        \
                                           DataType)>(OP_NAME)(         \
            arg, dims, /*keep_dim=*/false, dtype);                      \
      },                                                                \
      py::arg("arg"),                                                   \
      py::arg("dtype") = DataType::Null,                                \
      py::return_value_policy::reference);                              \
  ops.def(                                                              \
      NAME,                                                             \
      [](TensorView* arg, int dim, bool keep_dim, PrimDataType dtype)   \
          -> TensorView* {                                              \
        return static_cast<TensorView* (*)(TensorView*,                 \
                                           const std::vector<int64_t>&, \
                                           bool,                        \
                                           DataType)>(OP_NAME)(         \
            arg, {dim}, keep_dim, dtype);                               \
      },                                                                \
      py::arg("arg"),                                                   \
      py::arg("dim"),                                                   \
      py::arg("keep_dim") = false,                                      \
      py::arg("dtype") = DataType::Null,                                \
      py::return_value_policy::reference);                              \
  ops.def(                                                              \
      NAME,                                                             \
      [](TensorView* arg,                                               \
         const std::vector<int64_t>& dims,                              \
         bool keep_dim,                                                 \
         PrimDataType dtype) -> TensorView* {                           \
        return static_cast<TensorView* (*)(TensorView*,                 \
                                           const std::vector<int64_t>&, \
                                           bool,                        \
                                           DataType)>(OP_NAME)(         \
            arg, dims, keep_dim, dtype);                                \
      },                                                                \
      py::arg("arg"),                                                   \
      py::arg("dims"),                                                  \
      py::arg("keep_dim") = false,                                      \
      py::arg("dtype") = DataType::Null,                                \
      DOCSTRING,                                                        \
      py::return_value_policy::reference);

void bindUnaryOps(py::module_& ops) {
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "abs",
      abs,
      R"(
Element-wise absolute value.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The absolute value of the input.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "acos",
      acos,
      R"(
Element-wise inverse cosine.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The inverse cosine of the input in radians.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "acosh",
      acosh,
      R"(
Element-wise inverse hyperbolic cosine.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The inverse hyperbolic cosine of the input.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "asin",
      asin,
      R"(
Element-wise inverse sine.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The inverse sine of the input in radians.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "asinh",
      asinh,
      R"(
Element-wise inverse hyperbolic sine.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The inverse hyperbolic sine of the input.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "atan",
      atan,
      R"(
Element-wise inverse tangent.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The inverse tangent of the input in radians.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "atanh",
      atanh,
      R"(
Element-wise inverse hyperbolic tangent.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The inverse hyperbolic tangent of the input.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "ceil",
      ceil,
      R"(
Element-wise ceiling function.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The smallest integer greater than or equal to each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "cos",
      cos,
      R"(
Element-wise cosine.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The cosine of the input.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "cosh",
      cosh,
      R"(
Element-wise hyperbolic cosine.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The hyperbolic cosine of the input.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "exp",
      exp,
      R"(
Element-wise exponential function.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    e raised to the power of each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "exp2",
      exp2,
      R"(
Element-wise base-2 exponential function.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    2 raised to the power of each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "expm1",
      expm1,
      R"(
Element-wise exponential minus 1.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    exp(x) - 1 for each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "erf",
      erf,
      R"(
Element-wise error function.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The error function of each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "erfc",
      erfc,
      R"(
Element-wise complementary error function.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    1 - erf(x) for each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "erfinv",
      erfinv,
      R"(
Element-wise inverse error function.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The inverse error function of each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "erfcinv",
      erfcinv,
      R"(
Element-wise inverse complementary error function.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The inverse complementary error function of each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "floor",
      floor,
      R"(
Element-wise floor function.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The largest integer less than or equal to each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "frac",
      frac,
      R"(
Element-wise fractional part.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The fractional part of each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "lgamma",
      lgamma,
      R"(
Element-wise natural logarithm of the absolute value of the gamma function.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The natural logarithm of the absolute value of the gamma function.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "log",
      log,
      R"(
Element-wise natural logarithm.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The natural logarithm of each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "log10",
      log10,
      R"(
Element-wise base-10 logarithm.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The base-10 logarithm of each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "log1p",
      log1p,
      R"(
Element-wise natural logarithm of 1 plus x.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    log(1 + x) for each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "log2",
      log2,
      R"(
Element-wise base-2 logarithm.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The base-2 logarithm of each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "neg",
      neg,
      R"(
Element-wise negation.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The negative of each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "logical_not",
      logical_not,
      R"(
Element-wise logical NOT.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    True where input is False, False where input is True.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "bitwise_not",
      bitwise_not,
      R"(
Element-wise bitwise NOT.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The bitwise NOT of each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "relu",
      relu,
      R"(
Element-wise rectified linear unit.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    max(0, x) for each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "rand_like",
      rand_like,
      R"(
Generate random values with the same shape as input.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    Random values with the same shape as input.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "randn_like",
      randn_like,
      R"(
Generate random values from a normal distribution with the same shape as input.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    Random values from a normal distribution with the same shape as input.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "reciprocal",
      reciprocal,
      R"(
Element-wise reciprocal.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    1/x for each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "round",
      round,
      R"(
Element-wise rounding to nearest integer.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    Each element rounded to the nearest integer.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "rsqrt",
      rsqrt,
      R"(
Element-wise reciprocal square root.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    1/sqrt(x) for each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "set",
      set,
      R"(
Element-wise identity operation.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    A copy of the input.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "segment_set",
      segment_set,
      R"(
Element-wise identity operation, forces a segmentation between the producer and
consumer in generated kernel.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    Tensor with values set in the specified segment.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "sign",
      sign,
      R"(
Element-wise sign function.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    1 for positive values, -1 for negative values, 0 for zero.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "sigmoid",
      sigmoid,
      R"(
Element-wise sigmoid function.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    1/(1 + exp(-x)) for each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "signbit",
      signbit,
      R"(
Element-wise sign bit.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    True where the sign bit is set, False otherwise.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "silu",
      silu,
      R"(
Element-wise SiLU (Swish) activation function.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    x * sigmoid(x) for each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "sin",
      sin,
      R"(
Element-wise sine.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The sine of the input.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "sinh",
      sinh,
      R"(
Element-wise hyperbolic sine.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The hyperbolic sine of the input.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "sqrt",
      sqrt,
      R"(
Element-wise square root.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The square root of each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "tan",
      tan,
      R"(
Element-wise tangent.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The tangent of the input.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "tanh",
      tanh,
      R"(
Element-wise hyperbolic tangent.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The hyperbolic tangent of the input.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "trunc",
      trunc,
      R"(
Element-wise truncation.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The truncated value of each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "isfinite",
      isfinite,
      R"(
Element-wise finite check.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    True where the element is finite, False otherwise.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "isinf",
      isinf,
      R"(
Element-wise infinity check.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    True where the element is infinite, False otherwise.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "isnan",
      isnan,
      R"(
Element-wise NaN check.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    True where the element is NaN, False otherwise.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "isneginf",
      isneginf,
      R"(
Element-wise negative infinity check.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    True where the element is negative infinity, False otherwise.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "isposinf",
      isposinf,
      R"(
Element-wise positive infinity check.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    True where the element is positive infinity, False otherwise.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "isreal",
      isreal,
      R"(
Element-wise real number check.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    True where the element is a real number, False otherwise.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "real",
      real,
      R"(
Element-wise real part of complex number.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The real part of each element.
)")
  NVFUSER_DIRECT_BINDING_UNARY_OP(
      "imag",
      imag,
      R"(
Element-wise imaginary part of complex number.

Parameters
----------
x : Val or TensorView

Returns
-------
Val or TensorView
    The imaginary part of each element.
)")
}

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

void bindTernaryOps(py::module_& ops) {
  NVFUSER_DIRECT_BINDING_TERNARY_OP("lerp", lerp, R"(
Element-wise linear interpolation.

Parameters
----------
x : Val or TensorView
y : Val or TensorView
weight : Val or TensorView

Returns
-------
Val or TensorView
    Linear interpolation of the inputs.
)")

  NVFUSER_DIRECT_BINDING_TERNARY_OP("where", where, R"(
Select elements from either input or other tensors based on condition.

Parameters
----------
condition : Val or TensorView
x : Val or TensorView
y : Val or TensorView

Returns
-------
Val or TensorView
    Elements from x if condition is True, otherwise elements from y.
)")

  NVFUSER_DIRECT_BINDING_THRESHOLD_LIKE_OP("clamp", clamp, R"(
Clamps all elements in input into the range [ min, max ]

Parameters
----------
input : Val or TensorView
min : Val or TensorView
max : Val or TensorView

Returns
-------
Val or TensorView
    Clamped values.
)")

  NVFUSER_DIRECT_BINDING_THRESHOLD_LIKE_OP("threshold", threshold, R"(
Thresholds each element of the input Tensor.

Parameters
----------
input : Val or TensorView
threshold : Val or TensorView
value : Val or TensorView

Returns
-------
Val or TensorView
    Thresholded values.
)")

  NVFUSER_DIRECT_BINDING_TERNARY_WITH_ALPHA_OP("addcmul", addcmul, R"(
Element-wise multiplication of input1 and input2,
then adds alpha * input3 to the result.

Parameters
----------
input1 : Val or TensorView
input2 : Val or TensorView
input3 : Val or TensorView
alpha : Val

Returns
-------
Val or TensorView
    The result of the element-wise multiplication and addition.
)")
}

void bindReductionOps(py::module_& ops) {
  NVFUSER_DIRECT_BINDING_REDUCTION_OP(
      "max",
      max,
      R"(
Reduce a tensor by computing the maximum value along specified dimensions.

Parameters
----------
arg : TensorView
    Input tensor to reduce.
dim : int, optional
    Dimension to reduce over. If not specified, reduces over all dimensions.
keep_dim : bool, optional
    Whether to keep the reduced dimensions with size 1. Default is False.
dtype : PrimDataType, optional
    This argument is not used for max.

Returns
-------
TensorView
    A new tensor containing the maximum values along the specified dimensions.
)")
  NVFUSER_DIRECT_BINDING_REDUCTION_OP(
      "min",
      min,
      R"(
Reduce a tensor by computing the minimum value along specified dimensions.

Parameters
----------
arg : TensorView
    Input tensor to reduce.
dim : int, optional
    Dimension to reduce over. If not specified, reduces over all dimensions.
keep_dim : bool, optional
    Whether to keep the reduced dimensions with size 1. Default is False.
dtype : PrimDataType, optional
    This argument is not used for min.

Returns
-------
TensorView
    A new tensor containing the minimum values along the specified dimensions.
)")
  NVFUSER_DIRECT_BINDING_REDUCTION_OP(
      "prod",
      prod,
      R"(
Reduce a tensor by computing the product of elements along specified dimensions.

Parameters
----------
arg : TensorView
    Input tensor to reduce.
dim : int, optional
    Dimension to reduce over. If not specified, reduces over all dimensions.
keep_dim : bool, optional
    Whether to keep the reduced dimensions with size 1. Default is False.
dtype : PrimDataType, optional
    The data type to cast the arg to before computation. If the dtype argument
    is None, use the input data type if it is floating point. Otherwise, it is
    DataType::Int for boolean or integral input.

Returns
-------
TensorView
    A new tensor containing the product of elements along the specified dimensions.
)")
  NVFUSER_DIRECT_BINDING_REDUCTION_OP(
      "sum",
      sum,
      R"(
Reduce a tensor by computing the sum of elements along specified dimensions.

Parameters
----------
arg : TensorView
    Input tensor to reduce.
dim : int, optional
    Dimension to reduce over. If not specified, reduces over all dimensions.
keep_dim : bool, optional
    Whether to keep the reduced dimensions with size 1. Default is False.
dtype : PrimDataType, optional
    The data type to cast the arg to before computation. If the dtype argument
    is None, use the input data type if it is floating point. Otherwise, it is
    DataType::Int for boolean or integral input.

Returns
-------
TensorView
    A new tensor containing the sum of elements along the specified dimensions.
)")
}

void bindCastOps(py::module_& ops) {
  ops.def(
      "cast",
      [](TensorView* arg, PrimDataType dtype) -> TensorView* {
        return static_cast<TensorView* (*)(DataType, TensorView*)>(castOp)(
            dtype, arg);
      },
      py::arg("arg"),
      py::arg("dtype"),
      R"(
Cast a tensor to a different data type.

Parameters
----------
arg : TensorView
    Input tensor to cast.
dtype : PrimDataType
    Target data type for the cast operation.

Returns
-------
TensorView
    A new tensor with the specified data type.
)",
      py::return_value_policy::reference);
  ops.def(
      "cast",
      [](Val* arg, PrimDataType dtype) -> Val* {
        return static_cast<Val* (*)(DataType, Val*)>(castOp)(dtype, arg);
      },
      py::arg("arg"),
      py::arg("dtype"),
      R"(
Cast a scalar value to a different data type.

Parameters
----------
arg : Val
    Input scalar value to cast.
dtype : PrimDataType
    Target data type for the cast operation.

Returns
-------
Val
    A new scalar value with the specified data type.
)",
      py::return_value_policy::reference);
}

void bindMatmulOps(py::module_& ops) {
  ops.def(
      "matmul",
      static_cast<TensorView* (*)(TensorView*, TensorView*)>(matmul),
      py::arg("arg1"),
      py::arg("arg2"),
      R"(
The matrix product of two tensors.

Parameters
----------
arg1 : TensorView
arg2 : TensorView

Returns
-------
TensorView
    The result of the matrix multiplication.
)",
      py::return_value_policy::reference);
  ops.def(
      "linear",
      [](TensorView* arg1,
         TensorView* arg2,
         std::optional<TensorView*> bias = std::nullopt) -> TensorView* {
        if (bias.has_value()) {
          return static_cast<
              TensorView* (*)(TensorView*, TensorView*, TensorView*)>(linear)(
              arg1, arg2, bias.value());
        } else {
          return static_cast<TensorView* (*)(TensorView*, TensorView*)>(linear)(
              arg1, arg2);
        }
      },
      py::arg("arg1"),
      py::arg("arg2"),
      py::arg("bias") = std::nullopt,
      R"(
Applies an affine linear transformation to the incoming data:
output = arg1 @ transpose(arg2) + bias.

Parameters
----------
arg1 : TensorView
arg2 : TensorView
bias : TensorView, optional
    The bias vector to add to the output. If not provided, the bias is not added.

Returns
-------
TensorView
    The result of the affine linear transformation.
)",
      py::return_value_policy::reference);
}

template <class ITERABLE>
std::vector<Val*> define_vector_fn(ITERABLE& values, bool shape_check) {
  std::vector<Val*> args;
  size_t idx = 0;
  for (const auto& item : values) {
    if (py::isinstance<py::int_>(item)) {
      auto int_value = py::cast<int64_t>(item);
      NVF_CHECK(
          !shape_check || int_value >= -1,
          "The value ",
          int_value,
          " at index ",
          idx,
          " was neither symbolic(-1), zero_element(0), broadcast(1), or "
          "static(>1).");
      args.emplace_back(IrBuilder::create<Val>(int_value, DataType::Int));
    } else if (py::isinstance<Val>(item)) {
      args.emplace_back(py::cast<Val*>(item));
    } else {
      NVF_CHECK(
          false,
          "Unsupported iterable object type for define_vector! Index:",
          idx);
    }
    ++idx;
  }
  return args;
}

template <class ShapeType>
std::vector<Val*> SequenceAsVector(ShapeType shape, bool shape_check = true) {
  static_assert(
      std::is_same_v<ShapeType, py::list> ||
      std::is_same_v<ShapeType, py::tuple>);
  return define_vector_fn<ShapeType>(shape, /*shape_check=*/shape_check);
}

template <class ShapeType>
TensorView* reshape_fn(TensorView* arg, ShapeType generic_new_shape) {
  return reshape(arg, SequenceAsVector(generic_new_shape));
}

template <class ShapeType>
TensorView* expand_fn(TensorView* arg, ShapeType generic_new_shape) {
  return expand(arg, SequenceAsVector(generic_new_shape));
}

template <class ShapeType>
TensorView* broadcast_in_dim_fn(
    TensorView* arg,
    ShapeType generic_output_shape,
    std::vector<int64_t>& broadcast_dims) {
  std::vector<Val*> output_shape = SequenceAsVector(generic_output_shape);
  NVF_CHECK(
      output_shape.size() >= broadcast_dims.size(),
      "broadcast_dims vector size is too big for output shape!");

  const auto arg_ndims = arg->domain()->noReductions().size();
  NVF_CHECK(
      output_shape.size() >= broadcast_dims.size(),
      "The new shape is expected to be greater-then-or-equal to the input: ",
      output_shape.size(),
      " vs ",
      arg_ndims);
  NVF_CHECK(
      arg_ndims == broadcast_dims.size(),
      "The broadcast dimensions should match the input dimensions: ",
      arg_ndims,
      " vs ",
      broadcast_dims.size(),
      ". arg = ",
      arg->toString());

  std::vector<bool> is_broadcast_dim(output_shape.size(), true);
  for (const auto idx : arange(broadcast_dims.size())) {
    if (idx > 0) {
      NVF_CHECK(
          broadcast_dims[idx - 1] < broadcast_dims[idx],
          "Broadcast dimension is not greater than the previous value.");
    }
    NVF_CHECK(
        broadcast_dims[idx] < static_cast<int>(output_shape.size()),
        "Invalid broadcast_dims value.");
    is_broadcast_dim.at(broadcast_dims[idx]) = false;
  }

  auto bcast_output = broadcast(arg, is_broadcast_dim);
  return expand(bcast_output, output_shape);
}

template <class ShapeType>
TensorView* slice_fn(
    TensorView* arg,
    ShapeType start,
    ShapeType end,
    std::optional<ShapeType> strides,
    bool manual_normalization) {
  std::vector<Val*> start_vec = SequenceAsVector(start, /*shape_check=*/false);
  std::vector<Val*> end_vec = SequenceAsVector(end, /*shape_check=*/false);

  std::vector<Val*> stride_vec;
  if (strides.has_value()) {
    stride_vec = SequenceAsVector(strides.value(), /*shape_check=*/false);
    NVF_CHECK(
        start_vec.size() == stride_vec.size(),
        "Slice start_indices and strides don't match! Start Indices: ",
        start_vec.size(),
        " Strides: ",
        stride_vec.size());
  } else {
    // set stride with default value;
    stride_vec.reserve(start_vec.size());
    for ([[maybe_unused]] auto i : arange(start_vec.size())) {
      stride_vec.push_back(IrBuilder::create<Val>(1, DataType::Int));
    }
  }

  NVF_CHECK(
      arg->nDims() == (int64_t)start_vec.size(),
      "Number of tensor dimensions does not match slice dimensions! "
      "Tensor-dims: ",
      arg->nDims(),
      " Slice-dims: ",
      start_vec.size());
  NVF_CHECK(
      start_vec.size() == end_vec.size(),
      "Slice indexing attribute dimensions don't match! Start Indices: ",
      start_vec.size(),
      " End Indices: ",
      end_vec.size());

  std::vector<Slice> vec_slice;
  for (const auto idx : arange(arg->domain()->noReductions().size())) {
    // NOTE: there's an extra move, we can use emplace_back if we go write
    // some constructors for Slice.
    Val* start_idx = start_vec.at(idx);
    Val* end_idx = end_vec.at(idx);
    Val* stride_idx = stride_vec.at(idx);
    NVF_CHECK(
        !start_idx->isConstInt() || start_idx->evaluate().as<int64_t>() >= 0,
        "Slice operation start_indices must be greater than or equal to 0. "
        "Start Indices: ",
        start_idx->evaluate().as<int64_t>());
    NVF_CHECK(
        !start_idx->isConstInt() || !end_idx->isConstInt() ||
            end_idx->evaluate().as<int64_t>() >=
                start_idx->evaluate().as<int64_t>(),
        "Slice operation end_indices must be greater than or equal to "
        "start_indices. Start Indices: ",
        start_idx->evaluate().as<int64_t>(),
        " End Indices: ",
        end_idx->evaluate().as<int64_t>());
    NVF_CHECK(
        stride_idx->isConstInt() && stride_idx->evaluate().as<int64_t>() == 1,
        "nvFuser Limitation: All slice operation strides must be of const "
        "size 1.");
    vec_slice.push_back({start_idx, end_idx, stride_idx});
  }
  return slice(arg, vec_slice, manual_normalization);
}

void bindMetadataOps(py::module_& ops) {
  ops.def(
         "reshape",
         reshape_fn<py::list>,
         py::arg("arg"),
         py::arg("new_shape"),
         R"(
Reshape a tensor to a new shape.

Parameters
----------
arg : TensorView
new_shape : list or tuple
    The new shape of the tensor.

Returns
-------
TensorView
    The reshaped tensor.
      )",
         py::return_value_policy::reference)
      .def(
          "reshape",
          reshape_fn<py::tuple>,
          py::arg("arg"),
          py::arg("new_shape"),
          R"(
Reshape a tensor to a new shape.

Parameters
----------
arg : TensorView
new_shape : list or tuple
    The new shape of the tensor.

Returns
-------
TensorView
    The reshaped tensor.
      )",
          py::return_value_policy::reference);
  ops.def(
      "permute",
      [](TensorView* arg, std::vector<int64_t>& dims) -> TensorView* {
        NVF_CHECK(
            arg->nDims() == (int64_t)dims.size(),
            "Operator permute expects `dims` argument to have the same length "
            "as input!");
        return permute(arg, dims);
      },
      py::arg("arg"),
      py::arg("dims"),
      R"(
Permute a tensor.

Parameters
----------
arg : TensorView
dims : list or tuple
    The dimensions to permute.

Returns
-------
TensorView
    The permuted tensor.
)",
      py::return_value_policy::reference);
  ops.def(
      "expand",
      expand_fn<py::list>,
      py::arg("arg"),
      py::arg("shape"),
      R"(
Expand a tensor to a new shape.

Parameters
----------
arg : TensorView
shape : list or tuple
    The new shape of the tensor.

Returns
-------
TensorView
    The expanded tensor.
)",
      py::return_value_policy::reference);
  ops.def(
      "expand",
      expand_fn<py::tuple>,
      py::arg("arg"),
      py::arg("shape"),
      R"(
Expand a tensor to a new shape.

Parameters
----------
arg : TensorView
shape : list or tuple
    The new shape of the tensor.

Returns
-------
TensorView
    The expanded tensor.
)",
      py::return_value_policy::reference);
  ops.def(
      "squeeze",
      [](TensorView* arg,
         std::vector<int64_t> dims,
         const bool squeeze_expanded) -> TensorView* {
        return squeeze(arg, dims, squeeze_expanded);
      },
      py::arg("arg"),
      py::arg("dims"),
      py::arg("squeeze_expanded") = false,
      py::return_value_policy::reference,
      R"(
Reduce a tensor by removing specified dimensions.

Parameters
----------
arg : TensorView
dims : list or tuple
    The dimensions to remove.
squeeze_expanded : bool, optional
    Whether to squeeze expanded dimensions. Default is False.

Returns
-------
TensorView
    The squeezed tensor.
)",
      py::return_value_policy::reference);
  ops.def(
      "broadcast",
      [](TensorView* arg, std::vector<bool>& is_broadcast_dim) -> TensorView* {
        return broadcast(arg, is_broadcast_dim);
      },
      py::arg("arg"),
      py::arg("is_broadcast_dim"),
      R"(
Broadcast a tensor to a new shape.

Parameters
----------
arg : TensorView
is_broadcast_dim : list or tuple
    The dimensions to broadcast.

Returns
-------
TensorView
    The broadcasted tensor.
)",
      py::return_value_policy::reference);
  ops.def(
      "broadcast_in_dim",
      broadcast_in_dim_fn<py::list>,
      py::arg("arg"),
      py::arg("shape"),
      py::arg("broadcast_dims"),
      R"(
Broadcast a tensor to a new shape.

Parameters
----------
arg : TensorView
shape : list or tuple
    The new shape of the tensor.
broadcast_dims : list or tuple
    The dimensions to broadcast.

Returns
-------
TensorView
    The broadcasted tensor.
)",
      py::return_value_policy::reference);
  ops.def(
      "broadcast_in_dim",
      broadcast_in_dim_fn<py::tuple>,
      py::arg("arg"),
      py::arg("shape"),
      py::arg("broadcast_dims"),
      R"(
Broadcast a tensor to a new shape.

Parameters
----------
arg : TensorView
shape : list or tuple
    The new shape of the tensor.
broadcast_dims : list or tuple
    The dimensions to broadcast.

Returns
-------
TensorView
    The broadcasted tensor.
)",
          py::return_value_policy::reference);
  ops.def(
          "slice",
          slice_fn<py::list>,
          py::arg("arg"),
          py::arg("start_indices"),
          py::arg("end_indices"),
          py::arg("strides") = py::none(),
          py::arg("manual_normalization") = false,
          R"(
Slice a tensor.

Parameters
----------
arg : TensorView
start_indices : list or tuple
end_indices : list or tuple
strides : list or tuple, optional
    The strides of the slice. Default is None.
manual_normalization : bool, optional
    Whether to normalize the slice. Default is False.

Returns
-------
TensorView
    The sliced tensor.
      )",
          py::return_value_policy::reference);
  ops.def(
          "slice",
          slice_fn<py::tuple>,
          py::arg("arg"),
          py::arg("start_indices"),
          py::arg("end_indices"),
          py::arg("strides") = py::none(),
          py::arg("manual_normalization") = false,
          R"(
Slice a tensor.

Parameters
----------
arg : TensorView
start_indices : list or tuple
end_indices : list or tuple
strides : list or tuple, optional
    The strides of the slice. Default is None.
manual_normalization : bool, optional
    Whether to normalize the slice. Default is False.

Returns
-------
TensorView
    The sliced tensor.
      )",
          py::return_value_policy::reference);
}

void bindTensorUtilityOps(py::module_& ops) {
  ops.def(
      "size",
      [](TensorView* arg, int64_t dim) -> Val* { return size(arg, dim); },
      py::arg("arg"),
      py::arg("dim"),
      R"(
Get the size of a tensor.

Parameters
----------
arg : TensorView
dim : int
    The dimension to get the size of.

Returns
-------
int
    The size of the dimension.
)",
      py::return_value_policy::reference);
  ops.def(
      "shape",
      [](TensorView* arg) { return shape(arg); },
      py::return_value_policy::reference,
      R"(
Get the shape of a tensor.

Returns
-------
list of Val
    The shape of the tensor.
)");
}

void bindIndexingOps(py::module_& ops) {
  ops.def(
         "index_select",
         [](TensorView* arg, TensorView* index, int64_t dim) -> TensorView* {
           return indexSelect(arg, dim, index);
         },
         py::arg("arg"),
         py::arg("index"),
         py::arg("dim"),
         py::return_value_policy::reference,
         R"(
Select elements from a tensor along a specified dimension.

Parameters
----------
arg : TensorView
index : TensorView
dim : int
    The dimension to select along.

Returns
-------
TensorView
    The selected tensor.
)")
      .def(
          "select",
          [](TensorView* arg, Val* index, int64_t dim) -> TensorView* {
            return select(arg, dim, index);
          },
          py::arg("arg"),
          py::arg("index"),
          py::arg("dim"),
          py::return_value_policy::reference,
          R"(
Select elements from a tensor along a specified dimension.

Parameters
----------
arg : TensorView
index : TensorView
dim : int
    The dimension to select along.

Returns
-------
TensorView
    The selected tensor.
)");
}

} // namespace

void bindOperations(py::module& nvfuser) {
  py::module_ nvf_ops = nvfuser.def_submodule(
      "ops", "This submodule contains all operations for NvFuser.");
  bindUnaryOps(nvf_ops);
  bindBinaryOps(nvf_ops);
  bindTernaryOps(nvf_ops);
  bindReductionOps(nvf_ops);
  bindCastOps(nvf_ops);
  bindMatmulOps(nvf_ops);
  bindMetadataOps(nvf_ops);
  bindTensorUtilityOps(nvf_ops);
  bindIndexingOps(nvf_ops);
}

} // namespace nvfuser::python
