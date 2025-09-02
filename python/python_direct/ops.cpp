// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <ops/all_ops.h>
#include <ops/arith.h>

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

#define NVFUSER_DIRECT_BINDING_BINARY_OP(NAME, OP_NAME, DOCSTRING)       \
  ops.def(                                                               \
      NAME,                                                              \
      [](Val* lhs, Val* rhs) -> Val* {                                   \
        return static_cast<Val* (*)(Val*, Val*)>(OP_NAME)(lhs, rhs);     \
      },                                                                 \
      py::return_value_policy::reference);                               \
  ops.def(                                                               \
      NAME,                                                              \
      [](TensorView* lhs, Val* rhs) -> TensorView* {                     \
        return static_cast<TensorView* (*)(TensorView*, Val*)>(OP_NAME)( \
            lhs, rhs);                                                   \
      },                                                                 \
      py::return_value_policy::reference);                               \
  ops.def(                                                               \
      NAME,                                                              \
      [](Val* lhs, TensorView* rhs) -> TensorView* {                     \
        return static_cast<TensorView* (*)(Val*, TensorView*)>(OP_NAME)( \
            lhs, rhs);                                                   \
      },                                                                 \
      py::return_value_policy::reference);                               \
  ops.def(                                                               \
      NAME,                                                              \
      [](TensorView* lhs, TensorView* rhs) -> TensorView* {              \
        return static_cast<TensorView* (*)(TensorView*, TensorView*)>(   \
            OP_NAME)(lhs, rhs);                                          \
      },                                                                 \
      DOCSTRING,                                                         \
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
            arg, dims, /*keepdim=*/false, dtype);                       \
      },                                                                \
      py::arg("arg"),                                                   \
      py::arg("dtype") = DataType::Null,                                \
      py::return_value_policy::reference);                              \
  ops.def(                                                              \
      NAME,                                                             \
      [](TensorView* arg, int dim, bool keepdim, PrimDataType dtype)    \
          -> TensorView* {                                              \
        return static_cast<TensorView* (*)(TensorView*,                 \
                                           const std::vector<int64_t>&, \
                                           bool,                        \
                                           DataType)>(OP_NAME)(         \
            arg, {dim}, keepdim, dtype);                                \
      },                                                                \
      py::arg("arg"),                                                   \
      py::arg("dim"),                                                   \
      py::arg("keepdim") = false,                                       \
      py::arg("dtype") = DataType::Null,                                \
      py::return_value_policy::reference);                              \
  ops.def(                                                              \
      NAME,                                                             \
      [](TensorView* arg,                                               \
         const std::vector<int64_t>& dims,                              \
         bool keepdim,                                                  \
         PrimDataType dtype) -> TensorView* {                           \
        return static_cast<TensorView* (*)(TensorView*,                 \
                                           const std::vector<int64_t>&, \
                                           bool,                        \
                                           DataType)>(OP_NAME)(         \
            arg, dims, keepdim, dtype);                                 \
      },                                                                \
      py::arg("arg"),                                                   \
      py::arg("dims"),                                                  \
      py::arg("keepdim") = false,                                       \
      py::arg("dtype") = DataType::Null,                                \
      DOCSTRING,                                                        \
      py::return_value_policy::reference);

#define NVFUSER_DIRECT_BINDING_SCAN_OP(NAME, OP_NAME, OP_TYPE, DOCSTRING) \
  ops.def(                                                                \
      NAME,                                                               \
      [](TensorView* arg, int dim, Val* init) -> TensorView* {            \
        BinaryOpType op_type = OP_TYPE;                                   \
        return static_cast<                                               \
            TensorView* (*)(TensorView*, int64_t, BinaryOpType, Val*)>(   \
            OP_NAME)(arg, dim, op_type, init);                            \
      },                                                                  \
      py::arg("arg"),                                                     \
      py::arg("dim"),                                                     \
      py::arg("init").none(true) = py::none(),                            \
      DOCSTRING,                                                          \
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "sub",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "minimum",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "maximum",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "mod",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "eq",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "ge",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "gt",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "le",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "lt",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "ne",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "logical_and",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "logical_or",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "bitwise_and",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "bitwise_or",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "bitwise_xor",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "bitwise_left_shift",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "bitwise_right_shift",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "logical_right_shift",
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
)")
  NVFUSER_DIRECT_BINDING_BINARY_OP(
      "gcd",
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
  // complex does not support (TV, Val) and (Val, TV) argument combinations.
  ops.def("complex", [](Val* lhs, Val* rhs) -> Val* {
    return static_cast<Val* (*)(Val*, Val*)>(complex)(lhs, rhs);
  });
  ops.def(
      "complex",
      [](TensorView* lhs, TensorView* rhs) -> TensorView* {
        return static_cast<TensorView* (*)(TensorView*, TensorView*)>(complex)(
            lhs, rhs);
      },
      R"(
Create a complex number from real and imaginary parts.

Parameters
----------
real : Val or TensorView
imag : Val or TensorView

Returns
-------
Val or TensorView
    A complex number.
)",
      py::return_value_policy::reference);
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
keepdim : bool, optional
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
keepdim : bool, optional
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
keepdim : bool, optional
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
keepdim : bool, optional
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
  ops.def(
      "var",
      [](TensorView* arg,
         const std::vector<int64_t>& dims,
         int64_t correction,
         bool keepdim) -> TensorView* {
        return variance(arg, dims, correction, keepdim);
      },
      py::arg("arg"),
      py::arg("dims"),
      py::arg("correction") = 1,
      py::arg("keepdim") = false,
      R"(
Reduce a tensor by computing the variance along specified dimensions.

Parameters
----------
arg : TensorView
    Input tensor to reduce.
dims : list or tuple
    Dimensions to reduce over.
correction : int, optional
    The correction factor to apply to the variance. Default is 1.
keepdim : bool, optional
    Whether to keep the reduced dimensions with size 1. Default is False.

Returns
-------
TensorView
    A tensor containing the variance along the specified dimensions.
)",
      py::return_value_policy::reference);
  ops.def(
      "var_mean",
      [](TensorView* arg,
         const std::vector<int64_t>& dims,
         int64_t correction,
         bool keepdim) -> std::tuple<TensorView*, TensorView*> {
        VarMeanResult output = variance_mean(arg, dims, correction, keepdim);
        return std::make_tuple(output.var, output.mean);
      },
      py::arg("arg"),
      py::arg("dims"),
      py::arg("correction") = 1,
      py::arg("keepdim") = false,
      R"(
Reduce a tensor by computing the mean and variance along specified dimensions.

Parameters
----------
arg : TensorView
    Input tensor to reduce.
dims : list or tuple
    Dimensions to reduce over.
correction : int, optional
    The correction factor to apply to the variance. Default is 1.
keepdim : bool, optional
    Whether to keep the reduced dimensions with size 1. Default is False.

Returns
-------
tuple
    A tuple containing the variance and mean along the specified dimensions.
)",
      py::return_value_policy::reference);
  ops.def(
      "welford",
      [](TensorView* arg, const std::vector<int64_t>& dims) -> decltype(auto) {
        WelfordResult output = WelfordRaw(arg, dims);
        return std::make_tuple(output.avg, output.var_sum, output.n);
      },
      py::arg("arg"),
      py::arg("dims"),
      R"(
Reduce a tensor by computing the mean and variance along specified dimensions.

Parameters
----------
arg : TensorView
    Input tensor to reduce.
dims : list or tuple
    Dimensions to reduce over.

Returns
-------
tuple
    A tuple containing the mean, variance, and count along the specified dimensions.
)",
      py::return_value_policy::reference);
}

void bindScanOps(py::module_& ops) {
  // cumsum (prefix sum) along a dimension
  NVFUSER_DIRECT_BINDING_SCAN_OP(
      "cumsum",
      scan,
      BinaryOpType::Add,
      R"(
Cumulative sum along a dimension.

Parameters
----------
arg : TensorView
    Input tensor to compute cumulative sum.
dim : int
    Dimension to compute cumulative sum over.

Returns
-------
TensorView
    A new tensor containing the cumulative sum along the specified dimension.
)");

  NVFUSER_DIRECT_BINDING_SCAN_OP(
      "cumprod",
      scan,
      BinaryOpType::Mul,
      R"(
Cumulative product along a dimension.

Parameters
----------
arg : TensorView
    Input tensor to compute cumulative product.
dim : int
    Dimension to compute cumulative product over.

Returns
-------
TensorView
    A new tensor containing the cumulative product along the specified dimension.
)");

  NVFUSER_DIRECT_BINDING_SCAN_OP(
      "cummin",
      scan,
      BinaryOpType::Min,
      R"(
Cumulative minimum along a dimension.

Parameters
----------
arg : TensorView
    Input tensor to compute cumulative minimum.
dim : int
    Dimension to compute cumulative minimum over.

Returns
-------
TensorView
    A new tensor containing the cumulative minimum along the specified dimension.
)");
  NVFUSER_DIRECT_BINDING_SCAN_OP(
      "cummax",
      scan,
      BinaryOpType::Max,
      R"(
Cumulative maximum along a dimension.

Parameters
----------
arg : TensorView
    Input tensor to compute cumulative maximum.
dim : int
    Dimension to compute cumulative maximum over.

Returns
-------
TensorView
    A new tensor containing the cumulative maximum along the specified dimension.
)");
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

void bindCompositeOps(py::module_& ops) {
  ops.def(
      "triu",
      [](TensorView* arg, int64_t diagonal) -> TensorView* {
        Val* diagonal_val =
            IrBuilder::create<nvfuser::Val>(diagonal, DataType::Int);
        return triu(arg, diagonal_val);
      },
      py::arg("arg"),
      py::arg("diagonal") = 0,
      R"(
Get the upper triangular part of a tensor.

Parameters
----------
arg : TensorView
diagonal : int
    Offset of the diagonal relative to the main diagonal.

Returns
-------
TensorView
    The upper triangular part of the tensor.
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
      [](TensorView* arg1, TensorView* arg2, TensorView* bias) -> TensorView* {
        return static_cast<
            TensorView* (*)(TensorView*, TensorView*, TensorView*)>(linear)(
            arg1, arg2, bias);
      },
      py::arg("arg1"),
      py::arg("arg2"),
      py::arg("bias").none(true) = py::none(),
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
  ops.def(
      "grouped_mm",
      [](TensorView* mat1,
         TensorView* mat2,
         TensorView* offsets) -> TensorView* {
        // Calculate output dimensions based on mat1 & mat2 rank
        ScaledTensorView scaled_out = grouped_mm(mat1, mat2, offsets);
        return scaled_out.tv;
      },
      py::arg("mat1"),
      py::arg("mat2"),
      py::arg("offsets"),
      R"(
Grouped matrix multiplication.

Performs matrix multiplication on grouped sets of matrices using offsets
to define variable-sized groups.

Parameters
----------
mat1 : TensorView
    First set of matrices
mat2 : TensorView
    Second set of matrices
offsets : TensorView
    Offsets tensor defining group boundaries

Returns
-------
TensorView
    Result of grouped matrix multiplication
)",
      py::return_value_policy::reference);
  ops.def(
      "grouped_mm",
      [](TensorView* mat1,
         TensorView* mat2,
         TensorView* offsets,
         TensorView* scale1,
         TensorView* scale2,
         TensorView* alpha,
         TensorView* bias,
         TensorView* beta,
         PrimDataType dtype,
         int64_t output_block_scale_size,
         PrimDataType output_block_scale_dtype,
         bool output_gamma)
          -> std::tuple<
              TensorView*,
              std::optional<TensorView*>,
              std::optional<TensorView*>> {
        auto [output, block_scaling_factor, global_scaling_factor] = grouped_mm(
            mat1,
            mat2,
            offsets,
            scale1,
            scale2,
            alpha,
            bias,
            beta,
            dtype,
            output_block_scale_size,
            output_block_scale_dtype,
            output_gamma);

        if (output_gamma) {
          NVF_CHECK(
              output_block_scale_size > 0,
              "output_block_scale_size must be greater than 0 when "
              "output_gamma is true");
          return std::make_tuple(
              output, block_scaling_factor, global_scaling_factor);
        } else if (output_block_scale_size > 0) {
          return std::make_tuple(output, block_scaling_factor, std::nullopt);
        }
        return std::make_tuple(output, std::nullopt, std::nullopt);
      },
      py::arg("mat1"),
      py::arg("mat2"),
      py::arg("offsets"),
      py::arg("scale1"),
      py::arg("scale2"),
      py::arg("alpha").none(true) = py::none(),
      py::arg("bias").none(true) = py::none(),
      py::arg("beta").none(true) = py::none(),
      py::arg("dtype") = DataType::BFloat16,
      py::arg("output_block_scale_size") = 0,
      py::arg("output_block_scale_dtype") = DataType::BFloat16,
      py::arg("output_gamma") = false,
      R"(
Scaled Grouped matrix multiplication.

Performs matrix multiplication on grouped sets of matrices using offsets
to define variable-sized groups.

The math operation is roughly two steps:
    out = alpha * grouped_mm(dequant(mat1, scale1), dequant(mat2, scale2), offsets) + beta * bias

    (out_mat, out_scale, out_gamma) = Quantization(
        out,
        dtype,
        output_block_scale_size,
        output_block_scale_dtype,
        output_gamma)

Note 1: The post quantization only applies when output_block_scale_size > 0,
        which would produce out_scale tensor. Otherwise, None will be returned;
Note 2: When output_gamma is set to True, it should produce global scaling factor out_gamma tensor.
        Otherwise, None will be returned.

Parameters
----------
mat1 : TensorView
    First set of matrices
mat2 : TensorView
    Second set of matrices
offsets : TensorView
    Offsets tensor defining group boundaries
scale1 : TensorView
    Scale tensor for mat1
scale2 : TensorView
    Scale tensor for mat2
alpha : TensorView, optional
    Alpha tensor
bias : TensorView, optional
    Bias tensor
beta : TensorView, optional
    Beta tensor
dtype : PrimDataType, optional
    Output tensor type [default: DataType::BFloat16]
output_block_scale_size : int, optional
    Output block scale size
output_block_scale_dtype : PrimDataType, optional
    Output block scale dtype
output_gamma : bool, optional
    Output gamma [default: False]

Returns
-------
tuple
    A tuple containing the result of matrix multiplication, output block scale tensor, and output gamma tensor.
)",
      py::return_value_policy::reference);
  ops.def(
      "scaled_mm",
      [](TensorView* mat1,
         TensorView* mat2,
         TensorView* scale1,
         TensorView* scale2,
         TensorView* alpha,
         TensorView* bias,
         TensorView* beta,
         PrimDataType dtype,
         int64_t output_block_scale_size,
         PrimDataType output_block_scale_dtype,
         bool output_gamma)
          -> std::tuple<
              TensorView*,
              std::optional<TensorView*>,
              std::optional<TensorView*>> {
        /* Per https://pytorch.org/docs/stable/generated/torch.matmul.html */
        auto [output, block_scaling_factor, global_scaling_factor] = scaled_mm(
            mat1,
            mat2,
            scale1,
            scale2,
            alpha,
            bias,
            beta,
            dtype,
            output_block_scale_size,
            output_block_scale_dtype,
            output_gamma);

        if (output_gamma) {
          NVF_CHECK(
              output_block_scale_size > 0,
              "output_block_scale_size must be greater than 0 when "
              "output_gamma is true");
          return std::make_tuple(
              output, block_scaling_factor, global_scaling_factor);
        } else if (output_block_scale_size > 0) {
          return std::make_tuple(output, block_scaling_factor, std::nullopt);
        }
        return std::make_tuple(output, std::nullopt, std::nullopt);
      },
      py::arg("mat1"),
      py::arg("mat2"),
      py::arg("scale1"),
      py::arg("scale2"),
      py::arg("alpha").none(true) = py::none(),
      py::arg("bias").none(true) = py::none(),
      py::arg("beta").none(true) = py::none(),
      py::arg("dtype") = DataType::BFloat16,
      py::arg("output_block_scale_size") = 0,
      py::arg("output_block_scale_dtype") = DataType::BFloat16,
      py::arg("output_gamma") = false,
      R"(
Scaled matrix multiplication.

Note 1: The post quantization only applies when output_block_scale_size > 0,
        which would produce out_scale tensor. Otherwise, None will be returned;
Note 2: When output_gamma is set to True, it should produce global scaling factor out_gamma tensor.
        Otherwise, None will be returned.

Parameters
----------
mat1 : TensorView
    First set of matrices
mat2 : TensorView
    Second set of matrices
scale1 : TensorView
    Scale tensor for mat1
scale2 : TensorView
    Scale tensor for mat2
alpha : TensorView, optional
    Alpha tensor
bias : TensorView, optional
    Bias tensor
beta : TensorView, optional
    Beta tensor
dtype : PrimDataType, optional
    Output tensor type [default: DataType::BFloat16]
output_block_scale_size : int, optional
    Output block scale size [default: 0]
output_block_scale_dtype : PrimDataType, optional
    Output block scale dtype
output_gamma : bool, optional
    Output gamma [default: False]

Returns
-------
tuple
    A tuple containing the result of matrix multiplication, output block scale tensor, and output gamma tensor.
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
      py::return_value_policy::reference);
  ops.def(
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
    Permutation order.

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
      "slice",
      slice_fn<py::list>,
      py::arg("arg"),
      py::arg("start_indices"),
      py::arg("end_indices"),
      py::arg("strides") = py::none(),
      py::arg("manual_normalization") = false,
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
  ops.def(
      "stride_order",
      [](TensorView* arg, std::vector<int64_t>& stride_order) -> TensorView* {
        TensorView* output = set(arg);
        if (stride_order.empty()) {
          return output;
        }
        size_t ndims =
            TensorDomain::noReductions(arg->getLogicalDomain()).size();
        NVF_CHECK(
            ndims == stride_order.size(),
            "Operator stride_order expects `stride_order` argument to have the "
            "same length as input!");
        std::vector<IterDomain*> allocation_domain =
            ir_utils::strideOrderToAllocation(
                output->getLogicalDomain(), stride_order);
        output->setAllocationDomain(allocation_domain, /*new_contiguity=*/true);
        return output;
      },
      py::arg("arg"),
      py::arg("stride_order"),
      R"(
Create a copy of a tensor with a new memory layout.

Parameters
----------
arg : TensorView
stride_order : list or tuple
    The new order of the dimensions.

Returns
-------
TensorView
    The tensor with a new memory layout.
)",
      py::return_value_policy::reference);
}

template <class ShapeType>
TensorView* pad_fn(TensorView* arg, ShapeType generic_pad_widths, Val* value) {
  std::vector<Val*> pad_widths =
      SequenceAsVector(generic_pad_widths, /*shape_check=*/false);
  NVF_CHECK(
      (int64_t)pad_widths.size() <= 2 * arg->nDims(),
      "Number of pad widths must be at most twice the input dimension");
  return pad(arg, pad_widths, value);
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
)");
  ops.def(
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
index : Val
dim : int
    The dimension to select along.

Returns
-------
TensorView
    The selected tensor.
)");
  ops.def(
      "scatter",
      [](TensorView* arg, TensorView* index, TensorView* src, int64_t dim)
          -> TensorView* {
        NVF_CHECK(
            arg->nDims() == index->nDims() && arg->nDims() == src->nDims(),
            "Tensor arguments have different dimensions ",
            arg->nDims(),
            ", ",
            index->nDims(),
            " and ",
            src->nDims());
        NVF_CHECK(
            dim >= -arg->nDims() && dim < arg->nDims(),
            "Tensor arguments have dimension ",
            arg->nDims(),
            " so dim argument must satisfy ",
            -arg->nDims(),
            " <= dim < ",
            arg->nDims(),
            ", but received ",
            dim);
        return scatter(arg, dim, index, src);
      },
      py::arg("arg"),
      py::arg("index"),
      py::arg("src"),
      py::arg("dim"),
      R"(
Scatter a tensor.

Parameters
----------
arg : TensorView
    The tensor to scatter into.
index : TensorView
    The tensor containing the indices.
src : TensorView
    The source tensor to scatter from.
dim : int
    The dimension to scatter along.

Returns
-------
TensorView
    The scattered tensor.
)",
      py::return_value_policy::reference);
  ops.def(
      "scatter",
      [](TensorView* arg, TensorView* index, Val* src, int64_t dim)
          -> TensorView* {
        NVF_CHECK(
            dim >= -arg->nDims() && dim < arg->nDims(),
            "Tensor arguments have dimension ",
            arg->nDims(),
            " so dim argument must satisfy ",
            -arg->nDims(),
            " <= dim < ",
            arg->nDims(),
            ", but received ",
            dim);
        return scatter(arg, dim, index, src);
      },
      py::arg("arg"),
      py::arg("index"),
      py::arg("src"),
      py::arg("dim"),
      R"(
Scatter a tensor.

Parameters
----------
arg : TensorView
    The tensor to scatter into.
index : TensorView
    The tensor containing the indices.
src : Val
    The source scalar to scatter from.
dim : int
    The dimension to scatter along.

Returns
-------
TensorView
    The scattered tensor.
)",
      py::return_value_policy::reference);
  ops.def(
      "gather",
      [](TensorView* arg, TensorView* index, int64_t dim) -> TensorView* {
        NVF_CHECK(
            arg->nDims() == index->nDims(),
            "Tensor arguments have different dimensions ",
            arg->nDims(),
            " and ",
            index->nDims());
        NVF_CHECK(
            dim >= -arg->nDims() && dim < arg->nDims(),
            "Tensor arguments have dimension ",
            arg->nDims(),
            " so dim argument must satisfy ",
            -arg->nDims(),
            " <= dim < ",
            arg->nDims(),
            ", but received ",
            dim);
        return gather(arg, dim, index);
      },
      py::arg("arg"),
      py::arg("index"),
      py::arg("dim"),
      R"(
Gather values from arg tensor along dim at positions given by index.

The arg and index tensors must have the same number of dimensions. For all axes
other than dim the extent of index in that axis must be less than or equal to
its counterpart in arg.

Parameters
----------
arg : TensorView
    A TensorView of shape `(A_i..., B, A_k...)` where `B` is the extent of `arg`
    in the dimension `dim`.
index : TensorView
    A TensorView of dtype `DataType::Int` of shape `(C_i..., J, C_k...)` where
    `C_k <= A_k` for all extents other than `J`
dim : int
    Which position to index along.

Returns
-------
TensorView
    A TensorView of same dtype as `arg` and of shape `(C_i..., J, C_k...)` where
    the element at position `(i...,j,k...)` is equal to
    `arg[i,...,index[i,...,j,k,...],k,...]`.
)",
      py::return_value_policy::reference);
  ops.def(
      "pad",
      pad_fn<py::list>,
      py::arg("arg"),
      py::arg("pad_widths"),
      py::arg("value").none(true) = py::none(),
      py::return_value_policy::reference);
  ops.def(
      "pad",
      pad_fn<py::tuple>,
      py::arg("arg"),
      py::arg("pad_widths"),
      py::arg("value").none(true) = py::none(),
      R"(
Pad a tensor.

Parameters
----------
arg : TensorView
pad_widths : list or tuple
    The widths of the padding.
value : Val, optional
    The value to pad with. The python default value is None, which is translated
    to zero or False internally.

Returns
-------
TensorView
    The padded tensor.
)",
      py::return_value_policy::reference);
  ops.def(
      "take_along_axis",
      [](TensorView* arg, TensorView* index, int64_t dim) -> TensorView* {
        NVF_CHECK(
            arg->nDims() == index->nDims(),
            "Tensor arguments have different dimensions ",
            arg->nDims(),
            " and ",
            index->nDims());
        auto num_dims = (int64_t)arg->nDims();
        NVF_CHECK(
            dim >= -num_dims && dim < num_dims,
            "Tensor arguments have dimension ",
            num_dims,
            " so dim argument must satisfy ",
            -num_dims,
            " <= dim < ",
            num_dims,
            ", but received ",
            dim);
        return takeAlongAxis(arg, index, dim);
      },
      py::arg("arg"),
      py::arg("index"),
      py::arg("dim"),
      R"(
Index arg in dim at positions given by index.

This operation is very similar to gather, but it enforces that all
dimensions other than dim must be equal between arg and index.

Parameters
----------
arg : TensorView
    Tensor of shape `(Ni...,M,Nk...)` where `M` is the extent of `arg` in the
    dimension `dim`.
index : TensorView
    Tensor of dtype `DataType::Int` of shape `(Ni...,J,Nk...)`.
dim : int
    Which position to index along.

Returns
-------
TensorView
    Tensor of same dtype as `arg` and of shape `(Ni...,J,Nk...)` where the
    element at position `(i...,j,k...)` is equal to
    `arg[i,...,index[i,...,j,k,...],k,...]`.
      )",
      py::return_value_policy::reference);
  ops.def(
      "cat",
      [](std::vector<TensorView*> tensors,
         int64_t dim,
         bool manual_padding) -> TensorView* {
        return cat(
            tensors, dim, /*iter_type_opt=*/std::nullopt, manual_padding);
      },
      py::arg("tensors"),
      py::arg("dim") = 0,
      py::arg("manual_padding") = false,
      py::return_value_policy::reference);
  ops.def(
      "embedding_fwd",
      [](TensorView* input,
         TensorView* weight,
         Val* padding_idx,
         Val* max_norm,
         Val* norm_type,
         Val* scale_grad_by_freq,
         Val* sparse) -> decltype(auto) {
        return embedding_fwd(
            input,
            weight,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse);
      },
      py::arg("input"),
      py::arg("weight"),
      py::arg("padding_idx").none(true) = py::none(),
      py::arg("max_norm").none(true) = py::none(),
      py::arg("norm_type").none(true) = py::none(),
      py::arg("scale_grad_by_freq").none(true) = py::none(),
      py::arg("sparse").none(true) = py::none(),
      R"(
Forward pass for embedding layers that maps integer indices to vectors.

This function performs the forward pass of an embedding layer, which converts
integer indices into dense vector representations by looking up the corresponding
rows in the weight matrix.

Parameters
----------
input : TensorView
    A 1D tensor containing integer indices to be embedded. Each element should
    be a valid index into the weight matrix.
weight : TensorView
    A 2D tensor representing the embedding matrix. Shape should be (num_embeddings, embedding_dim).
padding_idx : Val, optional
    If specified, the embedding vector at this index will be filled with zeros.
    Default is None (no padding).
max_norm : Val, optional
    If specified, each embedding vector will be normalized to have a maximum norm
    of this value. Default is None (no normalization).
norm_type : Val, optional
    The p of the p-norm to use for normalization. Default is 2.0 (L2 norm).
scale_grad_by_freq : Val, optional
    If True, scale gradients by the inverse frequency of the indices in the batch.
    Default is False.
sparse : Val, optional
    If True, only update the gradients for the indices that appear in the batch.
    Default is False.

Returns
-------
TensorView
    A tensor with shape (input_shape + [embedding_dim]) containing the embedded
    vectors corresponding to the input indices.

Notes
-----
- The input tensor must be at least 1D.
- The weight tensor must be exactly 2D.
- All optional parameters must be scalar values when provided.
- This operation is equivalent to PyTorch's torch.nn.functional.embedding.
)",
      py::return_value_policy::reference);
}

template <class ShapeType>
TensorView* full_op_fn(
    ShapeType generic_output_shape,
    Val* fill_value,
    PrimDataType dtype) {
  std::vector<Val*> output_shape = SequenceAsVector(generic_output_shape);
  return full(output_shape, fill_value, dtype);
}

void bindTensorFactoryOps(py::module_& ops) {
  ops.def(
      "iota",
      [](Val* length, Val* start, Val* step, PrimDataType dtype)
          -> TensorView* { return iota(length, start, step, dtype); },
      py::arg("length"),
      py::arg("start").none(true) = py::none(),
      py::arg("step").none(true) = py::none(),
      py::arg("dtype") = DataType::Int,
      R"(
Create a tensor with values from 0 to length-1.

Parameters
----------
length : Val
    The length of the tensor.
start : Val, optional
    The start of the tensor. When the default is None, start is set to zero.
step : Val, optional
    The step of the tensor. When the default is None, step is set to zero.
dtype : PrimDataType, optional
    The data type of the tensor. Default is DataType::Int.

Returns
-------
TensorView
    The tensor with values from 0 to length-1.
)",
      py::return_value_policy::reference);
  ops.def(
      "full",
      full_op_fn<py::list>,
      py::arg("shape"),
      py::arg("fill_value"),
      py::arg("dtype"),
      py::return_value_policy::reference);
  ops.def(
      "full",
      full_op_fn<py::tuple>,
      py::arg("shape"),
      py::arg("fill_value"),
      py::arg("dtype"),
      R"(
Create a tensor with all elements set to a specified value.

Parameters
----------
shape : list or tuple
    The shape of the tensor.
fill_value : Val
    The value to fill the tensor with.
dtype : PrimDataType
    The data type of the tensor.

Returns
-------
TensorView
    The tensor with all elements set to the specified value.
)",
      py::return_value_policy::reference);
}

void bindSearchOps(py::module_& ops) {
  ops.def(
      "topk",
      [](TensorView* arg, Val* k, int64_t dim, bool largest, bool sorted)
          -> py::tuple {
        auto output = topk(arg, k, dim, largest, sorted);
        return py::make_tuple(output.values, output.indices);
      },
      R"(
      Find the k largest or smallest elements along a dimension.

      Args:
          arg (Tensor): Input tensor
          k (Val): Number of elements to return
          dim (int, optional): Dimension along which to find top-k. Defaults to -1.
          largest (bool, optional): If True, return largest elements. Defaults to True.
          sorted (bool, optional): If True, return elements in sorted order. Defaults to False.

      Returns:
          tuple[Tensor, Tensor]: A tuple of (values, indices) where values contains
                                the k largest/smallest elements and indices contains
                                their positions in the original tensor.
      )",
      py::arg("arg"),
      py::arg("k"),
      py::arg("dim") = -1,
      py::arg("largest") = true,
      py::arg("sorted") = false,
      py::return_value_policy::reference);
  ops.def(
      "argsort",
      [](TensorView* arg, int64_t dim, bool descending, bool stable)
          -> TensorView* { return argsort(arg, dim, descending, stable); },
      py::arg("arg"),
      py::arg("dim"),
      py::arg("descending") = false,
      py::arg("stable") = false,
      R"(
Sort the elements of a tensor.

Parameters
----------
arg : TensorView
    The input tensor.
dim : int, optional
    The dimension along which to sort. Defaults to -1.
descending : bool, optional
    If True, sort in descending order. Defaults to False.
stable : bool, optional
    If True, sort in stable order. Defaults to False.

Returns
-------
TensorView
    The sorted tensor.
      )",
      py::return_value_policy::reference);
}

void bindSdpaOps(py::module_& ops) {
  ops.def(
      "sdpfa_fwd",
      [](TensorView* query,
         TensorView* key,
         TensorView* value,
         Val* dropout_p,
         Val* is_causal,
         Val* scale) -> decltype(auto) {
        auto [output, log_sumexp, philox_seed, philox_offset] =
            sdpfa_fwd(query, key, value, dropout_p, is_causal, scale);
        return py::make_tuple(output, log_sumexp, philox_seed, philox_offset);
      },
      py::arg("query"),
      py::arg("key"),
      py::arg("value"),
      py::arg("dropout_p").none(true) = py::none(),
      py::arg("is_causal").none(true) = py::none(),
      py::arg("scale").none(true) = py::none(),
      R"(
Scaled Dot Product Flash Attention Forward.

Parameters
----------
query : TensorView
    The query tensor.
key : TensorView
    The key tensor.
value : TensorView
    The value tensor.
dropout_p : Val, optional
    The dropout probability. Default is None.
is_causal : Val, optional
    Whether the attention is causal. Default is None.
scale : Val, optional
    The scale of the attention. Default is None.

Returns
-------
tuple[TensorView, TensorView, TensorView, TensorView]
    A tuple of (output, log_sumexp, philox_seed, philox_offset).
      )",
      py::return_value_policy::reference);
  ops.def(
      "sdpfa_bwd",
      [](TensorView* grad_output,
         TensorView* query,
         TensorView* key,
         TensorView* value,
         TensorView* output,
         TensorView* log_sumexp,
         Val* dropout_p,
         Val* is_causal,
         TensorView* philox_seed,
         TensorView* philox_offset,
         Val* scale) -> decltype(auto) {
        auto [grad_query, grad_key, grad_value] = sdpfa_bwd(
            grad_output,
            query,
            key,
            value,
            output,
            log_sumexp,
            dropout_p,
            is_causal,
            philox_seed,
            philox_offset,
            scale);
        return std::make_tuple(grad_query, grad_key, grad_value);
      },
      py::arg("grad_output"),
      py::arg("query"),
      py::arg("key"),
      py::arg("value"),
      py::arg("output"),
      py::arg("log_sumexp"),
      py::arg("dropout_p"),
      py::arg("is_causal"),
      py::arg("philox_seed"),
      py::arg("philox_offset"),
      py::arg("scale"),
      R"(
Scaled Dot Product Flash Attention Backward.

Parameters
----------
grad_output : TensorView
    The gradient of the output.
query : TensorView
    The query tensor.
key : TensorView
    The key tensor.
value : TensorView
    The value tensor.
output : TensorView
    The output tensor.
log_sumexp : TensorView
    The log of the sum of the exponential of the key.
dropout_p : Val, optional
    The dropout probability.
is_causal : Val, optional
    Whether the attention is causal.
philox_seed : TensorView
    The seed for the philox random number generator.
philox_offset : TensorView
    The offset for the philox random number generator.
scale : Val, optional
    The scale of the attention.

Returns
-------
tuple[TensorView, TensorView, TensorView]
    A tuple of (grad_query, grad_key, grad_value).
      )",
      py::return_value_policy::reference);
}

template <
    class ShapeType,
    TensorView* (*RandomFuncWithSeed)(
        const std::vector<Val*>&,
        Val*,
        Val*,
        DataType,
        Val*,
        Val*,
        bool)>
TensorView* random_dist_op_fn(
    Val* arg1,
    Val* arg2,
    ShapeType generic_new_shape,
    Val* rng_seed,
    Val* rng_offset,
    PrimDataType dtype) {
  NVF_CHECK(
      !((rng_seed == nullptr) ^ (rng_offset == nullptr)),
      "rng_seed and rng_offset must be provided together!");
  NVF_CHECK(
      isFloatingPointType(dtype),
      "Random distributions only create floating point types! ",
      dtype);
  std::vector<Val*> new_shape = SequenceAsVector(generic_new_shape);
  return RandomFuncWithSeed(
      new_shape,
      arg1,
      arg2,
      dtype,
      rng_seed,
      rng_offset,
      /*maybe_symbolic=*/true);
}

void bindRandomOps(py::module_& ops) {
  ops.def(
      "normal",
      random_dist_op_fn<py::list, normal>,
      py::arg("mean"),
      py::arg("std"),
      py::arg("shape"),
      py::kw_only(),
      py::arg("rng_seed").none(true) = py::none(),
      py::arg("rng_offset").none(true) = py::none(),
      py::arg("dtype") = DataType::Float,
      py::return_value_policy::reference);
  ops.def(
      "normal",
      random_dist_op_fn<py::tuple, normal>,
      py::arg("mean"),
      py::arg("std"),
      py::arg("shape"),
      py::kw_only(),
      py::arg("rng_seed").none(true) = py::none(),
      py::arg("rng_offset").none(true) = py::none(),
      py::arg("dtype") = DataType::Float,
      R"(
Create a tensor with normal distribution.
Parameters
----------
mean : Val
    The mean of the normal distribution.
std : Val
    The standard deviation of the normal distribution.
shape : list or tuple
    The shape of the tensor.
rng_seed : Val, optional
    The seed for the random number generator.
rng_offset : Val, optional
    The offset for the random number generator.
dtype : PrimDataType, optional
    The data type of the tensor.

Returns
-------
TensorView
The tensor with normal distribution.
      )",
      py::return_value_policy::reference);
  ops.def(
      "uniform",
      random_dist_op_fn<py::list, uniform>,
      py::arg("minval"),
      py::arg("maxval"),
      py::arg("shape"),
      py::kw_only(),
      py::arg("rng_seed").none(true) = py::none(),
      py::arg("rng_offset").none(true) = py::none(),
      py::arg("dtype") = DataType::Float,
      py::return_value_policy::reference);
  ops.def(
      "uniform",
      random_dist_op_fn<py::tuple, uniform>,
      py::arg("minval"),
      py::arg("maxval"),
      py::arg("shape"),
      py::kw_only(),
      py::arg("rng_seed").none(true) = py::none(),
      py::arg("rng_offset").none(true) = py::none(),
      py::arg("dtype") = DataType::Float,
      R"(
Create a tensor with uniform distribution.
Parameters
----------
minval : Val
    The minimum value of the uniform distribution.
maxval : Val
    The maximum value of the uniform distribution.
shape : list or tuple
    The shape of the tensor.
rng_seed : Val, optional
    The seed for the random number generator.
rng_offset : Val, optional
    The offset for the random number generator.
dtype : PrimDataType, optional
    The data type of the tensor.

Returns
-------
TensorView
The tensor with normal distribution.
      )",
      py::return_value_policy::reference);
}

} // namespace

void bindOperations(py::module& nvfuser) {
  py::module_ nvf_ops = nvfuser.def_submodule(
      "ops", "This submodule contains all operations for NvFuser.");
  bindUnaryOps(nvf_ops);
  bindBinaryOps(nvf_ops);
  bindTernaryOps(nvf_ops);
  bindReductionOps(nvf_ops);
  bindScanOps(nvf_ops);
  bindCastOps(nvf_ops);
  bindCompositeOps(nvf_ops);
  bindMatmulOps(nvf_ops);
  bindMetadataOps(nvf_ops);
  bindTensorUtilityOps(nvf_ops);
  bindIndexingOps(nvf_ops);
  bindTensorFactoryOps(nvf_ops);
  bindSearchOps(nvf_ops);
  bindSdpaOps(nvf_ops);
  bindRandomOps(nvf_ops);
}

} // namespace nvfuser::python
