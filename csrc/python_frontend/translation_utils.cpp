
// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
//
#include <ops/all_ops.h>
#include <python_frontend/translation_utils.h>

namespace nvfuser::python_frontend {

std::string getString(const UnaryOp* uop) {
  switch (uop->getUnaryOpType()) {
    case UnaryOpType::Abs:
      return "abs";
    case UnaryOpType::Acos:
      return "acos";
    case UnaryOpType::Acosh:
      return "acosh";
    case UnaryOpType::Asin:
      return "asin";
    case UnaryOpType::Asinh:
      return "asinh";
    case UnaryOpType::Atan:
      return "atan";
    case UnaryOpType::Atanh:
      return "atanh";
    case UnaryOpType::Ceil:
      return "ceil";
    case UnaryOpType::Cos:
      return "cos";
    case UnaryOpType::Cosh:
      return "cosh";
    case UnaryOpType::Exp:
      return "exp";
    case UnaryOpType::Exp2:
      return "exp2";
    case UnaryOpType::Expm1:
      return "expm1";
    case UnaryOpType::Erf:
      return "erf";
    case UnaryOpType::Erfc:
      return "erfc";
    case UnaryOpType::Erfinv:
      return "erfinv";
    case UnaryOpType::Erfcinv:
      return "erfcinv";
    case UnaryOpType::Floor:
      return "floor";
    case UnaryOpType::Frac:
      return "frac";
    case UnaryOpType::Lgamma:
      return "lgamma";
    case UnaryOpType::Log:
      return "log";
    case UnaryOpType::Log10:
      return "log10";
    case UnaryOpType::Log1p:
      return "log1p";
    case UnaryOpType::Log2:
      return "log2";
    case UnaryOpType::Neg:
      return "neg";
    case UnaryOpType::LogicalNot:
      return "logical_not";
    case UnaryOpType::BitwiseNot:
      return "bitwise_not";
    case UnaryOpType::Reciprocal:
      return "reciprocal";
    case UnaryOpType::Relu:
      return "relu";
    case UnaryOpType::Rsqrt:
      return "rsqrt";
    case UnaryOpType::Round:
      return "round";
    case UnaryOpType::Sigmoid:
      return "sigmoid";
    case UnaryOpType::Signbit:
      return "signbit";
    case UnaryOpType::Silu:
      return "silu";
    case UnaryOpType::Sin:
      return "sin";
    case UnaryOpType::Sinh:
      return "sinh";
    case UnaryOpType::Sqrt:
      return "sqrt";
    case UnaryOpType::Tan:
      return "tan";
    case UnaryOpType::Tanh:
      return "tanh";
    case UnaryOpType::Trunc:
      return "trunc";
    case UnaryOpType::IsFinite:
      return "isfinite";
    case UnaryOpType::IsInf:
      return "isinf";
    case UnaryOpType::IsNan:
      return "isnan";
    case UnaryOpType::IsNegInf:
      return "isneginf";
    case UnaryOpType::IsPosInf:
      return "isposinf";
    case UnaryOpType::IsReal:
      return "isreal";
    case UnaryOpType::Real:
      return "real";
    case UnaryOpType::Imag:
      return "imag";
    default:
      NVF_CHECK(
          false,
          "Unexpected operator type: ",
          uop->getUnaryOpType(),
          " in ",
          uop->toString());
  }
}

std::string getString(const BinaryOp* bop) {
  switch (bop->getBinaryOpType()) {
    case BinaryOpType::Add:
      return "add";
      break;
    case BinaryOpType::Atan2:
      return "atan2";
      break;
    case BinaryOpType::Div:
      return "div";
      break;
    case BinaryOpType::Fmod:
      return "fmod";
      break;
    case BinaryOpType::Mul:
      return "mul";
      break;
    case BinaryOpType::Nextafter:
      return "nextafter";
      break;
    case BinaryOpType::Pow:
      return "pow";
      break;
    case BinaryOpType::Remainder:
      return "remainder";
      break;
    case BinaryOpType::Sub:
      return "sub";
      break;
    case BinaryOpType::Mod:
      return "mod";
      break;
    case BinaryOpType::Eq:
      return "eq";
      break;
    case BinaryOpType::NE:
      return "ne";
      break;
    case BinaryOpType::GT:
      return "gt";
      break;
    case BinaryOpType::GE:
      return "ge";
      break;
    case BinaryOpType::LT:
      return "lt";
      break;
    case BinaryOpType::LE:
      return "le";
      break;
    case BinaryOpType::BitwiseAnd:
      return "bitwise_and";
      break;
    case BinaryOpType::BitwiseOr:
      return "bitwise_or";
      break;
    case BinaryOpType::BitwiseXor:
      return "bitwise_xor";
      break;
    case BinaryOpType::LogicalAnd:
      return "logical_and";
      break;
    case BinaryOpType::LogicalOr:
      return "logical_or";
      break;
    case BinaryOpType::Lshift:
      return "bitwise_left_shift";
      break;
    case BinaryOpType::Rshift:
      return "bitwise_right_shift";
      break;
    case BinaryOpType::Gcd:
      return "gcd";
      break;
    case BinaryOpType::Max:
      return "maximum";
      break;
    case BinaryOpType::Min:
      return "minimum";
      break;
    case BinaryOpType::CeilDiv:
      return "ceilDiv";
      break;
    default:
      NVF_CHECK(
          false,
          "Unexpected operator type: ",
          bop->getBinaryOpType(),
          " in ",
          bop->toString());
  }
}

std::string getString(const TernaryOp* top) {
  switch (top->getTernaryOpType()) {
    case TernaryOpType::Clamp:
      return "clamp";
      break;
    case TernaryOpType::Lerp:
      return "lerp";
      break;
    case TernaryOpType::Threshold:
      return "threshold";
      break;
    case TernaryOpType::Where:
      return "where";
      break;
    default:
      NVF_CHECK(
          false,
          "Unexpected operator type: ",
          top->getTernaryOpType(),
          " in ",
          top->toString());
  }
}

#define GET_FUNCTION_TERNARY_SPECIALIZATION(                                 \
    ResultType, InType1, InType2, InType3)                                   \
  template <>                                                                \
  std::function<ResultType(InType1, InType2, InType3)>                       \
  getFunction<ResultType, InType1, InType2, InType3>(const TernaryOp* top) { \
    auto wrap_function = [](ResultType (*fn)(InType1, InType2, InType3)) {   \
      return fn;                                                             \
    };                                                                       \
                                                                             \
    switch (top->getTernaryOpType()) {                                       \
      case TernaryOpType::Clamp:                                             \
        return wrap_function(clamp);                                         \
        break;                                                               \
      case TernaryOpType::Lerp:                                              \
        return wrap_function(lerp);                                          \
        break;                                                               \
      case TernaryOpType::Threshold:                                         \
        return wrap_function(threshold);                                     \
        break;                                                               \
      case TernaryOpType::Where:                                             \
        return wrap_function(where);                                         \
        break;                                                               \
      default:                                                               \
        NVF_CHECK(                                                           \
            false,                                                           \
            "Unexpected operator type: ",                                    \
            top->getTernaryOpType(),                                         \
            " in ",                                                          \
            top->toString());                                                \
    }                                                                        \
  }

// Template specializations for std::function for TernaryOp
GET_FUNCTION_TERNARY_SPECIALIZATION(TensorView*, TensorView*, Val*, Val*)
GET_FUNCTION_TERNARY_SPECIALIZATION(Val*, Val*, Val*, Val*)

} // namespace nvfuser::python_frontend
