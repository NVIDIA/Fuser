// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <ir/all_nodes.h>
#include <ops/all_ops.h>

namespace nvfuser::python_frontend {

// Get std::function for UnaryOp
template <typename ResultType, typename... ArgTypes>
std::function<ResultType(ArgTypes...)> getFunction(const UnaryOp* uop) {
  auto wrap_function = [](ResultType (*fn)(ArgTypes...)) { return fn; };

  switch (uop->getUnaryOpType()) {
    case UnaryOpType::Abs:
      return wrap_function(abs);
    case UnaryOpType::Acos:
      return wrap_function(acos);
    case UnaryOpType::Acosh:
      return wrap_function(acosh);
    case UnaryOpType::Asin:
      return wrap_function(asin);
    case UnaryOpType::Asinh:
      return wrap_function(asinh);
    case UnaryOpType::Atan:
      return wrap_function(atan);
    case UnaryOpType::Atanh:
      return wrap_function(atanh);
    case UnaryOpType::Ceil:
      return wrap_function(ceil);
    case UnaryOpType::Cos:
      return wrap_function(cos);
    case UnaryOpType::Cosh:
      return wrap_function(cosh);
    case UnaryOpType::Exp:
      return wrap_function(exp);
    case UnaryOpType::Exp2:
      return wrap_function(exp2);
    case UnaryOpType::Expm1:
      return wrap_function(expm1);
    case UnaryOpType::Erf:
      return wrap_function(erf);
    case UnaryOpType::Erfc:
      return wrap_function(erfc);
    case UnaryOpType::Erfinv:
      return wrap_function(erfinv);
    case UnaryOpType::Erfcinv:
      return wrap_function(erfcinv);
    case UnaryOpType::Floor:
      return wrap_function(floor);
    case UnaryOpType::Frac:
      return wrap_function(frac);
    case UnaryOpType::Lgamma:
      return wrap_function(lgamma);
    case UnaryOpType::Log:
      return wrap_function(log);
    case UnaryOpType::Log10:
      return wrap_function(log10);
    case UnaryOpType::Log1p:
      return wrap_function(log1p);
    case UnaryOpType::Log2:
      return wrap_function(log2);
    case UnaryOpType::Neg:
      return wrap_function(neg);
    case UnaryOpType::LogicalNot:
      return wrap_function(logical_not);
    case UnaryOpType::BitwiseNot:
      return wrap_function(bitwise_not);
    case UnaryOpType::Reciprocal:
      return wrap_function(reciprocal);
    case UnaryOpType::Relu:
      return wrap_function(relu);
    case UnaryOpType::Rsqrt:
      return wrap_function(rsqrt);
    case UnaryOpType::Round:
      return wrap_function(round);
    case UnaryOpType::Sigmoid:
      return wrap_function(sigmoid);
    case UnaryOpType::Signbit:
      return wrap_function(signbit);
    case UnaryOpType::Silu:
      return wrap_function(silu);
    case UnaryOpType::Sin:
      return wrap_function(sin);
    case UnaryOpType::Sinh:
      return wrap_function(sinh);
    case UnaryOpType::Sqrt:
      return wrap_function(sqrt);
    case UnaryOpType::Tan:
      return wrap_function(tan);
    case UnaryOpType::Tanh:
      return wrap_function(tanh);
    case UnaryOpType::Trunc:
      return wrap_function(trunc);
    case UnaryOpType::IsFinite:
      return wrap_function(isfinite);
    case UnaryOpType::IsInf:
      return wrap_function(isinf);
    case UnaryOpType::IsNan:
      return wrap_function(isnan);
    case UnaryOpType::IsNegInf:
      return wrap_function(isneginf);
    case UnaryOpType::IsPosInf:
      return wrap_function(isposinf);
    case UnaryOpType::IsReal:
      return wrap_function(isreal);
    case UnaryOpType::Real:
      return wrap_function(real);
    case UnaryOpType::Imag:
      return wrap_function(imag);
    default:
      NVF_CHECK(
          false,
          "Unexpected operator type: ",
          uop->getUnaryOpType(),
          " in ",
          uop->toString());
  }
}

// Get std::function for BinaryOp
template <typename ResultType, typename... ArgTypes>
std::function<ResultType(ArgTypes...)> getFunction(const BinaryOp* bop) {
  auto wrap_function = [](ResultType (*fn)(ArgTypes...)) { return fn; };

  switch (bop->getBinaryOpType()) {
    case BinaryOpType::Add:
      return wrap_function(add);
      break;
    case BinaryOpType::Atan2:
      return wrap_function(atan2);
      break;
    case BinaryOpType::Div:
      return wrap_function(div);
      break;
    case BinaryOpType::Fmod:
      return wrap_function(fmod);
      break;
    case BinaryOpType::Mul:
      return wrap_function(mul);
      break;
    case BinaryOpType::Nextafter:
      return wrap_function(nextafter);
      break;
    case BinaryOpType::Pow:
      return wrap_function(pow);
      break;
    case BinaryOpType::Remainder:
      return wrap_function(remainder);
      break;
    case BinaryOpType::Sub:
      return wrap_function(sub);
      break;
    case BinaryOpType::Mod:
      return wrap_function(mod);
      break;
    case BinaryOpType::Eq:
      return wrap_function(eq);
      break;
    case BinaryOpType::NE:
      return wrap_function(ne);
      break;
    case BinaryOpType::GT:
      return wrap_function(gt);
      break;
    case BinaryOpType::GE:
      return wrap_function(ge);
      break;
    case BinaryOpType::LT:
      return wrap_function(lt);
      break;
    case BinaryOpType::LE:
      return wrap_function(le);
      break;
    case BinaryOpType::BitwiseAnd:
      return wrap_function(bitwise_and);
      break;
    case BinaryOpType::BitwiseOr:
      return wrap_function(bitwise_or);
      break;
    case BinaryOpType::BitwiseXor:
      return wrap_function(bitwise_xor);
      break;
    case BinaryOpType::LogicalAnd:
      return wrap_function(logical_and);
      break;
    case BinaryOpType::LogicalOr:
      return wrap_function(logical_or);
      break;
    case BinaryOpType::Lshift:
      return wrap_function(bitwise_left_shift);
      break;
    case BinaryOpType::Rshift:
      return wrap_function(bitwise_right_shift);
      break;
    case BinaryOpType::Gcd:
      return wrap_function(gcd);
      break;
    case BinaryOpType::Min:
      return wrap_function(minimum);
      break;
    case BinaryOpType::Max:
      return wrap_function(maximum);
      break;
    case BinaryOpType::CeilDiv:
      return wrap_function(ceilDiv);
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

// Get std::function for TernaryOp
template <typename ResultType, typename... ArgTypes>
std::function<ResultType(ArgTypes...)> getFunction(const TernaryOp* top) {
  auto wrap_function = [](ResultType (*fn)(ArgTypes...)) { return fn; };

  // clamp and threshold define a subset of TernaryOp configurations, so they
  // are handled in a separate template specialization.
  switch (top->getTernaryOpType()) {
    case TernaryOpType::Lerp:
      return wrap_function(lerp);
      break;
    case TernaryOpType::Where:
      return wrap_function(where);
      break;
    case TernaryOpType::Threshold:
    case TernaryOpType::Clamp:
      NVF_CHECK(
          false,
          "Invalid function arguments for operator type",
          top->getTernaryOpType(),
          " in ",
          top->toString());
    default:
      NVF_CHECK(
          false,
          "Unexpected operator type: ",
          top->getTernaryOpType(),
          " in ",
          top->toString());
  }
}

// Fully specialized template functions to create std::function for TernaryOp.
template <>
std::function<TensorView*(TensorView*, Val*, Val*)> getFunction<
    TensorView*,
    TensorView*,
    Val*,
    Val*>(const TernaryOp* top);

template <>
std::function<Val*(Val*, Val*, Val*)> getFunction<Val*, Val*, Val*, Val*>(
    const TernaryOp* top);

// Get std::function for ReductionOp
template <typename ResultType, typename... ArgTypes>
std::function<ResultType(ArgTypes...)> getFunction(const ReductionOp* rop) {
  switch (rop->getReductionOpType()) {
    case BinaryOpType::Add:
      return sum;
      break;
    case BinaryOpType::Mul:
      return prod;
      break;
    case BinaryOpType::Max:
      return max;
      break;
    case BinaryOpType::Min:
      return min;
      break;
    default:
      NVF_CHECK(
          false,
          "Unexpected reduction operator type: ",
          rop->getReductionOpType(),
          " in ",
          rop->toString());
  }
}

// Get serde record type for ReductionOp
serde::RecordType getSerdeType(const ReductionOp* rop);

} // namespace nvfuser::python_frontend
