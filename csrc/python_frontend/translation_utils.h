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
  auto get_std_function = [](ResultType (*fn)(ArgTypes...)) {
    return static_cast<ResultType (*)(ArgTypes...)>(fn);
  };

  switch (uop->getUnaryOpType()) {
    case UnaryOpType::Abs:
      return get_std_function(abs);
    case UnaryOpType::Acos:
      return get_std_function(acos);
    case UnaryOpType::Acosh:
      return get_std_function(acosh);
    case UnaryOpType::Asin:
      return get_std_function(asin);
    case UnaryOpType::Asinh:
      return get_std_function(asinh);
    case UnaryOpType::Atan:
      return get_std_function(atan);
    case UnaryOpType::Atanh:
      return get_std_function(atanh);
    case UnaryOpType::Ceil:
      return get_std_function(ceil);
    case UnaryOpType::Cos:
      return get_std_function(cos);
    case UnaryOpType::Cosh:
      return get_std_function(cosh);
    case UnaryOpType::Exp:
      return get_std_function(exp);
    case UnaryOpType::Exp2:
      return get_std_function(exp2);
    case UnaryOpType::Expm1:
      return get_std_function(expm1);
    case UnaryOpType::Erf:
      return get_std_function(erf);
    case UnaryOpType::Erfc:
      return get_std_function(erfc);
    case UnaryOpType::Erfinv:
      return get_std_function(erfinv);
    case UnaryOpType::Erfcinv:
      return get_std_function(erfcinv);
    case UnaryOpType::Floor:
      return get_std_function(floor);
    case UnaryOpType::Frac:
      return get_std_function(frac);
    case UnaryOpType::Lgamma:
      return get_std_function(lgamma);
    case UnaryOpType::Log:
      return get_std_function(log);
    case UnaryOpType::Log10:
      return get_std_function(log10);
    case UnaryOpType::Log1p:
      return get_std_function(log1p);
    case UnaryOpType::Log2:
      return get_std_function(log2);
    case UnaryOpType::Neg:
      return get_std_function(neg);
    case UnaryOpType::LogicalNot:
      return get_std_function(logical_not);
    case UnaryOpType::BitwiseNot:
      return get_std_function(bitwise_not);
    case UnaryOpType::Reciprocal:
      return get_std_function(reciprocal);
    case UnaryOpType::Relu:
      return get_std_function(relu);
    case UnaryOpType::Rsqrt:
      return get_std_function(rsqrt);
    case UnaryOpType::Round:
      return get_std_function(round);
    case UnaryOpType::Sigmoid:
      return get_std_function(sigmoid);
    case UnaryOpType::Signbit:
      return get_std_function(signbit);
    case UnaryOpType::Silu:
      return get_std_function(silu);
    case UnaryOpType::Sin:
      return get_std_function(sin);
    case UnaryOpType::Sinh:
      return get_std_function(sinh);
    case UnaryOpType::Sqrt:
      return get_std_function(sqrt);
    case UnaryOpType::Tan:
      return get_std_function(tan);
    case UnaryOpType::Tanh:
      return get_std_function(tanh);
    case UnaryOpType::Trunc:
      return get_std_function(trunc);
    case UnaryOpType::IsFinite:
      return get_std_function(isfinite);
    case UnaryOpType::IsInf:
      return get_std_function(isinf);
    case UnaryOpType::IsNan:
      return get_std_function(isnan);
    case UnaryOpType::IsNegInf:
      return get_std_function(isneginf);
    case UnaryOpType::IsPosInf:
      return get_std_function(isposinf);
    case UnaryOpType::IsReal:
      return get_std_function(isreal);
    case UnaryOpType::Real:
      return get_std_function(real);
    case UnaryOpType::Imag:
      return get_std_function(imag);
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
  auto get_std_function = [](ResultType (*fn)(ArgTypes...)) {
    return static_cast<ResultType (*)(ArgTypes...)>(fn);
  };

  // clamp and threshold define a subset of TernaryOp configurations, so they
  // are handled in a separate template specialization.
  switch (top->getTernaryOpType()) {
    case TernaryOpType::Lerp:
      return get_std_function(lerp);
      break;
    case TernaryOpType::Where:
      return get_std_function(where);
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

// Get string name for UnaryOp
std::string getString(const UnaryOp* uop);

// Get string name for BinaryOp
std::string getString(const BinaryOp* bop);

// Get string name for TernaryOp
std::string getString(const TernaryOp* bop);

} // namespace nvfuser::python_frontend
