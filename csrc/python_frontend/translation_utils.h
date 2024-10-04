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

// Get string name for BinaryOp
std::string getString(const BinaryOp* bop);

} // namespace nvfuser::python_frontend
