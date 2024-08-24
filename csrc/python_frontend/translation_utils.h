// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <ir/all_nodes.h>

namespace nvfuser::python_frontend {

// Get std::function for BinaryOp
template <typename ResultType, typename... ArgTypes>
std::function<ResultType(ArgTypes...)> getFunction(const BinaryOp* bop) {
  auto get_std_function = [](ResultType (*fn)(ArgTypes...)) {
    return static_cast<ResultType (*)(ArgTypes...)>(fn);
  };

  switch (bop->getBinaryOpType()) {
    case BinaryOpType::Add:
      return get_std_function(add);
      break;
    case BinaryOpType::Atan2:
      return get_std_function(atan2);
      break;
    case BinaryOpType::Div:
      return get_std_function(div);
      break;
    case BinaryOpType::Truediv:
      return get_std_function(truediv);
      break;
    case BinaryOpType::Fmod:
      return get_std_function(fmod);
      break;
    case BinaryOpType::Mul:
      return get_std_function(mul);
      break;
    case BinaryOpType::Nextafter:
      return get_std_function(nextafter);
      break;
    case BinaryOpType::Pow:
      return get_std_function(pow);
      break;
    case BinaryOpType::Remainder:
      return get_std_function(remainder);
      break;
    case BinaryOpType::Sub:
      return get_std_function(sub);
      break;
    case BinaryOpType::Mod:
      return get_std_function(mod);
      break;
    case BinaryOpType::Eq:
      return get_std_function(eq);
      break;
    case BinaryOpType::NE:
      return get_std_function(ne);
      break;
    case BinaryOpType::GT:
      return get_std_function(gt);
      break;
    case BinaryOpType::GE:
      return get_std_function(ge);
      break;
    case BinaryOpType::LT:
      return get_std_function(lt);
      break;
    case BinaryOpType::LE:
      return get_std_function(le);
      break;
    case BinaryOpType::BitwiseAnd:
      return get_std_function(bitwise_and);
      break;
    case BinaryOpType::BitwiseOr:
      return get_std_function(bitwise_or);
      break;
    case BinaryOpType::BitwiseXor:
      return get_std_function(bitwise_xor);
      break;
    case BinaryOpType::LogicalAnd:
      return get_std_function(logical_and);
      break;
    case BinaryOpType::LogicalOr:
      return get_std_function(logical_or);
      break;
    case BinaryOpType::Lshift:
      return get_std_function(bitwise_left_shift);
      break;
    case BinaryOpType::Rshift:
      return get_std_function(bitwise_right_shift);
      break;
    case BinaryOpType::Gcd:
      return get_std_function(gcd);
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
