
// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
//
#include <python_frontend/translation_utils.h>

namespace nvfuser::python_frontend {

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

} // namespace nvfuser::python_frontend
