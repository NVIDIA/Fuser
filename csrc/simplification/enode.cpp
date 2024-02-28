// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/internal_nodes.h>
#include <ir/interface_nodes.h>
#include <simplification/egraph_type.h>
#include <simplification/enode.h>
#include <type.h>

#include <optional>
#include <unordered_set>
#include <variant>

namespace nvfuser {

namespace egraph {

FunctionType FunctionType::fromVal(Val* val) {
  auto symbol = FunctionSymbol::NoDefinition;
  FunctionType::OpType op_type;
  if (Expr* def = val->definition()) {
    if (auto bop = dynamic_cast<::nvfuser::BinaryOp*>(def)) {
      op_type = bop->getBinaryOpType();
    } else {
      NVF_ERROR(false, "Val ", val->toString(), " has an unsupported Expr type");
    }
  }
  return {symbol, op_type};
}

PolymorphicValue FunctionType::evaluate(
    const std::vector<PolymorphicValue>& inputs) const {
  switch (symbol) {
    case FunctionSymbol::NoDefinition:
      NVF_ERROR(
          false,
          "Cannot evaluate AST function that does not have a definition");
    case FunctionSymbol::LoadStoreOp:
      NVF_ERROR(inputs.size() == 1);
      return inputs[0];
      break;
      // TODO: Refactor ops so that we can call static versions of each op and
      // avoid creating new Exprs in the Fusion here.
    default:
      NVF_ERROR(false, "not yet implemented");
      return std::monostate{};
  }
}

ASTNode ASTNode::fromVal(Val* val) {
  return {.definition = FunctionType::fromVal(val), .representing_vals = {val}};
}

} // namespace egraph

} // namespace nvfuser

