// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/interface_nodes.h>
#include <ir/internal_nodes.h>
#include <simplification/eclass.h>
#include <simplification/egraph_type.h>
#include <simplification/enode.h>
#include <type.h>

#include <optional>
#include <unordered_set>
#include <variant>

namespace nvfuser {

namespace egraph {

FunctionDesc FunctionDesc::fromVal(Val* val) {
  auto symbol = FunctionSymbol::NoDefinition;
  FunctionDesc::OpType op_type;
  if (Expr* def = val->definition()) {
    if (auto bop = dynamic_cast<::nvfuser::BinaryOp*>(def)) {
      op_type = bop->getBinaryOpType();
    } else {
      NVF_ERROR(
          false, "Val ", val->toString(), " has an unsupported Expr type");
    }
  }
  return {symbol, op_type};
}

PolymorphicValue FunctionDesc::evaluate(
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

// Since ASTNodes are owned by their ENodes, we need to
static

    Id
    ENode::fromVal(Val* val) {
  NVF_ERROR(val->isScalar(), "EGraph currently only models scalars");
  EGraph* eg = EGraphGuard::getCurEGuard();
  FunctionDesc definition;
  std::vector<Id> producer_ids;
  if (Expr* def = val->definition()) {
    // Recursive case
    if (auto lsop = dynamic_cast<UnaryOp*>(def)) {
      // Note: we currently ignore cacheOp here since we intend this only for
      // use on scalars.
      definition.symbol = FunctionSymbol::LoadStoreOp;
    } else if (auto cop = dynamic_cast<CastOp*>(def)) {
      definition.symbol = FunctionSymbol::CastOp;
      definition.op_type = val->dtype();
    } else if (auto uop = dynamic_cast<UnaryOp*>(def)) {
      definition.symbol = FunctionSymbol::UnaryOp;
      definition.op_type = uop->getUnaryOpType();
    } else {
      NVF_ERROR(false, "Unsupported definition: ", def->toString());
    }

    // Recurse into producers and get their Ids
    producer_ids.reserve(def->inputs().size());
    for (Val* inp : def->inputs()) {
      producer_ids.push_back(fromVal(inp));
    }
  } else {
    // Variable without definition. This could be an input scalar or a loop
    // index.
    definition.symbol = FunctionSymbol::NoDefinition;
  }

  // The immediate producer ENodes have representing Vals, which are the
  // def->inputs(). So there will now be ASTNodes attached to those ENodes if we
  // look up the Ids in producer_ids. We will gather those and combine them to
  // form this ASTNode.
  std::vector<ASTNode*> producer_astnodes;
  producer_astnodes.reserve(producer_ids.size());
  for (Id producer_id : producer_ids) {
    ENode* producer_enode = eg->getENodeFromId(producer_id);
  }

  std::unique_ptr<ENode> enode_ptr{definition, producer_ids, {astnode}};

  // add and get Id
  Id id = eg->add(std::move(enode_ptr));

  if (val->isConst()) {
    // Immediate constant
    // TODO: check that this constant does not clash with possibly pre-existing
    // constant. This could happen if the ENode already exists and has a
    // constant folded.
    EClass* eclass = eg->getEClassFromId(id);
    NVF_ERROR();
  }
}

} // namespace egraph

} // namespace nvfuser
