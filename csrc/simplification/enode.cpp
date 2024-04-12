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
#include <simplification/egraph.h>
#include <simplification/egraph_type.h>
#include <simplification/enode.h>
#include <type.h>

#include <optional>
#include <unordered_set>
#include <variant>

namespace nvfuser {

namespace egraph {

// Given a Val return an ENode Id representing it as well as well as a pointer
// to the ASTNode that represents it. This helper is necessary because although
// ENode owns the ASTNode inside its `astnodes` attribute, if we only returned
// the ENode Id then we would not be able to tell which representing ASTNode
// corresponds to this Val.
static std::pair<Id, ASTNode*> fromValHelper(EGraph* eg, const Val* val) {
  std::vector<Id> producer_ids;
  std::vector<ASTNode*> producer_astnodes;
  // OpType defaults to std::monostate which signifies a undefined (root) Val
  OpType op_type;
  if (const Expr* def = val->definition()) {
    // Recursive case
    if (const auto lsop = dynamic_cast<const LoadStoreOp*>(def)) {
      op_type = lsop->opType();
    } else if (const auto uop = dynamic_cast<const UnaryOp*>(def)) {
      op_type = uop->getUnaryOpType();
      if (op_type == UnaryOpType::Cast) {
        // Record the dtype in place of the op type
        op_type = val->dtype();
      }
    } else if (const auto bop = dynamic_cast<const BinaryOp*>(def)) {
      op_type = bop->getBinaryOpType();
    } else if (const auto top = dynamic_cast<const TernaryOp*>(def)) {
      op_type = top->getTernaryOpType();
    } else {
      NVF_ERROR(false, "Unsupported definition: ", def->toString());
    }

    // Recurse into producers and get their Ids
    producer_ids.reserve(def->inputs().size());
    producer_astnodes.reserve(def->inputs().size());
    for (Val* inp : def->inputs()) {
      auto& [producer_id, producer_astnode] = fromValHelper(eg, inp);
      producer_ids.push_back(producer_id);
      producer_astnodes.push_back(producer_astnode);
    }
  }

  std::unique_ptr<ENode> enode_ptr{op_type, producer_ids, {nullptr}};
  enode_ptr->astnodes.back() = std::make_unique<ASTNode>(
      /*is_unrolled_loop_index=*/false,
      /*complexity=*/0,
      /*producer_astnodes=*/producer_astnodes,
      /*representing_vals=*/{val});
  ASTNode* astnode = enode_ptr->astnodes.back().get();

  // add and get Id
  const Id id = eg->add(std::move(enode_ptr));

  if (val->isConst()) {
    // Immediate constant
    // TODO: check that this constant does not clash with possibly pre-existing
    // constant. This could happen if the ENode already exists and has a
    // constant folded.
    EClass* eclass = eg->getEClassFromId(id);
    if (eclass->data.constant.hasValue()) {
      NVF_ERROR(
          eclass->data.constant == val->value(),
          "Found conflicting values ",
          eclass->data.constant,
          " and ",
          val->value(),
          " for EClass representing ",
          val->toInlineString());
    } else {
      eclass->data.constant = val->value();
    }
  }

  // [Modeling term visibility in ASTNode]
  // TODO: we can represent visibility of each representing Val here
  // For example, 8 bytes can represent the position of an ordered tree of
  // scopes with height and node width at most 8.
  //
  // Visibility could be used to filter out representing vals recursively,
  // enabling us to extract a representation that is visible at a particular
  // scope. We could potentially also alter the analysis method to encourage
  // higher visibility of intermediate nodes, so that we would automatically
  // create easily-hoisted ASTNodes.

  return {id, astnode};
}

Id ENode::fromVal(const Val* val) {
  NVF_ERROR(val->isScalar(), "EGraph currently only models scalars");
  EGraph* eg = EGraphGuard::getCurEGraph();
  auto& [/*Id*/ id, /*ASTNode*/ astnode] = fromValHelper(eg, val);
  return id;
}

} // namespace egraph

} // namespace nvfuser
