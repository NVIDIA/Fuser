// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/interface_nodes.h>
#include <polymorphic_value.h>
#include <simplification/egraph_type.h>
#include <type.h>

#include <cstdint>
#include <optional>
#include <unordered_set>
#include <variant>

namespace nvfuser {

namespace egraph {

//! This determines the actual operation of an ENode or ASTNode, e.g.
//! BinaryOpType::Add.
//!
//! Note that the target DataType for CastOp can be inferred by the dtype of
//! this ENode's EClass, but since we need to hash and compare ENodes which do
//! not hold their output DataType (it's held in AnalysisData) we include that
//! DataType here as data as well.
//!
//! LoadStoreOpType is included so that we can easily detect that a node
//! represents a LoadStoreOp, even though we expect to only support
//! LoadStoreOpType::Set since we only support scalar expressions.
using OpType = std::variant<
    std::monostate,
    UnaryOpType,
    BinaryOpType,
    TernaryOpType,
    PrimDataType,
    LoadStoreOpType>;

//! This is a simple function symbol that can represent all of the operations we
//! model in the AST.
struct FunctionSymbol {
  OpType op_type;

 public:
  //! In this context, an undefined term represents a Val without a
  //! definition(). This could be a constant or a free variable like a loop
  //! index or input scalar.
  bool isUndefined() const {
    return std::holds_alternative<std::monostate>(op_type);
  };

  bool isUnaryOp() const {
    return std::holds_alternative<UnaryOpType>(op_type);
  }

  //! Given some inputs, evaluate this operation. This is primarily useful for
  //! constant folding.
  PolymorphicValue evaluate(const std::vector<PolymorphicValue>& inputs) const;
};

//! These objects mimic the Val abstract syntax tree (AST) and can be used to
//! record input Vals and to select the form of simplified values that we will
//! need to construct.
//!
//! Multiple Val*s can represent a single ASTNode.
struct ASTNode : FunctionSymbol {
  //! Compute a coarse estimate of the complexity of computing this value.
  size_t complexity = 0;

  //! Point to producers
  std::vector<ASTNode*> producer_astnodes;

  //! This is a collection of Vals from the Fusion that have this exact form.
  //! This can be used during extraction to select pre-existing Vals with a
  //! desired form.
  Val* representing_val = nullptr;

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
};

//! An ENode is an abstraction of a Val where its producers have been replaced
//! with EClasses. It is like an ASTNode, but whereas an ASTNode holds a Val, an
//! ENode holds a function symbol along with a vector of EClass IDs describing
//! equivalent producers. This lets us represent a combinatorially massive
//! amount of possible Vals with a handful of ENodes.
struct ENode : FunctionSymbol {
  //! EClass IDs of all function arguments.
  std::vector<Id> producer_ids;

  //! When a Val is registered using EGraph::registerVal(v), then v is
  //! associated with an ASTNode. Multiple of these can exist for one ENode,
  //! corresponding to the various members of producer EClasses.
  std::vector<std::unique_ptr<ASTNode>> astnodes;

 public:
  //! Construct and add() an ENode from a Val*. To do this, we construct and add
  //! its producers then get their Ids, recursively. Instead of an ENode*, this
  //! returns an Id to emphasize that this also adds the ENode to the EGraph.
  static Id fromVal(const Val* val);
};

} // namespace egraph

} // namespace nvfuser
