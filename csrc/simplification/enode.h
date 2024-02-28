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

//! An ENode represents either a constant, a definition-less scalar (such as a
//! loop variable or input scalar), or a scalar defined by some function.

//! This struct describes a function without describing any of its arguments (or
//! even how many arguments there are).
struct FunctionType {
  //! What type of node is this
  FunctionSymbol symbol = FunctionSymbol::NoDefinition;

  //! This determines the actual operation, e.g. BinaryOpType::Add
  //! Note that the target DataType for CastOp can be inferred by the dtype of
  //! this ENode's EClass, but since we need to hash and compare ENodes we
  //! include that DataType here as data as well.
  using OpType = std::variant<
      std::monostate,
      UnaryOpType,
      BinaryOpType,
      TernaryOpType,
      DataType>;
  OpType op_type;

 public:
  static FunctionType fromVal(Val* val);

  //! Evaluate this type of function, given some arguments
  PolymorphicValue evaluate(const std::vector<PolymorphicValue>& inputs) const;
};

//! These objects mimic the Val AST and can be used to record input Vals and to
//! select the form of simplified values that we will need to construct.
//!
//! Multiple Val*s can represent a single ASTNode.
struct ASTNode {
  //! This describes the type of the definition, but not the actual arguments.
  FunctionType definition;

  //! Unrolled loop indices are not constants as Vals (i.e. v->isConstInt() is
  //! false), but in the generated kernel they are constant. This is useful for
  //! analyzing register usage. See
  //! https://github.com/csarofeen/pytorch/pull/2276 and related PRs
  bool is_unrolled_loop_index = false;

  //! Compute a coarse estimate of the complexity of computing this value.
  size_t complexity = 0;

  //! This is a collection of Vals from the Fusion that have this exact form.
  //! This can be used during extraction to select pre-existing Vals with a
  //! desired form.
  std::unordered_set<Val*> representing_vals{};

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

 public:
  //! Given a Val*, map its definition and producers to ASTNodes
  //! recursively.
  static ASTNode fromVal(Val* val);
};

//! An ENode is an abstraction of a Val where its producers have been replaced
//! with EClasses. It is like an ASTNode, but whereas an ASTNode holds a Val, an
//! ENode holds a function symbol along with a vector of EClass IDs describing
//! equivalent producers. This lets us represent a combinatorially massive
//! amount of possible Vals with a handful of ENodes.
struct ENode {
  //! This describes the type of the definition, but not the actual arguments.
  FunctionType definition;

  //! EClass IDs of all function arguments.
  std::vector<Id> producer_ids;

  //! When a Val is registered using EGraph::registerVal(v), then v is
  //! associated with an ASTNode. Multiple of these can exist for one ENode,
  //! corresponding to the various members of producer EClasses.
  std::list<ASTNode> concrete_enodes;
};

} // namespace egraph

} // namespace nvfuser
