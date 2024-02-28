// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <simplification/egraph_type.h>

#include <optional>

namespace nvfuser {

namespace egraph {

//! An ENode represents either a constant, a definition-less scalar (such as a
//! loop variable or input scalar), or a scalar defined by some function.

//! This struct describes a function without describing any of its arguments (or
//! even how many arguments there are).
struct FunctionType {
  //! What type of node is this
  ENodeFunctionSymbol function_symbol = ENodeFunctionSymbol::NoDefinition;

  //! This determines the actual operation, e.g. BinaryOpType::Add
  //! Note that the target DataType for CastOp can be inferred by the dtype of
  //! this ENode's EClass, but since we need to hash and compare ENodes we
  //! include that DataType here as data as well.
  std::variant<
      std::monostate,
      UnaryOpType,
      BinaryOpType,
      TernaryOpType,
      DataType>
      op_type = std::monostate;

 public:
  PolymorphicValue evaluate(const std::vector<PolymorphicValue>& inputs) const {
    switch (function_symbol) {
      case ENodeFunctionSymbol::NoDefinition:
        NVF_ERROR(
            false,
            "Cannot evaluate AST function that does not have a definition");
      case ENodeFunctionSymbol::LoadStoreOp:
        NVF_ERROR(inputs.size() == 1);
        return inputs[0];
        break;
        // TODO: Refactor ops so that we can call static versions of each op and
        // avoid creating new Exprs in the Fusion here.
      default:
        NVF_ERROR(false, "not yet implemented");
        return std::monostate;
    }
  }
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

 public:
  //! Given a Val*, map its definition and producers to ASTNodes
  //! recursively.
  static ASTNode fromVal(Val* val) {
    auto symb = ENodeFunctionSymbol::NoDefinition;
    if (Expr* def = val->definition()) {
      if (auto bop = dynamic_cast<BinaryOp*>(def)) {
      } else {
        NVF_ERROR(false, "Val ");
      }
    }
    return {.representing_vals = {val}};
  }
}

//! An ENode is an abstraction of a Val where its producers have been replaced
//! with EClasses. It is like an ASTNode, but whereas an ASTNode holds a Val, an
//! ENode holds a function symbol along with a vector of EClass IDs describing
//! equivalent producers. This lets us represent a combinatorially massive
//! amount of possible Vals with a handful of ENodes.
struct ENode {
  //! This describes the type of the definition, but not the actual arguments.
  FunctionType definition;

  //! EClass IDs of all function arguments.
  std::vector<EClassIdType> producer_eclass_ids;

  //! When a Val is registered using EGraph::registerVal(v), then v is
  //! associated with an ASTNode and that
  std::list<ASTNode> concrete_enodes;
};

} // namespace egraph

} // namespace nvfuser
