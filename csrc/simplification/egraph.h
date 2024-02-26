// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <simplification/union_find.h>

#include <optional>

namespace nvfuser {

namespace egraph {

struct EClass;
struct ENode;

using EClassIdType = uint32_t;

//! This class is built to simplify Vals using E-graphs. The method we employ is
//! based on equality saturation with rebuilding, as described in detail here:
//! https://doi.org/10.5281/zenodo.4072013
//!
//! [E-Graph Expression Simplification (Interface)]
//! In this method, Vals can be simplified in batches. The user first provides a
//! collection of values that will need simplifying, via `registerVal(Val*
//! val)`. Additionally, axioms can be declared via the following helper methods:
//!  - assumeTrue
//!  - assumeFalse
//!
//! Simplified values can then be extracted via `getSimplifiedVal(Val*)`.
class EGraphSimplifier {
 public:
  //! Register a Val so it becomes visible.
  //!
  //! This can either create a new ENode to represent val, or reuse an existing
  //! one:
  //!  - If val has a definition, then its inputs are registered and a new ENode
  //!  is created if an equivalent one does not exist.
  //!  - If val has no definition, we always create a new ENode for it
  //!
  //! In case val is an immediate constant, we additionally look for any
  //! existing registered ENodes with that value and if we find any, we merge
  //! their EClasses.
  ENode* registerVal(Val* val);

  //! Register val and merge its EClass with that of `true`.
  void assumeTrue(Val* val);

  //! Extract a Val* that is a simplified form of the original.
  //!
  //! This process works by selecting a representative ConcreteENode which is
  //! just like an ENode but whose arguments are ENode IDs instead of EClass
  //! IDs. This is a recursive process as we first must select ConcreteENodes
  //! for each argument eclass and then find a ConcreteENode for ENode in that
  //! class and so on. In order to choose a ConcreteENode for a class, we use
  //! the following rules:
  //!  1. We prioritize constants. Any constant ENode in the class will be
  //!   chosen.
  //!  2. If no constant ENodes exist, then an ENode of type NoDefinition, i.e.
  //!   a non-constant ENode, will be chosen. There may be multiple such ENodes,
  //!   in which case we choose the one with the lowest ENode ID for
  //!   repeatability.
  //!  3. We then prefer ConcreteENodes with the lowest "complexity"; actually a
  //!   crude surrogate for the computational demand of each operation, summed
  //!   across all the inputs of the ConcreteENode recursively. For example, a
  //!   modulo or exponential operation will be considered more complex than an
  //!   add or multiply.
  //!
  //! If acceptable_root_vals is non-null, then when selecting a representative
  //! containing some ENodes without definitions (but no constant ENodes as
  //! those would be chosen with higher priority), we will select only from this
  //! vector of acceptable root Vals. This is useful for simplifying expressions
  //! within multiple scopes, where some variables might be out of scope for the
  //! purpose at hand.
  Val* getSimplifiedVal(
      Val* orig_val,
      std::vector<Val*>* acceptable_root_vals = nullptr);

  //! Check whether a Val, whose type must be DataType::Bool, has been proven.
  //! If not, return std::nullopt. If the predicate has been proven, return
  //! true. If its negation has been proven, return false.
  //!
  //! Note that this is an equivalent but lighter-weight form of
  //!   extract(registerVal(predicate))->getValue()
  //! In particular, this method does not perform a full call to extract(),
  //! instead only a shallow check is done to determine whether
  //! registerVal(predicate) is equivalent to true_enode_ or false_enode_.
  std::optional<bool> provenValue(Val* predicate);

 private:
  //! [E-Graph Expression Simplification (Internals)]
  //! Two phases are used

  //! Run the exploration phase. This phase is iterative, and repeats until
  //! either the time limit is reached.
  void explore();

  // Suppose we start with
  //   a < b  ~  true
  //   c < d  ~  false
  //   b == d ~  true
  // Then during exploration, we might query a < c to see whether it has been
  // proven true yet. This introduces another enode, so that we have
  //   a < b  ~  true
  //   c < d  ~  false
  //   b == d ~  true
  //   a < c  ~  {}
  // Then we will iteratively explore, using pattern matching on the parents of
  // a, b, and c:
  //   a < b  ~  true
  //   c < d  ~  false
  //   b == d ~  true
  //   a < c  ~  {}
  //   // Rule 1: merge negation of all enodes equiv to true with false and vice
  //   versa
  //   !(a < b) ~ false
  //   !(c < d) ~ true
  //   b != d ~  false
  //   // Rule 2: trivial rewrites e.g. !(x < y) => x >= y and x < y => y > x
  //   a >= b  ~  false
  //   c >= d  ~  true
  //   b > a   ~  false
  //   d > c   ~  false
  //   d == b  ~  true
  //   // Rule 3: if x == y ~ true then merge classes of x and y
  //   b == d  => merge {b} and {d}
  //   // Rule 4: Look at parent enodes of each side of inequality. If any of
  //   // those parents is also an inequality, perform some basic matches such
  //   // as x < y && y <= z => x < z
  //   {c} >= {b, d} is true && a < b, so merge a < b ~ true
  // At this point, a < c is proven, but we will continue exploring until
  // saturation (i.e. no more matches) or the time limit is reached.

  //! Return an optimal Val* representing an ENode
  Val* extract(ENode* enode);

 private:
  // This class owns all its ENodes and EClass objects
  std::vector<std::unique_ptr<ENode>> enodes_up_;
  std::vector<std::unique_ptr<EClass>> eclasses_up_;

  //! The UnionFind used to define the equivalence relation describing EClasses
  UnionFind<EClassIdType> uf_;

  //! An immediate constant Val* is represented by a unique ENode. This maps any
  //! such value to that ENode.
  std::unordered_map<PolymorphicValue, ENode*> value_to_enode_;

  //! Hold references to true and false eclasses
  EClass *true_enode_ = nullptr, *false_enode_ = nullptr;

  //! Soft limit for the time spent by explore(). Note that an iteration might
  //! finish after this time has elapsed, but no iteration will be launched
  //! after that.
  size_t exploration_time_limit_ms_ = 5000;
};

// An ENode represents either a constant, a definition-less scalar (such as a
// loop variable or input scalar), or a scalar defined by some function. This is
// a lightweight surrogate for Vals which additionally holds a collection of
// actual Vals having the exact same form. We support the following subset of
// Exprs.
//
// NOTE: although we support BinaryOp, we also _flatten_ expressions like w + (x
// + (y + z)) using symbols named FlattenedAdd and similar.
enum ENodeFunctionSymbol {
  NoDefinition,
  LoadStoreOp,
  CastOp,
  UnaryOp,
  BinaryOp,
  TernaryOp,
  FlattenedAdd,
  FlattenedMul,
  FlattenedLogicalOr,
  FlattenedLogicalAnd,
  FlattenedBitwiseOr,
  FlattenedBitwiseAnd
};

struct ENode {
  //! What type of node is this
  ENodeFuncSymbol function_symbol;

  //! This determines the actual operation, e.g. BinaryOpType::Add
  std::variant<UnaryOpType, BinaryOpType, TernaryOpType> op_type;

  //! EClass IDs of all function arguments
  std::vector<EClassIdType> arg_eclass_ids;

  //! We hold a set of Vals from the Fusion that map to this ENode.
  std::unordered_set<Val*> representing_vals;
};

//! An EClass is simply an equivalence class of ENodes. This represents Vals
//! that are all proven to have exactly the same value in the generated kernel.
//! It holds a datatype, and a list of parent ENodes. The parents are ENodes
//! representing functions having a member of this EClass as one of the
//! arguments.
struct EClass {
  //! Position of this EClass within the EGraphSimplifier::eclasses_up_ vector.
  //! This ID is used 
  EClassIdType id;

  //! All members of a class must have exactly the same dtype in order to avoid
  //! implicit casts.
  DataType dtype;

  //! ENodes that are members of this class
  std::list<ENode*> members_;

  //! Parent ENodes represent functions of members of this EClass.
  std::list<ENode*> parents;
};

} // namespace egraph

} // namespace nvfuser
