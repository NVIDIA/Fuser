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

// The classes declared here implement an E-Graph approach to expression
// simplification that can simultaneously be used for theorem proving.
//
// In this approach, we simplify terms by repeatedly adding rewritten terms and
// marking them as equivalent to the original ones. Finally, once we run out of
// rewrites or surpass a time limit, we finalize the simplification by
// extracting a particular "simplest" member of each equivalence class in order
// to construct a simplified term.
//
// The equivalence classes that we construct can be used for theorem proving by
// inspecting whether a bool-valued term is equivalent to a constant `true`
// term. In this way, as we iterate and the equivalence classes grow and merge,
// more and more proven propositions (i.e. members of the equivalence class
// [true]) are collected.
//
// The method we employ is based on equality saturation with rebuilding, as
// described in detail here:
// Willsey et al. egg: Fast and Extensible Equality Saturation. POPL 2021.
// https://doi.org/10.5281/zenodo.4072013
// Our terminology in the classes below attempts to match the paper when
// possible.

using EClassIdType = uint32_t;

//! An ENode represents either a constant, a definition-less scalar (such as a
//! loop variable or input scalar), or a scalar defined by some function. This
//! is a lightweight surrogate for Vals which additionally holds a collection of
//! actual Vals having the exact same form. We support the following subset of
//! Exprs.
//!
//! NOTE: although we support BinaryOp, we also _flatten_ expressions like
//!
//!   u + (v + (w + (x + (y + z))))
//!
//! using the CommutativeBinaryOp symbol. ENodes with this symbol might have
//! more than 2 arguments and their order is arbitrary; two ENodes with this
//! symbol and the same op_type, with the same collection of arguments but in
//! permutated order should always map to the same EClass ID.
enum ENodeFunctionSymbol {
  NoDefinition,
  LoadStoreOp,
  CastOp,
  UnaryOp,
  BinaryOp,
  TernaryOp,
  CommutativeBinaryOp,
};

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

//! An analysis holds properties of EClasses that are combined when EClasses are
//! merged.
//!
//! There is a mapping make(n) that maps an ENode to data (we call this function
//! AnalysisData::fromENode(n)).
//!
//! The analysis invariant must be preserved: the analysis data of an eclass
//! must always be equal to the result of
//!
//! For more details see Sec. 4.1 of Willsey et al. 2021.
struct AnalysisData {
  //! Each EClass must represent terms whose types match
  DataType dtype;

  //! EClasses can represent a single unique value (this is checked in join()).
  PolymorphicValue value;

 public:
  //! This is make(n) from Willsey et al. 2021.
  static AnalysisData fromENode(const ENode& a);

  //! Join this AnalysisData with data from another EClass to form data for
  //! their merged EClass.
  //!
  //! Here we check that dtypes and
  AnalysisData join(const AnalysisData& other) const {
    NVF_ERROR(
        dtype == other.dtype,
        "Attempted to merge EClasses with different dtypes");
    PolymorphicValue joined_value = value;
    if (value.hasValue() && other.value.hasValue()) {
      NVF_ERROR(
          value == other.value,
          "Attempted to merge EClasses with differing values ",
          value,
          " and ",
          other.value);
    } else if (value.hasValue()) {
      joined_value = value;
    } else {
      joined_value = other.value;
      return {.dtype = dtype, .value = joined_value};
    }
  }
};

//! An EClass is simply an equivalence class of ENodes. This represents Vals
//! that are all proven to have exactly the same value in the generated kernel.
//! It holds a datatype, and a list of parent ENodes. The parents are ENodes
//! representing functions having a member of this EClass as one of the
//! arguments.
struct EClass {
  AnalysisData data;

  //! ENodes that are members of this class
  std::list<EClassIdType> members;

  //! Holds pairs of ENodes and EClasses. Parent ENodes represent functions
  //! some of whose arguments are members of this EClass. The corresponding
  //! EClasses are the EClasses of those ENodes which might need to be merged
  //! during repair(). See the code listing in Fig. 4 of Willsey et al. 2021.
  std::list<std::pair<EClassIdType, EClassIdType>> parents;

 public:
  static EClass fromENode(const ENode& n) {
    // create new analysis data
    return {.data = AnalysisData::fromENode(n), .members = {n}, .parents = {}};
  }
};

//! This class is built to simplify Vals using E-graphs. The method we employ is
//! based on equality saturation with rebuilding, as described in detail here:
//! Willsey et al. egg: Fast and Extensible Equality Saturation. POPL 2021.
//! https://doi.org/10.5281/zenodo.4072013
//!
//! [E-Graph Expression Simplification (Interface)]
//! In this approach, Vals can be simplified in batches. The user first provides
//! a collection of values that will need simplifying, via `registerVal(Val*
//! val)`. Additionally, axioms can be declared via the assumeTrue method.
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
  ENode* registerVal(Val* val) {
    auto c = ASTNode::fromVal(val);
  }

  //! Register val and merge its EClass with that of `true`.
  void assumeTrue(Val* val);

  //! Extract a Val* that is a simplified form of the original.
  //!
  //! This process works by selecting a representative ASTNode which is
  //! just like an ENode but whose arguments are ENode IDs instead of EClass
  //! IDs. This is a recursive process as we first must select ASTNodes
  //! for each producer eclass and then find an ASTNode in that class and so on.
  //! In order to choose an ASTNode for a class, we use the following rules:
  //!  1. We prioritize constants. Any constant ENode in the class will be
  //!   chosen.
  //!  2. If no constant ENodes exist, then an ENode of type NoDefinition, i.e.
  //!   a non-constant ENode, will be chosen. There may be multiple such ENodes,
  //!   in which case we choose the one with the lowest ENode ID for
  //!   repeatability.
  //!  3. We then prefer ASTNodes with the lowest "complexity"; actually a
  //!   crude surrogate for the computational demand of each operation, summed
  //!   across all the inputs of the ASTNode recursively. For example, a
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

  //! If n is in hashcons_, then map it to its canonical ENode. If it is not yet
  //! in hashcons_
  ENode canonicalizeENode(const ENode& n) {}

  //! After we have merged EClasses, the hashcons structure might not
  //! This method is called upward merging: see Section 3.1 of Willsey et al.
  //! 2021.
  void repair();

  //! See Section 3.2 of Willsey et al. 2021.
  void rebuild();

  //! Note that this is non-const since uf_.find(a) is non-const due to path
  //! compression.
  EClass* getCanonicalEClass(EClassIdType a) {
    return eclasses_up_.at(uf_.find(a)).get();
  }

 private:
  //! These containers owns all of this EGraph's ENodes and EClass objects. Note
  //! that these vectors should always have the same length. EClass or ENode
  //! IDs refer to positions within these vectors.
  std::vector<std::unique_ptr<ENode>> enodes_up_;
  std::vector<std::unique_ptr<EClass>> eclasses_up_;

  // Definition 2.1 of Willsey et al. describes the basic structure of an
  // EGraph. Such an object consists of a UnionFind U over EClass IDs, an EClass
  // map M mapping EClass IDs to EClasses, and a HashCons H from ENodes to
  // EClass IDs.
  //  - The UnionFind U is called uf_
  //  - The map M is getCanonicalEClass() and consists of a |->
  //    eclasses_up_[uf_.find(a)].
  //  - The HashCons H is called hashcons_.

  //! The UnionFind used to define the equivalence relation describing EClasses
  UnionFind<EClassIdType> uf_;

  //! This maps ENode IDs to EClass IDs.
  //! Importantly, we maintain "The HashCons Invariant" which states that given
  //! a "canonical" ENode n (see canonicalize()) this object will map n to
  //! uf_.find(n.id) (see Definition 2.7 of Willsey et al. 2021).
  ENodeHashCons<EClassIdType> hashcons_;

  //! Soft limit for the time spent by explore(). Note that an iteration might
  //! finish after this time has elapsed, but no iteration will be launched
  //! after that.
  size_t exploration_time_limit_ms_ = 5000;
};

} // namespace egraph

} // namespace nvfuser
