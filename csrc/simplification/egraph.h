// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <simplification/eclass.h>
#include <simplification/egraph_type.h>
#include <simplification/enode.h>
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

class EGraphGuard {
 public:
  //! Set the active fusion so it can be manipulated.
  NVF_API explicit EGraphGuard(EGraph* egraph);

  NVF_API ~EGraphGuard();

  NVF_API static EGraph* getCurEGraph();
  static void setCurEGraph(EGraph* egraph);

 private:
  EGraph* prev_egraph_;

  static thread_local EGraph* active_egraph_;
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
