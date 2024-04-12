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
#include <simplification/rules.h>
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

//! This class is built to simplify Vals using E-graphs. The method we employ is
//! based on equality saturation with rebuilding, as described in detail here:
//! Willsey et al. egg: Fast and Extensible Equality Saturation. POPL 2021.
//! https://doi.org/10.5281/zenodo.4072013
class EGraph {
 public:
  //! Add val and merge its EClass with that of `true`.
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
  // TODO: Instead of this acceptable_root_vals argument, we should pass a
  // description of the scope in which all symbols need to be visible. See the
  // note [Modeling term visibility in ASTNode] in enodes.h
  Val* getSimplifiedVal(
      Val* orig_val,
      std::vector<Val*>* acceptable_root_vals = nullptr);

  //! Get the constant value (if any) of the equivalence class that this Val*
  //! belongs to. Note that this is an equivalent but lighter-weight form of
  //!   extract(registerVal(predicate))->getValue()
  //! In particular, this method does not perform a full call to extract().
  PolymorphicValue getMaybeConstantValue(Val* val);

 protected:
  friend ENode;
  friend EClass;
  friend RuleRunner;

  //! We generally prefer to pass around Id integers instead of pointers. We can
  //! get the pointer in constant time as it's held in the eclass_up_ vector.
  ENode* getENodeFromId(Id a) const {
    return enodes_up_.at(a).get();
  }

  //! Note that this is non-const since uf_.find(a) is non-const due to path
  //! compression.
  Id getCanonicalENodeId(Id a) {
    return uf_.find(a);
  }

  //! Note that this is non-const since uf_.find(a) is non-const due to path
  //! compression.
  EClass* getEClassFromId(Id a) {
    return eclasses_up_.at(getCanonicalENodeId(a)).get();
  }

  //! Merge two EClasses
  Id merge(Id a, Id b);

  Id numENodes() const {
    return uf_.size();
  }

  //! Take ownership of an ENode by adding it to enodes_up_, enlarge the
  //! union-find to hold it, and create a new singleton eclass containing only
  //! this ENode. Note that if this ENode already exists (according to the
  //! hashcons), then we will reuse it, merging in the list of ASTNodes
  //! representing enode_ptr to the existing ENode.
  Id add(std::unique_ptr<ENode> enode_ptr);

 private:
  //! Replace all producer EClass IDs with their canonicalized versions
  void canonicalizeENode(ENode& n) {
    for (Id& producer_id : n.producer_ids) {
      producer_id = getCanonicalENodeId(producer_id);
    }
  }

  //! After we have merged EClasses, the hashcons structure might not
  //! This method is called upward merging: see Section 3.1 of Willsey et al.
  //! 2021.
  void repair(Id eclass_id);

  //!
  //! See Section 3.2 of Willsey et al. 2021.
  void rebuild();

  //! If necessary (see saturated_), iteratively find apply matches and
  //! rebuild().
  void saturate();

  //! Return an optimal Val* representing an ENode. First the selected ASTNode
  //! from the eclass is extracted. Then if that ASTNode has a representing
  //! Val*, we return it. If not, then we recursively obtain a Val* for each of
  //! the ASTNode's producer ASTNodes and construct a new Val* combining those
  //! producers into a simplified result.
  Val* extract(Id eclass_id);

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
  //  - The map M is eclasses_up_[getCanonicalENodeId()]->members and consists
  //  of a |->
  //    eclasses_up_[uf_.find(a)].
  //  - The HashCons H is called hashcons_.

  //! The UnionFind used to define the equivalence relation describing EClasses
  UnionFind<Id> uf_;

  //! This maps ENode IDs to EClass IDs.
  //! Importantly, we maintain "The HashCons Invariant" which states that given
  //! a "canonical" ENode n (see canonicalize()) this object will map n to
  //! uf_.find(n.id) (see Definition 2.7 of Willsey et al. 2021).
  // TODO
  // ENodeHashCons<Id> hashcons_;

  //! This is the object responsible for matching during each iteration of
  //! explore()
  RuleRunner rule_runner_;

  //! This holds a vector of EClasses that were merged during this iteration and
  //! need to be repaired. These might not be canonical or even unique.
  std::vector<Id> worklist_;

  //! Soft limit for the time spent by explore(). Note that an iteration might
  //! finish after this time has elapsed, but no iteration will be launched
  //! after that. In case explore() is called multiple times, this limit is
  //! applied to the total runtime across all invocations.
  float exploration_time_limit_ms_ = 5000.0f;
  float total_exploration_time_ms_ = 0.0f;

  //! Determines whether we need to run equality saturation. This defaults to
  //! true and is set to false whenever equality saturation completes with no
  //! new rule matches. It is reset to true whenever an ENode is added.
  bool saturated_ = true;
};

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

} // namespace egraph

} // namespace nvfuser
