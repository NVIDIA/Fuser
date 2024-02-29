// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <simplification/egraph.h>
#include <simplification/egraph_type.h>

#include <string_view>

namespace nvfuser {

namespace egraph {

class RuleRunner;

// If RuleRunner.rules_ exceeds this capacity then an error will be thrown in
// its constructor.
using RuleId = uint8_t;

//! Describes a rewrite rule that inspects ENodes. Based on the inspection, it
//! creates new ENodes and merges EClass. Each rule may depend on the structure
//! of the ENode itself (pattern matching), and it can also make use of the
//! `AnalysisData` associated to each of the ENode's producer EClasses and so
//! on.
struct Rule {
  std::string name = "unnamed-rule";

  //! Function that checks whether this Rule applies to this ENode based on a
  //! shallow check of the ENode attributes ignoring the producer EClasses. For
  //! example, consider the following rule:
  //!
  //!  Rule{
  //!    .name="gt-implies-not-le",
  //!    .is_eligible_fn=[](ENode* n) {
  //!        return n->definition.symbol == ENodeFunctionSymbol::BinaryOp &&
  //!               n->definition.op_type == BinaryOpType::GT;},
  //!    .check_match_fn=[](RuleRunner* runner, ENode* n) { ... }};
  //!
  //! This indicates that only ENodes that are greater-than expressions should
  //! be added to .targets.
  std::function<bool(ENode*)> is_eligible_fn = [](ENode*) {
    NVF_ERROR("Running uninitialized Rule");
    return false;
  };

  //! Check whether an element of `targets` is a match. An iterator is passed so
  //! that the class can erase the ENode from the target list if the rule
  //! should no longer be fired.
  std::function<void(RuleRunner*, ENode*)> check_match_fn =
      [](RuleRunner* runner, ENode* n_it) {
        NVF_ERROR("Running uninitialized rule");
      };
};

//! This is the main interface for running the matching pass. This holds a
//! collection of Rules to execute, each of which maintains a list of target
//! ENodes.
class RuleRunner {
 public:
  RuleRunner();

  //! Run all rules and return the number of matches detected.
  size_t runMatching();

  //! Apply substitutions found during runMatching().
  void applySubstitutions();

  //! Registers a substitution; i.e. request merging the EClasses of ENodes a
  //! and b. Note that if a == b, then this method has no effect. Otherwise we
  //! push the a,b pair along with the currently running rule ID onto the
  //! substitutions_ vector.
  void registerSub(Id a, Id b);

  //! This method should be called from any rule that detects it should not be
  //! called again for the given ENode since all possible substitutions have
  //! been performed.
  void markRuleExhausted();

 private:
  //! This allows us to use the RuleRunner as a context object, so that a match
  //! can mark itself exhausted and prevent it running again needlessly.
  bool current_rule_exhausted_;

  //! This is simply the number of ENodes we have proceesed in the worklist so
  //! far. When we begin runMatching(), if we detect that there are unseen
  //! ENodes in the EGraph then we gather process those ENodes to check their
  //! eligibility against our rules and add them to the worklist as appropriate.
  Id seen_enodes_ = 0;

  //! This tracks the currently running rule and associates it with each
  //! registered substitution.
  size_t current_rule_;

  //! Collection of all rules, populated in the constructor
  std::vector<Rule> rules_;

  //! This is a flattened vector of vectors of Rules representing a worklist,
  //! with one vector of Rule Ids for each ENode. It is a sparse representation
  //! of a NxR array of flags
  //!
  //!  N0 i0 i1 i2 N1 i3 i4 N2 N3 i5 ...
  //!
  //! Here Na indicates the length of the worklist for ENode a. After reading
  //! this length Na the following Na many IDs are the RuleIds of rules we
  //! should apply to ENode a.
  std::vector<RuleId> worklist_;

  //! When runMatching is called, this vector is cleared and each rule from
  //! rules_ is run on each term for which that rule has not been exhausted.
  std::vector<std::pair<Id, Id>> substitutions_;
};

} // namespace egraph

} // namespace nvfuser
