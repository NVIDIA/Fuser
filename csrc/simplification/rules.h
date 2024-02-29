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

//! This generic object defines a matched rewrite rule. See Figure 5 from
//! Willsey et al. 2021.
//!
//! Rewrite rules implement `Rule::checkMatch` which can construct one of these
//! objects if the pattern matches. When building a Match, a Rule may create new
//! ENodes, but should not merge any EClasses. Instead, they should describe the
//! merge like follows:
//!
//! match_fn = [&name](std::list<ENode*>::iterator& n_it) {
//!   Match match(name);
//!   ...
//!   match.recordMerge(id1, id2);
//!   ...
//!   return match;
//! };
class Match {
 public:
  Match(size_t rule_id) : rule_id_(rule_id) {}

  //! Record but do not perform a merge of two EClasses
  void recordMerge(Id a, Id b) {
    merges_.emplace_back(a, b);
  }

  //! Apply all of the recorded merges
  void apply(EGraph* egraph) const;

 private:
  size_t rule_id_;
  std::list<std::pair<Id, Id>> merges_;
};

using ENodeListIterator = std::list<ENode*>::const_iterator;

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
  //!    .check_match_fn=[](size_t rule_id, ENodeListIterator& n_it) { ... }};
  //!
  //! This indicates that only ENodes that are greater-than expressions should
  //! be added to .targets.
  std::function<bool(size_t, ENode*)> is_eligible_fn = [](size_t rule_id,
                                                          ENode*) {
    NVF_ERROR("Running uninitialized Rule");
    return false;
  };

  //! Check whether an element of `targets` is a match. An iterator is passed so
  //! that the class can erase the ENode from the target list if the rule
  //! should no longer be fired.
  std::function<std::optional<Match>(size_t, ENodeListIterator&)>
      check_match_fn = [](size_t rule_id, ENodeListIterator& n_it) {
        NVF_ERROR("Running uninitialized rule");
        return std::nullopt;
      };

  //! Obtain a list of match objects for this rule. At each iteration of
  //! equality saturation these lists are chained together to collect all
  //! matches across all rules before applying the matches and rebuilding.
  std::list<Match> findMatches(size_t rule_id) const {
    std::list<Match> matches;
    for (auto n_it = targets.begin(); n_it != targets.end(); n_it++) {
      std::optional<Match> m = check_match_fn(rule_id, n_it);
      if (m.has_value()) {
        matches.push_back(m.value());
      }
    }
    return matches;
  }

  //! This is a list of target ENodes. Each of these will be considered for
  //! matching during each pass. If this Rule determines that there is no need
  //! to try matching an ENode again, it can be removed from this list. Note
  //! that once an ENode is removed, it will never be reinserted.
  std::list<ENode*> targets = {};
};

//! This is the main interface for running the matching pass. This holds a
//! collection of Rules to execute, each of which maintains a list of target
//! ENodes.
class RuleRunner {
 public:
  RuleRunner();

  std::list<Match> runMatching();

 private:
  std::vector<Rule> rules_;
};

} // namespace egraph

} // namespace nvfuser
