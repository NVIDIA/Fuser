// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <simplification/egraph.h>
#include <simplification/egraph_type.h>

#include <list>

namespace nvfuser {

namespace egraph {


// Patterns allow us to quickly find matches that may be relevant to Rules.
//
// A Pattern resembles an ENode. It contains a function symbol and a collection
// of subpatterns.
//
// For example, consider how we might represent the rewrite rule (a * b) % a => b
//
//   Pattern (%) 
//     Pattern (*)
//       a
//       b
//     b
//     
// Here parens indicate the function symbol for each pattern and square
// brackets indicate "roles", i.e. the set of positions that could possibly
// belong to distinct eclasses. Indentation denotes that these are children of
// the parent.
using RoleId = uint8_t;
struct Pattern : FunctionSymbol {
  // If this FunctionSymbol has no definition, then it represents a pattern
  // that could match any EClass. It corresponds to a role, since it may be
  // part of a larger pattern. If op_type is anything other than
  // std::monostate, this attribute is ignored.
  std::optional<RoleId> role;

  // Currently we own producer patterns here. In the future we could try to
  // reuse rules so that subpattern matches don't require separate matching.
  // TODO: think about pattern re-use between rules for acceleration
  std::vector<Pattern> producer_patterns;

  //! Condition to check whether the Pattern actually matches given the roles
  //! found in the pattern. This condition can access analysis data.
  std::function<bool(EGraph*, const std::vector<Id>&())> condition =
      [](EGraph* eg, const std::vector<Id>& role_eclasses) { return true; };
};

//! This describes the structure of the substitution ENode. The substitution
//! ENode itself can be formed by simply replacing roles with Ids.
struct Substitution : FunctionSymbol {
  std::vector<Substitution> producer_substitutions;
};

//! This represents a very simple rule that matches a single pattern and 
struct SimpleRule {
  Pattern pattern;

  //! When we find a match, we will apply these substitutions to the matching
  //! ENode
  std::vector<Substitution> substitution;
};

//! A Match merely maps roles from a Pattern to EClass Ids. To create an ENode
//! we fill out Substitution by replacing each role with its corresponding Id
//! using Match::role_eclasses.
struct Match {
  SimpleRule* rule;

  //! ENode of the match
  Id matched_enode;

  // A mapping from roles to eclasses
  std::vector<Id> role_eclasses;
};

// Example: (x == x) => true if dtype(x) == Int
//
void foo() {
  auto intOnly = [](EGraph* eg, const std::vector<Id>& role_eclasses) {
    return eg->getEClassFromId(role_eclasses.at(0))->data.dtype ==
        DataType::Int;
  };
  Pattern p{
      BinaryOpType::Eq,
      std::nullopt,
      {Pattern{std::monostate, 0}, Pattern{std::monostate, 0}},
      intOnly};
  Substitution s{};
  SimpleRule r{p, {s}};
}

RuleRunner::RuleRunner() {
  // TESTING RULES

  // Some rules for testing only. This fires always and never matches.
  Rule never_fire{
      .name = "never-fire",
      .is_eligible_fn = [](ENode* n) -> bool { return false; },
      .check_match_fn = [](RuleRunner* runner, ENode* n) {}};
  rules_.push_back(never_fire);

  NVF_ERROR(
      rules_.size() < std::numeric_limits<RuleId>::max(),
      "RuleId type ",
      typeid(RuleId).name(),
      " has capacity ",
      std::numeric_limits<RuleId>::max(),
      " but there are ",
      rules_.size(),
      // Overhead is due to the encoding of worklist e.g. running 4 rules
      // requires a worklist encoded as 4 0 1 2 3 even though the highest RuleId
      // used is 3.
      " defined rules and RuleRunner has overhead of 1 element.");
}

size_t RuleRunner::runMatching() {
  NVF_ERROR(
      substitutions_.empty(),
      "runMatching() must not be called twice before applySubstitutions()");

  EGraph* eg = EGraphGuard::getCurEGraph();

  // Before matching, check for unseen ENodes and insert them into the worklist
  // if necessary. Because of this, we don't need to explicitly register new
  // ENodes when they are created.
  std::vector<RuleId> enode_rules;
  for (Id unseen_enode_id = seen_enodes_; unseen_enode_id < eg->numENodes();
       ++unseen_enode_id) {
    ENode* unseen_enode = eg->getENodeFromId(unseen_enode_id);
    enode_rules.clear();
    for (RuleId r_id : c10::irange(rules_.size())) {
      const Rule& r = rules_[r_id];
      // TODO: rules can be filtered with an EnableOption here
      if (r.is_eligible_fn(unseen_enode)) {
        enode_rules.push_back(r_id);
      }
    }
    // Encode and append enode_rules to worklist
    worklist_.reserve(worklist_.size() + 1LL + enode_rules.size());
    worklist_.push_back(enode_rules.size());
    worklist_.insert(worklist_.end(), enode_rules.begin(), enode_rules.end());
  }
  seen_enodes_ = eg->numENodes();

  // After this point, no new enodes will be processed in the worklist until the
  // next call to the present function. This is intentional, since running rules
  // on ENodes before their original substitutions are performed could lead to
  // infinite recursion.

  // NOTE: This algorithm assumes that new rules are not added for seen ENodes
  // during matching of a Rule. However, rules can be removed for a given ENode
  // if they mark themselves as exhausted. In this case we need to decrement the
  // rule count for that ENode and skip it in the worklist. This type of
  // operation can be done in place while looping over the worklist in ascending
  // order.
  size_t read_pos = 0;
  std::vector<RuleId>::iterator write_it = worklist_.begin();
  Id enode_id = 0;
  substitutions_.clear();
  while (read_pos < worklist_.size()) {
    ENode* enode = eg->getENodeFromId(enode_id++);
    enode_rules.clear();
    for (RuleId i = worklist_[read_pos++]; i > 0; --i) {
      const RuleId rule_id = worklist_[read_pos++];
      const Rule& rule = rules_[rule_id];
      current_rule_exhausted_ = false;
      rule.check_match_fn(this, enode);
      if (!current_rule_exhausted_) {
        enode_rules.push_back(rule_id);
      }
    }
    // write enode_rules back to worklist, since it contains non-exhausted rules
    // for this enode
    *(write_it++) = enode_rules.size();
    worklist_.insert(write_it, enode_rules.begin(), enode_rules.end());
  }

  return substitutions_.size();
}

void RuleRunner::applySubstitutions() {
  EGraph* eg = EGraphGuard::getCurEGraph();
  for (auto& [old_id, new_id] : substitutions_) {
    eg->merge(old_id, new_id);
  }
}

} // namespace egraph

} // namespace nvfuser
