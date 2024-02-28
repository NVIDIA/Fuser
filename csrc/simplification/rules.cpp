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

void Match::apply() const {
  EGraph* eg = EGraphGuard::getCurEGraph();
  for (auto& [a, b] : merges_) {
    eg->merge(a, b);
  }
}

RuleRunner::RuleRunner() {
  // TESTING RULES

  // Some rules for testing only. This fires always and never matches.
  rules_.emplace_back(
      "never-fires", [](size_t rule_id, ENode* n) -> bool { return true; });

  // This matches all ENodes but does not perform any merges
  rules_.emplace_back(
      "empty-match",
      /*is_eligible_fn=*/[](size_t rule_id, ENode* n) { return true; },
      /*check_match_fn=*/
      [](size_t rule_id, ENodeListIterator& n) { return Match(rule_id); });

  return all_rules;
}

std::list<Match> RuleRunner::runMatching() {
  std::list<Match> all_matches;
  for (auto& r : rules_) {
    std::list<Match> r_matches = r.findMatches();
    for (auto& m : r_matches) {
      // Label each match with the name of the rule that produced it
      m.name = r.name;
    }
    all_matches.splice(all_matches.end(), r_matches);
  }
  return all_matches;
}

} // namespace egraph

} // namespace nvfuser
