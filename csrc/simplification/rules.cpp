// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <simplification/egraph.h>
#include <simplification/egraph_type.h>

namespace egraph {

RuleRunner::RuleRunner() {
  if (initialized) {
    return all_rules;
  }
  // This is where we define rules. Not that order does not matter.

  // TESTING RULES

  // Some rules for testing only. This fires always and never matches.
  rules_.push_back(Rule{"never-fires", [](ENode* n) { return true; }});

  // This matches all ENodes but does not perform any merges
  rules_.push_back(Rule{
      .name = "empty-match",
      .is_eligible_fn = [](ENode* n) { return true; },
      .check_match_fn = [&name](ENodeListIterator& n) { return Match(); }});

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
