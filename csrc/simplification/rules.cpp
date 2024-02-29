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
  Rule never_fire{
      .name = "never-fire",
      .is_eligible_fn = [](size_t rule_id, ENode* n) -> bool { return false; },
      .check_match_fn = [](size_t rule_id, ENodeListIterator& n_it)
          -> std::optional<Match> { return std::nullopt; }};
  rules_.push_back(never_fire);
  // rules_.push_back(
  // Rule{"never-fire", [](size_t rule_id, ENode* n) { return false; }});

  // This matches all ENodes but does not perform any merges
  // rules_.emplace_back(
  //    "empty-match",
  //    /*is_eligible_fn=*/[](size_t rule_id, ENode* n) { return true; },
  //    /*check_match_fn=*/
  //    [](size_t rule_id, ENodeListIterator& n) { return Match(rule_id); });
}

std::list<Match> RuleRunner::runMatching() {
  std::list<Match> all_matches;
  for (int r_id : c10::irange(rules_.size())) {
    // We attach IDs to matches to aid in debugging, since this will let us
    // print the name of the corresponding rule once we have the list of matches
    Rule& r = rules_[r_id];
    std::list<Match> r_matches = r.findMatches(r_id);
    all_matches.splice(all_matches.end(), r_matches);
  }
  return all_matches;
}

} // namespace egraph

} // namespace nvfuser
