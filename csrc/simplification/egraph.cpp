// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <simplification/egraph.h>
#include <simplification/rules.h>

namespace nvfuser {

namespace egraph {

Id EGraph::registerVal(Val* val) {
  // Create an ASTNode for this Val
  auto a = ASTNode::fromVal(val);
  // Use the ASTNode HashCons to see if we have an existing registered node. If
  // so, replace a with it.

  // TODO: Finish this using the hashcons
  return 0;
}

Val* EGraph::getSimplifiedVal(
    Val* orig_val,
    std::vector<Val*>* acceptable_root_vals) {
  saturate();
  return orig_val;
}

PolymorphicValue EGraph::getMaybeConstantValue(Val* val) {
  Id id = registerVal(val);
  saturate();
  return getEClassFromId(id)->data.constant;
}

void EGraph::repair(Id eclass_id) {
  EClass* eclass = getEClassFromId(eclass_id);
  for ([[maybe_unused]] auto parent_id : eclass->parents) {
    // Parents are ENode consumers of eclass
    // TODO: upward merging
  }
}

void EGraph::rebuild() {
  while (!worklist_.empty()) {
    // deduplicate into a local variable while we iterate. The iterations might
    // add new elements to worklist_ in which case we will loop again here.
    std::unordered_set<Id> worklist_dedup;
    for (auto id : worklist_) {
      worklist_dedup.insert(uf_.find(id));
    }
    worklist_.clear();
    for (auto it = worklist_dedup.begin(); it != worklist_dedup.end(); it++) {
      Id eclass_id = *it;
      repair(eclass_id);
    }
  }
}

void EGraph::saturate() {
  if (saturated_ || total_exploration_time_ms_ > exploration_time_limit_ms_) {
    return;
  }

  using Clock = std::chrono::steady_clock;
  Clock::time_point start_time = Clock::now();
  float time_allowed_ms =
      exploration_time_limit_ms_ - total_exploration_time_ms_;
  float elapsed_ms = 0.0f;

  while (true) {
    saturated_ = rule_runner_.runMatching() == 0;
    if (saturated_) {
      break;
    }

    rule_runner_.applySubstitutions();

    // Since we applied some matches, we need to rebuild derived EClasses by
    // canonicalizing their ENodes and rebuilding their data.
    rebuild();

    elapsed_ms = std::chrono::duration_cast<std::chrono::duration<float>>(
                     (Clock::now() - start_time))
                     .count();
    if (elapsed_ms > time_allowed_ms) {
      break;
    }
  }
  total_exploration_time_ms_ += elapsed_ms;
}

Id EGraph::merge(Id a, Id b) {
  // First canonicalize a and b
  a = getCanonicalENodeId(a);
  b = getCanonicalENodeId(b);

  Id new_id = uf_.merge(a, b);

  // Since a and b are canonical, the merged Id must be either a or b
  EClass* new_eclass = getEClassFromId(new_id == a ? a : b);
  Id old_id = new_id == a ? b : a;
  new_eclass->mergeFrom(old_id);

  // Add the new class to the worklist so that we can repair it during the next
  // rebuild
  worklist_.push_back(new_id);

  return new_id;
}

/*static*/ thread_local EGraph* EGraphGuard::active_egraph_ = nullptr;

EGraphGuard::EGraphGuard(EGraph* egraph) : prev_egraph_(active_egraph_) {
  active_egraph_ = egraph;
}

EGraphGuard::~EGraphGuard() {
  active_egraph_ = prev_egraph_;
}

// Cast to non-cast because many users need it.
/*static*/ EGraph* EGraphGuard::getCurEGraph() {
  return active_egraph_;
}

/*static*/ void EGraphGuard::setCurEGraph(EGraph* egraph) {
  active_egraph_ = egraph;
}

} // namespace egraph

} // namespace nvfuser
