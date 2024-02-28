// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <simplification/egraph.h>

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

Id EGraph::merge(Id a, Id b) {
  // First canonicalize a and b
  a = getCanonicalEClassId(a);
  b = getCanonicalEClassId(b);

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
