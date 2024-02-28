// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <simplification/egraph.h>

namespace egraph {

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
