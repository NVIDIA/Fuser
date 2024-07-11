// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <id_model/id_model.h>

namespace nvfuser {

inline std::vector<ForLoop*> getMaxPathLoops(const std::vector<ForLoop*>& for_loops) {
  std::vector<ForLoop*> unswitched_domains;

  bool within_unswitch = false;

  for (const auto fl : for_loops) {
    auto parallel_type = fl->iter_domain()->getParallelType();

    if (parallel_type == ParallelType::Unswitch ||
        parallel_type == ParallelType::Unroll) {
      within_unswitch = true;
    }

    // Don't unswitch threaded loops even when unswitched
    if (fl->iter_domain()->isThread() ||
        (fl->iter_domain()->getParallelType() != ParallelType::Vectorize &&
         !within_unswitch && !predicateAtEnd(fl))) {
      continue;
    } else {
      unswitched_domains.push_back(fl);
    }
  }

  return unswitched_domains;
}

// TODO: Use this from getPredicateIndexReplacementMap
inline std::unordered_set<ValGroup> getMaxPathLoopDomains(
    TensorView* consumer_tv,
    const std::vector<ForLoop*>& for_loops,
    const ValGraph& loop_graph,
    const ValGraph& traversal_graph) {
  auto unswitched_loops = getMaxPathLoops(for_loops);
  std::unordered_set<ValGroup> max_path_loop_domains;

  for (auto loop_domain : consumer_tv->getLoopDomain()) {
    const auto& loop_group = loop_graph.toGroup(loop_domain);
    auto it = std::find_if(
        unswitched_loops.begin(),
        unswitched_loops.end(),
        [&loop_group](ForLoop* fl) -> bool {
          return loop_group->has(fl->iter_domain());
        });
    if (it != unswitched_loops.end()) {
      max_path_loop_domains.emplace(traversal_graph.toGroup(loop_domain));
    }
  }

  return max_path_loop_domains;
}

// Currently it's only Shared or Local but Global can be the case
// too.
bool isAllocationBasedOnLeaf(TensorView* tv) {
  return tv->getMemoryType() == MemoryType::Shared ||
      tv->getMemoryType() == MemoryType::Local;
}

} // nvfuser nvfuser
