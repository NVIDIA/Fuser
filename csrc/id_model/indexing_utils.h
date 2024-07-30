// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <device_lower/analysis/index_compute.h>
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <id_model/id_model.h>
#include <id_model/to_string.h>

#include <id_model/utils.h>

namespace nvfuser {
namespace indexing_utils {

// Get a matching ForLoop for a given loop iter domain. There may not
// be such a loop if this loop-nest is for initializing a reduction
// buffer.
inline ForLoop* getForLoop(
    IterDomain* loop_id,
    const std::vector<ForLoop*>& for_loops,
    const ValGraph& loop_graph) {
  auto it = std::find_if(
      for_loops.begin(), for_loops.end(), [&](ForLoop* for_loop) -> bool {
        IterDomain* for_loop_id = for_loop->iter_domain();
        return loop_graph.disjointValSets().strictAreMapped(
            loop_id, for_loop_id);
      });
  if (it != for_loops.end()) {
    return *it;
  } else {
    return nullptr;
  }
}

// Get the promotion domain of a given loop domain.
inline IterDomain* getLoopPromotion(
    IterDomain* loop_id,
    const IdModel& id_model) {
  const auto& loop_graph = id_model.idGraph(IdMappingMode::LOOP);
  const auto& loop_promotion_map = id_model.loopPromotionMap();
  const auto& loop_group = loop_graph.toGroup(loop_id);

  auto loop_promotion_map_it = loop_promotion_map.find(loop_group);
  NVF_ERROR(
      loop_promotion_map_it != loop_promotion_map.end(),
      "No loop promotion found: ",
      loop_id->toString(),
      ". Loop group: ",
      nvfuser::toString(loop_group));

  return loop_promotion_map_it->second;
}

// Check if unswitching a given for-loop actually matters. For example,
// if a loop is parallelized, unswitching doesn't mean anything as we
// don't unswitch threading dimensions, e.g., "threadIdx.x + ... < N"
// is generated rather than "blockDim.x + ... < N".
inline bool isEffectiveUnswitchLoop(ForLoop* fl) {
  // Threaded domain is not unswitched
  if (fl->iter_domain()->isThread() || fl->iter_domain()->isDeviceDim()) {
    return false;
  }

  // If it's vectorized, it must be true that any of the iteration
  // values can be used to generate the predicates of the tensor, so
  // unswitching has no effect. Same for loops that are known to be
  // safe to just predicate at the end.
  if (fl->iter_domain()->getParallelType() == ParallelType::Vectorize ||
      lower_utils::predicateAtEnd(fl)) {
    return false;
  }

  return true;
}

inline std::vector<ForLoop*> getMaxPathLoops(
    const std::vector<ForLoop*>& for_loops) {
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
         !within_unswitch && !lower_utils::predicateAtEnd(fl))) {
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

} // namespace indexing_utils
} // namespace nvfuser
