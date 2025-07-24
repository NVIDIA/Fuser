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

} // namespace indexing_utils
} // namespace nvfuser
