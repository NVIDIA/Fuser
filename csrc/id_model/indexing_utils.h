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

} // namespace indexing_utils
} // namespace nvfuser
