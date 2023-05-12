// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <ir_utils.h>
#include <partial_split_map.h>

namespace nvfuser {

void PartialSplitMap::build(Fusion* fusion) {
  auto used_vals = ir_utils::allTvs(fusion);

  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    auto exprs = StmtSort::getExprs(
        fusion, {tv->getLeafDomain().begin(), tv->getLeafDomain().end()});
    for (auto split : ir_utils::filterByType<Split>(exprs)) {
      // Only needs to check root domains as partial split is only
      // allowed with root domains
      if (std::find(
              tv->getRootDomain().begin(),
              tv->getRootDomain().end(),
              split->in()) == tv->getRootDomain().end()) {
        continue;
      }
      auto root_domain = split->in();
      auto start_offset = split->startOffset();
      start_offset_map_.insert({root_domain, start_offset});
      auto stop_offset = split->stopOffset();
      stop_offset_map_.insert({root_domain, stop_offset});
    }
  }
}

Val* PartialSplitMap::getStartOffset(IterDomain* root_domain) const {
  auto it = start_offset_map_.find(root_domain);
  if (it == start_offset_map_.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}

Val* PartialSplitMap::getStopOffset(IterDomain* root_domain) const {
  auto it = stop_offset_map_.find(root_domain);
  if (it == stop_offset_map_.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}

} // namespace nvfuser
