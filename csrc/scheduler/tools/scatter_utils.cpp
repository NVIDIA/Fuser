// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/all_nodes.h>
#include <scheduler/tools/scatter_utils.h>

namespace nvfuser {
namespace scheduler_tools {

void scheduleScatterLoopDomainAsIndexDomain(ScatterOp* sop) {
  auto index_tv = sop->index()->as<TensorView>();
  auto out_tv = sop->out()->as<TensorView>();
  
  std::vector<IterDomain*> out_loop;
  out_loop.reserve(index_tv->getLogicalDomain().size());
  
  std::ranges::transform(
      index_tv->getLogicalDomain(), std::back_inserter(out_loop), [](IterDomain* id) {
        return IterDomainBuilder(id).build();
      });
  
  out_tv->domain()->setLoopDomain(
      out_loop,
      /*skip_validation=*/true);
  return;
}

} // namespace scheduler_tools
} // namespace nvfuser

