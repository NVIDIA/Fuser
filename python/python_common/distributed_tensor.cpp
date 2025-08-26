// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <distributed_tensor.h>
#include <exceptions.h>
#include <ir/interface_nodes.h>
#include <type.h>
#include <utils.h>

namespace nvfuser {

void Sharding::setAxisIsShardedOn(
    const int64_t axis,
    const ParallelType parallel_type) {
  NVF_CHECK(isParallelTypeDeviceDim(parallel_type));
  NVF_CHECK(mesh_.size() > 0, "Cannot shard a non-distributed tensor.");
  const auto i = axis_sharded_on_.find(parallel_type);
  NVF_CHECK(
      i == axis_sharded_on_.end(),
      "Parallel type ",
      parallel_type,
      " was already used to shard axis ",
      i->second);
  axis_sharded_on_[parallel_type] = axis;
}

int64_t Sharding::axisShardedOn(const ParallelType parallel_type) const {
  return getOrDefault(axis_sharded_on_, parallel_type, -1L);
}

std::vector<Sharding> getOutputShardings(Fusion* fusion) {
  std::vector<TensorView*> all_tvs = fusion->allTvs();
  if (std::none_of(
          all_tvs.begin(),
          all_tvs.end(),
          std::mem_fn(&TensorView::hasDeviceMesh))) {
    return {};
  }

  std::vector<Sharding> output_shardings;
  output_shardings.reserve(fusion->outputs().size());
  for (Val* out_val : fusion->outputs()) {
    if (auto* out_tv = dynamic_cast<TensorView*>(out_val)) {
      if (fusion->getOutputAlias(out_tv).hide_output) {
        continue;
      }
      const DeviceMesh& mesh = out_tv->getDeviceMesh();
      Sharding& output_sharding = output_shardings.emplace_back(mesh);
      if (mesh.size() > 0) {
        for (const ParallelType parallel_type : kParallelTypeDIDs) {
          if (const auto axis = getShardedLogicalAxis(out_tv, parallel_type);
              axis != -1) {
            output_sharding.setAxisIsShardedOn(axis, parallel_type);
          }
        }
      }
    } else {
      output_shardings.emplace_back(DeviceMesh());
    }
  }

  return output_shardings;
}

} // namespace nvfuser
