// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <exceptions.h>
#include <python_frontend/distributed_tensor.h>
#include <type.h>
#include <utils.h>

namespace nvfuser::python_frontend {

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

} // namespace nvfuser::python_frontend
