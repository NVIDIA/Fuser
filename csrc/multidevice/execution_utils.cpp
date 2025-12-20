// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "multidevice/execution_utils.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "exceptions.h"
#include "fusion.h"
#include "multidevice/device_mesh.h"
#include "multidevice/utils.h"

namespace nvfuser {

int64_t requestedNumberOfDevices(Fusion* fusion) {
  DeviceIdxType max_index = 0;
  for (auto tv : fusion->allTvs()) {
    if (tv->hasDeviceMesh()) {
      max_index = std::max(max_index, tv->getDeviceMesh().maxDeviceId());
    }
  }
  return max_index + 1;
}

at::Tensor shardTensor(
    at::Tensor tensor,
    const int64_t axis,
    const DeviceMesh& mesh,
    const DeviceIdxType device_id) {
  auto i = mesh.linearIndexOf(device_id);
  auto extent = tensor.size(axis);
  auto nslices = mesh.size();
  NVF_CHECK(
      extent % nslices == 0, "Sharded axis must be evenly divisble by mesh");
  auto stride = extent / nslices;
  // TODO: returning slice 0 temporarily when device is not in the mesh.
  i = (i < 0) ? 0 : i;
  // The following slicing is problematic when DID is on an inner split (cf.
  // MultiDeviceTest.ShardTensor_InnerSplit). We currently disallow that and
  // it's enforced by getShardedLogicalAxis.
  return tensor.slice(axis, i * stride, (i + 1) * stride).contiguous();
}

std::vector<int64_t> unshardedSizes(
    const TensorView* tv,
    c10::IntArrayRef sizes) {
  std::vector<int64_t> unsharded_sizes = sizes.vec();
  for (ParallelType parallel_type : deviceAndStreamParallelTypes()) {
    const DomainType domain_type = parallel_type == ParallelType::Stream
        ? DomainType::kAllocation
        : DomainType::kLoop;
    IterDomain* sharded_id =
        getShardedIterDomain(tv, parallel_type, domain_type);
    if (sharded_id == nullptr) {
      continue;
    }

    const int64_t sharded_axis = getShardedLogicalAxis(tv, parallel_type);
    NVF_ERROR(
        sharded_axis != -1,
        "Producing logical axis not found for ",
        sharded_id);

    auto multiplier = [&]() -> int64_t {
      if (parallel_type == ParallelType::Stream) {
        return 1;
      }

      if (isParallelTypeDeviceDim(parallel_type)) {
        return tv->getDeviceMesh().size(parallel_type);
      }

      NVF_THROW("Unexpected parallel type: ", parallel_type);
    }();
    unsharded_sizes.at(sharded_axis) *= multiplier;
  }

  return unsharded_sizes;
}

} // namespace nvfuser
