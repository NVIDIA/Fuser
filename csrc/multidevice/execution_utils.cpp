// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "multidevice/execution_utils.h"

#include <algorithm>
#include <vector>

#include "exceptions.h"
#include "fusion.h"
#include "multidevice/communicator.h"
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

at::Tensor shardTensor1D(
    at::Tensor tensor,
    const int64_t axis,
    const DeviceMesh& mesh) {
  const auto device_id = Communicator::getInstance().deviceId();
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

at::Tensor shardTensor(at::Tensor tensor, const TensorView* tv) {
  if (!isSharded(tv)) {
    return tensor;
  }

  NVF_ERROR(tv->hasDeviceMesh(), "`tv` has no DeviceMesh: ", tv);
  const DeviceMesh& mesh = tv->getDeviceMesh();

  // This function still assumes the mesh is 1D at this very moment. But the
  // plan is to support multi-dimensional meshes here and leave shardTensor1D
  // for 1D meshes only and eventually deprecated.
  NVF_ERROR_EQ(mesh.rank(), 1);
  return shardTensor1D(
      tensor, getShardedLogicalAxis(tv, ParallelType::DIDx), mesh);
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
        // TODO(#5525): hack for MultiDeviceExecutor.  MultiDeviceExecutor looks
        // for ParallelType::Stream only in logical domains and assumes a
        // stream-parallelized dimension is always fully allocated.  So we set
        // the multiplier to 1 when `sharded_id` is a logical IterDomain. This
        // will have to change when FusionExecutorCache requires a logical
        // dimension to be stream-parallelized, both loop and allocation. Refer
        // to
        // https://github.com/NVIDIA/Fuser/blob/f8e84e52296cdecd318dd2ce904139616d7bd434/tests/cpp/test_overlap.cpp#L155
        // for an example. An alternative to consider is to create a new
        // ParallelType for stream parallelization and use it in
        // FusionExecutorCache.
        if (std::find(
                tv->getLogicalDomain().begin(),
                tv->getLogicalDomain().end(),
                sharded_id) != tv->getLogicalDomain().end()) {
          return 1;
        }

        NVF_ERROR(
            sharded_id->extent()->isConstInt(),
            "DIDs/Stream extent is expected to be constant: ",
            sharded_id);
        return sharded_id->extent()->evaluate().as<int64_t>();
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
