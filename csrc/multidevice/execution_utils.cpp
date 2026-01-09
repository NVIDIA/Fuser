// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "multidevice/execution_utils.h"

#include <algorithm>
#include <numeric>
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

at::Tensor shardTensor(
    at::Tensor tensor,
    const std::vector<int64_t>& tensor_axes,
    const std::vector<int64_t>& mesh_axes,
    const DeviceMesh& mesh,
    const DeviceIdxType device_id) {
  NVF_CHECK(
      tensor_axes.size() == mesh_axes.size(),
      "tensor_axes and mesh_axes must have the same size. Got ",
      tensor_axes.size(),
      " and ",
      mesh_axes.size());

  // Get the multi-dimensional index of the device in the mesh
  at::Tensor device_index = mesh.multiDimensionalIndexOf(device_id);
  NVF_CHECK(
      device_index.defined(), "Device ", device_id, " is not in mesh ", mesh);

  at::Tensor result = tensor;

  // Shard along each tensor axis according to its corresponding mesh axis
  // We need to track axis shifts because slicing reduces dimensions
  std::vector<int64_t> axis_shifts(tensor.dim(), 0);

  // Sort by tensor_axes to handle negative indexing and slicing in order
  std::vector<size_t> indices(tensor_axes.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
    int64_t axis_a =
        tensor_axes[a] >= 0 ? tensor_axes[a] : tensor.dim() + tensor_axes[a];
    int64_t axis_b =
        tensor_axes[b] >= 0 ? tensor_axes[b] : tensor.dim() + tensor_axes[b];
    return axis_a < axis_b;
  });

  for (size_t idx : indices) {
    int64_t tensor_axis = tensor_axes[idx];
    int64_t mesh_axis = mesh_axes[idx];

    // Normalize negative axes
    if (tensor_axis < 0) {
      tensor_axis += tensor.dim();
    }
    if (mesh_axis < 0) {
      mesh_axis += mesh.rank();
    }

    NVF_CHECK(
        tensor_axis >= 0 && tensor_axis < result.dim(),
        "tensor_axis ",
        tensor_axes[idx],
        " is out of bounds for tensor with ",
        result.dim(),
        " dimensions");

    NVF_CHECK(
        mesh_axis >= 0 && mesh_axis < mesh.rank(),
        "mesh_axis ",
        mesh_axes[idx],
        " is out of bounds for mesh with rank ",
        mesh.rank());

    // Get the coordinate of this device along the mesh axis
    int64_t mesh_coord = device_index[mesh_axis].item<int64_t>();

    // Get the size of the mesh along this axis
    int64_t mesh_size = mesh.size(mesh_axis);

    // Calculate the slice for this axis
    int64_t extent = result.size(tensor_axis);
    NVF_CHECK(
        extent % mesh_size == 0,
        "Tensor axis ",
        tensor_axes[idx],
        " with size ",
        extent,
        " must be evenly divisible by mesh axis ",
        mesh_axes[idx],
        " with size ",
        mesh_size);

    int64_t stride = extent / mesh_size;
    int64_t start = mesh_coord * stride;
    int64_t end = (mesh_coord + 1) * stride;

    result = result.slice(tensor_axis, start, end);
  }

  return result.contiguous();
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
