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
#include "expr_evaluator.h"
#include "fusion.h"
#include "linked_hash_map.h"
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
  NVF_CHECK(tv->hasDeviceMesh(), "`tv` has no DeviceMesh: ", tv);
  const DeviceMesh& mesh = tv->getDeviceMesh();

  // The following is similar to
  // [transformFromAllocationToLogical](https://github.com/NVIDIA/Fuser/blob/538ea84fe75c8b516114a46ac159e888ec8ac684/csrc/runtime/allocations.cpp#L771).
  // I didn't reuse the code yet because
  // 1.
  // https://github.com/NVIDIA/Fuser/blob/538ea84fe75c8b516114a46ac159e888ec8ac684/csrc/runtime/allocations.cpp#L902
  // suggests the code may want to move away from manipulating tensors.
  // 2. I'm not comfortable with how merges are handled. See code comments
  // below.
  // 3. I prefer using LinkedHashMap, avoiding some linear scans.

  // Ignore reduction dimensions because they won't appear in tensor.sizes().
  auto source = tv->getLogicalDomain() | TensorDomain::kNoReductions;
  NVF_CHECK(
      std::ranges::none_of(
          tv->getMaybeAllocationDomain(), std::mem_fn(&IterDomain::isStream)),
      "shardTensor is expected to be called in tests on the complete fusion's "
      "inputs and outputs, whose allocation is not stream-parallelized: ",
      toDelimitedString(tv->getMaybeAllocationDomain()));
  auto target = tv->getLoopDomain() |
      std::views::filter(std::mem_fn(&IterDomain::isDeviceDim)) |
      TensorDomain::kNoReductions;
  std::vector<Expr*> transforms = DependencyCheck::getAllExprsBetween(
      {source.begin(), source.end()}, {target.begin(), target.end()});

  // Bind the logical domain.
  ExpressionEvaluator evaluator;
  LinkedHashMap<IterDomain*, int64_t> id_to_size;
  NVF_CHECK_EQ(std::ranges::distance(source), tensor.dim());
  for (auto [id, size] : zip(source, tensor.sizes())) {
    evaluator.bind(id->getMaybeExpandedExtent(), size);
    id_to_size.pushBack(id, size);
  }

  // Traverse down to view `tensor` as the loop domain.
  for (Expr* transform : transforms) {
    auto* split = dynamic_cast<Split*>(transform);
    // For example, tv =
    //   [4, 3]
    //    \ /
    //     12
    //    / \.
    //   3   4
    //      / \.
    //     2   2(DIDx)
    //
    // tensor = [[0, 1, 2],
    //           [3, 4, 5],
    //           [6, 7, 8],
    //           [9, 10, 11]]
    //
    // GPU 0 is supposed to hold values 0, 2, 4, 6, 8, 10, and GPU 1 is supposed
    // to hold values 1, 3, 5, 7, 9, 11. But in what shape?
    NVF_CHECK(
        split != nullptr,
        "Stay simple for now. It's tricky to support merges (see code "
        "comments). I don't expect to see merges between **logical** (not "
        "root) and loop in the foreseeable future: ",
        transform->toString());

    auto [in_size, i] = id_to_size.erase(split->in());
    const auto outer_size =
        evaluator.evaluate(split->outer()->getMaybeExpandedExtent())
            .as<int64_t>();
    id_to_size.insert(i, split->outer(), outer_size);
    const auto inner_size =
        evaluator.evaluate(split->inner()->getMaybeExpandedExtent())
            .as<int64_t>();
    id_to_size.insert(i, split->inner(), inner_size);

    std::vector<int64_t> new_sizes;
    new_sizes.reserve(id_to_size.size());
    for (auto size : id_to_size | std::views::values) {
      new_sizes.push_back(size);
    }
    tensor = tensor.reshape(new_sizes);
  }

  // Slice the tensor so it contains data only for the current GPU.
  {
    const auto device_id = Communicator::getInstance().deviceId();
    at::Tensor md_index = mesh.multiDimensionalIndexOf(device_id);
    if (!md_index.defined()) {
      // If the device is not in the mesh, return the first slice.
      md_index = at::zeros({mesh.rank()});
    }
    int64_t axis = 0;
    for (auto [id, size] : id_to_size) {
      if (id->isParallelized()) {
        auto mesh_size = mesh.size(id->getParallelType());
        NVF_ERROR_EQ(size, mesh_size);
        auto mesh_axis = mesh.parallelTypeToAxis(id->getParallelType());
        auto index = md_index[mesh_axis].item<int64_t>();
        tensor = tensor.slice(axis, index, index + 1);
      }
      axis++;
    }
  }

  // Traverse up to view the sharded tensor as the logical domain.
  {
    LinkedHashMap<IterDomain*, std::monostate> ids;
    for (auto [id, _] : id_to_size) {
      ids.pushBack(id, std::monostate());
    }

    for (Expr* transform : transforms | std::views::reverse) {
      auto* split = dynamic_cast<Split*>(transform);
      NVF_ERROR(split != nullptr);

      auto i = ids.erase(split->outer()).second;
      auto axis = std::ranges::distance(ids.begin(), i);
      NVF_ERROR(i != ids.end() && i->first == split->inner());
      i = ids.erase(split->inner()).second;
      ids.insert(i, split->in(), std::monostate());

      tensor = tensor.flatten(axis, axis + 1);
    }
  }

  // Make sure the returned tensor is at least as contiguous as the original
  // tensor.  This is likely an overkill but probably OK for now because all
  // multi-GPU tests create contiguous test tensors.
  return tensor.contiguous();
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
