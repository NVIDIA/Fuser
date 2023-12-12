// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <multidevice/utils.h>

#include <c10/util/irange.h>

namespace nvfuser {

bool isSharded(TensorView* tv) {
  std::vector<bool> is_sharded;
  for (IterDomain* id : TensorDomain::noReductions(tv->getLeafDomain())) {
    is_sharded.push_back(id->isDeviceDim());
  }
  // Currently, only the most external dim is allowed to be sharded
  NVF_ERROR(tv->getMaybeRFactorDomain() == tv->getLeafDomain());
  for (auto i : c10::irange(1, is_sharded.size())) {
    NVF_ERROR(
        !is_sharded.at(i),
        "only the outmost dimension can be device-parallelized");
  }
  return is_sharded.empty() ? false : is_sharded.at(0);
}

int64_t requestedNumberOfDevices(Fusion* fusion) {
  std::set<DeviceIdxType> device_indices;
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->vals())) {
    if (tv->hasDeviceMesh()) {
      std::copy(
          tv->getDeviceMesh()->vector().begin(),
          tv->getDeviceMesh()->vector().end(),
          std::inserter(device_indices, device_indices.begin()));
    }
  }
  return static_cast<int64_t>(device_indices.size());
}

} // namespace nvfuser
