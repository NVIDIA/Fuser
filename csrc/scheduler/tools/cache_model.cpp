// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <exceptions.h>
#include <scheduler/tools/cache_model.h>

#include <ATen/cuda/CUDAContextLight.h>

namespace nvfuser {

namespace scheduler_tools {

bool NonOverlappingLRUCacheModel::access(int64_t address, int64_t size) {
  const auto it = lookup_.find(address);
  bool hit = it != lookup_.end();
  if (hit) {
    NVF_ERROR(it->second->address == address);
    NVF_ERROR(
        it->second->size == size,
        "Received cache hit for address ",
        address,
        " but new size ",
        size,
        " does not match cache entry size ",
        it->second->size);
    priority_.erase(it->second);
  } else {
    // miss
    allocated_ += size;
    evict();
  }
  priority_.push_front({address, size});
  lookup_[address] = priority_.cbegin();
  return hit;
}

void NonOverlappingLRUCacheModel::evict() {
  while (allocated_ > capacity_) {
    const Entry& entry = priority_.back();
    allocated_ -= entry.size;
    lookup_.erase(entry.address);
    priority_.pop_back();
  }
}

} // namespace scheduler_tools
} // namespace nvfuser
