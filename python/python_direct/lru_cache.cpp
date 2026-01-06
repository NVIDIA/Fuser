// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <bindings.h>
#include <lru_cache.h>

namespace nvfuser::python {

FusionExecutorCache* LRUCache::cacheCompile(std::shared_ptr<Fusion> fusion) {
  std::lock_guard<std::mutex> guard(lru_mutex_);
  auto it = items_map.find(fusion);
  num_cache_lookups_++;

  // short-circuit: Fusion already exists; Get FusionExecutorCache
  if (it != items_map.end()) {
    num_cache_hits_++;
    it->second->visits++;
    // Move the item to the front of the list (most recently used)
    items_list.splice(items_list.begin(), items_list, it->second);
    return it->second->executor_cache.get();
  }

  // The fusion is new, check for capacity and evict LRU if necessary.
  if (items_map.size() == max_fusions_) {
    // Evict the least recently used item (the one at the back)
    std::shared_ptr<Fusion> lru_key = items_list.back().fusion;
    items_list.pop_back();
    items_map.erase(lru_key);
  }

  // Insert the new item at the front of the list
  items_list.push_front(
      {fusion,
       std::make_unique<FusionExecutorCache>(
           std::make_unique<Fusion>(*fusion),
           /*fusion_id=*/numFusionsCompiled()),
       /*visits=*/0});
  // Store the iterator to the new item in the map
  items_map.emplace(fusion, items_list.begin());

  return items_list.front().executor_cache.get();
}

std::string LRUCache::stats() const {
  std::lock_guard<std::mutex> guard(lru_mutex_);
  std::stringstream ss;
  ss << "Max Fusions Allowed: " << max_fusions_ << "\n";

  // short-circuit: It is unnecessary to print stats if the cache is empty.
  if (items_list.empty()) {
    ss << "The fusion cache is empty.\n";
    return ss.str();
  }

  ss << "Total Fusions in Cache: " << items_list.size() << "\n";
  ss << "Total Unique Fusions Compiled: " << numFusionsCompiled() << "\n";

  ss << "Cache Hits by LRU ordering:\n";
  for (const auto&& [index, item] : enumerate(items_list)) {
    ss << "\t" << index << " -> " << item.visits << " hits\n";
  }

  float hit_rate = (num_cache_lookups_ == 0)
      ? 0.0f
      : static_cast<float>(num_cache_hits_) /
          static_cast<float>(num_cache_lookups_) * 100.0;
  ss << "Cache Lookups: " << num_cache_lookups_ << "\n";
  ss << "Cache Hits: " << num_cache_hits_ << "\n";
  ss << "Hit Rate: " << hit_rate << "%" << "\n";
  return ss.str();
}

void bindLRUCache(nb::module_& nvfuser) {
  nb::class_<LRUCache>(nvfuser, "LRUCache")
      .def(
          nb::init<size_t>(),
          nb::arg("max_fusions"),
          R"(
Create a new LRUCache.

Parameters
----------
max_fusions : int
    The maximum number of fusions to cache.
)")
      .def(
          "cache_compile",
          &LRUCache::cacheCompile,
          nb::arg("fusion"),
          R"(
Compile a fusion and its executor cache into the cache.

If the fusion is already in the cache, it will be moved to the front of the
cache.

If the cache is full, the least recently used fusion will be evicted.

Parameters
----------
fusion : Fusion
    The fusion to cache.

Returns
------
FusionExecutorCache
    The executor cache for the fusion.
)",
          nb::rv_policy::reference)
      .def(
          "stats",
          &LRUCache::stats,
          R"(
Get stats about the LRU cache.

Returns
------
str
    The stats about the LRU cache.
)")
      .def(
          "num_fusions",
          &LRUCache::numFusions,
          R"(
Get the number of fusions in the LRU cache.

Returns
------
int
    The number of fusions in the LRU cache.
)");
}

} // namespace nvfuser::python
