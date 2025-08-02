// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <lru_cache.h>

namespace nvfuser::python {

FusionExecutorCache* LRUCache::cacheCompile(std::shared_ptr<Fusion> fusion) {
  auto it = items_map.find(fusion);

  // short-circuit: Fusion already exists; Get FusionExecutorCache
  if (it != items_map.end()) {
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
           std::make_unique<Fusion>(*fusion), num_fusions_compiled_),
       /*visits=*/0});
  num_fusions_compiled_++;
  // Store the iterator to the new item in the map
  items_map.emplace(fusion, items_list.begin());

  return items_list.front().executor_cache.get();
}

std::string LRUCache::stats() const {
  std::stringstream ss;
  ss << "Total Fusions: " << items_list.size() << "\n";

  // Does not make sense to print stats if the cache is disabled.
  if (!items_list.empty()) {
    ss << "Cache Hits by LRU ordering:\n";
    for (const auto&& [index, item] : enumerate(items_list)) {
      ss << "\t" << index << " -> " << item.visits << " hits\n";
    }

    float hit_rate = static_cast<float>(num_cache_hits_) /
        static_cast<float>(num_cache_lookups_) * 100.0;
    ss << "Cache Lookups: " << num_cache_lookups_ << "\n";
    ss << "Cache Hits: " << num_cache_hits_ << "\n";
    ss << "Hit Rate: " << hit_rate << "%" << "\n";
  }
  return ss.str();
}

void bindLRUCache(py::module_& nvfuser) {
  py::class_<LRUCache>(nvfuser, "LRUCache")
      .def(
          py::init<size_t>(),
          py::arg("max_fusions"),
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
          py::arg("fusion"),
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
          py::return_value_policy::reference)
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
