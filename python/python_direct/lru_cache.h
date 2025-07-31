// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <runtime/fusion_executor_cache.h>
#include <runtime/fusion_kernel_runtime.h>

#include <iostream>
#include <list>
#include <optional>
#include <unordered_map>

class FusionKernelRuntime; // Forward declaration

namespace nvfuser::python {

class LRUCache {
 public:
  // Constructor
  explicit LRUCache(size_t max_fusions) : max_fusions_(max_fusions) {}
  //! Copy and Assignment of the LRUCache is not supported
  //! clang-tidy: deleted member function should be public
  LRUCache(const LRUCache&) = delete;
  LRUCache& operator=(const LRUCache&) = delete;

  // Put a key-value pair into the cache.
  FusionExecutorCache* cacheCompile(std::shared_ptr<Fusion> fusion) {
    auto it = items_map.find(fusion);

    // Key already exists
    if (it != items_map.end()) {
      // Move the item to the front of the list (most recently used)
      items_list.splice(items_list.begin(), items_list, it->second);
      return it->second->executor_cache.get();
    }

    // Key is new, check for capacity
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

  // Print stats about LRU Cache
  std::string stats() const {
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

  // Number of fusions cached
  size_t numFusions() const {
    return items_list.size();
  }

  // TODO Serialize LRU cache using flatbuffers
  // void serialize(std::string filename) const;

  // TODO Deserialize LRU Cache using flatbuffers
  // void deserialize(std::string filename);

 private:
  // Item is a tuple of key, value, and visits per key
  struct Item {
    std::shared_ptr<Fusion> fusion;
    std::unique_ptr<FusionExecutorCache> executor_cache;
    size_t visits;
  };

  // Custom Hasher Functor for Fusion
  struct FusionHasher {
    size_t operator()(const std::shared_ptr<Fusion>& fusion) const {
      return fusion->hash();
    }
  };

  // Custom Equality Functor for Fusion
  struct FusionEqualTo {
    bool operator()(
        const std::shared_ptr<Fusion>& lhs,
        const std::shared_ptr<Fusion>& rhs) const {
      return lhs->checkDefinition(*rhs);
    }
  };

  // The list stores key-value pairs to maintain the order of use.
  // Front of the list is the most recently used.
  std::list<Item> items_list;

  // TODO Replace std::unordered_map with absl::flat_hash_map
  // Reference: Use https://abseil.io/docs/cpp/guides/container
  // The map provides O(1) access to list nodes.
  // It stores keys and iterators to the corresponding pairs in the list.
  std::unordered_map<
      std::shared_ptr<Fusion>,
      typename std::list<Item>::iterator,
      FusionHasher,
      FusionEqualTo>
      items_map;

  //! The max allowed number of fusions in the cache
  size_t max_fusions_;

  // Cummulative number of fusions compiled
  size_t num_fusions_compiled_ = 0;

  // Number of visits to the cache
  size_t num_cache_lookups_ = 0;

  // Number of cache hits
  size_t num_cache_hits_ = 0;
};

} // namespace nvfuser::python
