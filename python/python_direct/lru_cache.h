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

#include <iostream>
#include <list>
#include <optional>
#include <unordered_map>

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
  void put(
      std::shared_ptr<Fusion> key,
      std::shared_ptr<FusionExecutorCache> value) {
    auto it = items_map.find(key);

    // Key already exists
    if (it != items_map.end()) {
      // Update the value
      it->second->value = value;
      // Move the item to the front of the list (most recently used)
      items_list.splice(items_list.begin(), items_list, it->second);
      return;
    }

    // Key is new, check for capacity
    if (items_map.size() == max_fusions_) {
      // Evict the least recently used item (the one at the back)
      std::shared_ptr<Fusion> lru_key = items_list.back().key;
      items_list.pop_back();
      items_map.erase(lru_key);
    }

    // Insert the new item at the front of the list
    items_list.push_front({key, value, /*visits=*/0});
    // Store the iterator to the new item in the map
    items_map.emplace(key, items_list.begin());
  }

  // Get a value from the cache by its key.
  std::shared_ptr<FusionExecutorCache> get(std::shared_ptr<Fusion> key) {
    // Increment the overall visits count
    num_cache_lookups_++;

    // Find the item in the map
    auto it = items_map.find(key);

    // Key not found
    if (it == items_map.end()) {
      return nullptr;
    }

    // Increment the number of total cache hits
    num_cache_hits_++;

    // Increment the visits count per key
    it->second->visits++;

    // Key found, move its item to the front of the list
    items_list.splice(items_list.begin(), items_list, it->second);

    // Return the value
    return it->second->value;
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
    std::shared_ptr<Fusion> key;
    std::shared_ptr<FusionExecutorCache> value;
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

  // Number of visits to the cache
  size_t num_cache_lookups_ = 0;

  // Number of cache hits
  size_t num_cache_hits_ = 0;
};

} // namespace nvfuser::python
