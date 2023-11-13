// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <list>
#include <unordered_map>
#include <utility>

#include <exceptions.h>

namespace nvfuser {

// This mimics
// https://docs.oracle.com/javase/8/docs/api/java/util/LinkedHashMap.html. The
// implementation is a combination of a linked list of <K,V> and a hash map from
// K to a position in the linked list.
template <typename K, typename V>
class LinkedHashMap {
 public:
  using value_type = std::pair<K, V>;
  using const_iterator = typename std::list<value_type>::const_iterator;

  LinkedHashMap() = default;
  LinkedHashMap(const LinkedHashMap&) = delete;
  LinkedHashMap& operator=(const LinkedHashMap&) = delete;
  LinkedHashMap(LinkedHashMap&&) = default;
  LinkedHashMap& operator=(LinkedHashMap&&) = default;

  std::pair<V, const_iterator> erase(const K& key);

  void insert(const_iterator i, const K& key, const V& value);

  void pushBack(const K& key, const V& value);

  const_iterator begin() const {
    return order_.begin();
  }
  const_iterator end() const {
    return order_.end();
  }

 private:
  std::list<value_type> order_;
  std::unordered_map<K, const_iterator> key_to_index_;
};

template <typename K, typename V>
std::pair<V, typename LinkedHashMap<K, V>::const_iterator> LinkedHashMap<K, V>::
    erase(const K& key) {
  const_iterator index = key_to_index_.at(key);
  key_to_index_.erase(key);
  return {index->second, order_.erase(index)};
}

template <typename K, typename V>
void LinkedHashMap<K, V>::insert(
    LinkedHashMap<K, V>::const_iterator i,
    const K& key,
    const V& value) {
  bool inserted =
      key_to_index_.emplace(key, order_.insert(i, {key, value})).second;
  NVF_CHECK(inserted, "Key already existed");
}

template <typename K, typename V>
void LinkedHashMap<K, V>::pushBack(const K& key, const V& value) {
  insert(order_.end(), key, value);
}

} // namespace nvfuser
