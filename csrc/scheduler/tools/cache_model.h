// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cstdint>
#include <list>
#include <unordered_map>

namespace nvfuser {

namespace scheduler_tools {

//! This class models an LRU cache where all of the requests are
//! non-overlapping. That is, requests come with an address and a size and the
//! ranges requested between any two requests either are assumed to match
//! entirely or to not overlap at all.
//!
//!
class NonOverlappingLRUCacheModel {
 public:
  NonOverlappingLRUCacheModel(int64_t capacity) : capacity_(capacity) {}

  //! Either a read or a write. Returns true if this is a hit
  bool access(int64_t address, int64_t size);

  //! Accessors
  int64_t capacity() const {
    return capacity_;
  }

  int64_t allocated() const {
    return allocated_;
  }

  int64_t missedBytes() const {
    return bytes_missed_;
  }

  int64_t hitBytes() const {
    return bytes_hit_;
  }

 private:
  //! Remove least recently used entries until allocated_ is within capacity_
  void evict();

 private:
  int64_t capacity_;

  struct Entry {
    int64_t address;
    int64_t size;
  };
  // The head of this list is the most recently used address and the tail is the
  // least recently used. When we do a write or read, we remove any pre-existing
  // entries for that address using lookup_, add an entry to the head, and then
  // remove tail entries until the allocated_ is within the capacity once
  // again.
  std::list<Entry> priority_;
  // Maps from an address to an Entry in the list. There should always be the
  // same number of entries here as the size of priority_
  std::unordered_map<int64_t, std::list<Entry>::const_iterator> lookup_;
  // This should always be equal to the sum of the sizes of all entries in
  // priority_
  int64_t allocated_ = 0L;

  int64_t bytes_missed_ = 0L;
  int64_t bytes_hit_ = 0L;
};

} // namespace scheduler_tools
} // namespace nvfuser
