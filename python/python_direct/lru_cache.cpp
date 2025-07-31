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
