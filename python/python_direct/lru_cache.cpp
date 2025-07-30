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
          "put",
          &LRUCache::put,
          py::arg("fusion"),
          py::arg("fusion_executor_cache"),
          R"(
Put a fusion and its executor cache into the cache.

Parameters
----------
fusion : Fusion
    The fusion to cache.

fusion_executor_cache : FusionExecutorCache
    The executor cache to cache.

Returns
------
None
)")
      .def(
          "get",
          &LRUCache::get,
          py::arg("fusion"),
          R"(
Get executor cache for a fusion.

Parameters
----------
fusion : Fusion
    The fusion to get.

Returns
------
FusionExecutorCache
    The executor cache for the fusion.
)")
      .def(
          "stats",
          &LRUCache::stats,
          R"(
Print stats about the LRU cache.

Returns
------
None
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
