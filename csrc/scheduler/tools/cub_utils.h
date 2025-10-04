// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <type.h>

namespace nvfuser {
namespace scheduler_tools {

class CubSharedMemoryBuffer {
 public:
  // bdimx is common across all calls, so not included here
  struct ArgsortTemplateParameters {
    int64_t items_per_thread;
    DataType dtype;

    bool operator==(const ArgsortTemplateParameters& other) const {
      return items_per_thread == other.items_per_thread && dtype == other.dtype;
    }
  };

  struct ArgsortTemplateParametersHash {
    std::size_t operator()(const ArgsortTemplateParameters& key) const {
      return std::hash<int64_t>()(key.items_per_thread);
    }
  };

  void registerArgsort(int64_t bdimx, int64_t items_per_thread, DataType dtype);

  int64_t getTotalSizeInBytes() const;

 private:
  int64_t getArgsortTotalSizeInBytes() const;

 private:
  int64_t max_bdimx_ = -1;
  std::unordered_set<ArgsortTemplateParameters, ArgsortTemplateParametersHash>
      argsort_calls_;
};

} // namespace scheduler_tools
} // namespace nvfuser
