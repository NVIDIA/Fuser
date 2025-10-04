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
  struct BlockRadixSortParameters {
    int64_t items_per_thread;
    DataType dtype;

    bool operator==(const BlockRadixSortParameters& other) const {
      return items_per_thread == other.items_per_thread && dtype == other.dtype;
    }
  };

  struct BlockRadixSortParametersHash {
    std::size_t operator()(const BlockRadixSortParameters& key) const {
      return std::hash<int64_t>()(key.items_per_thread);
    }
  };

  // bdimx is common across all calls, so not included here
  struct BlockScanParameters {
    DataType dtype;
    // TODO: Remove this once reuse is implemented
    int64_t index;

    bool operator==(const BlockScanParameters& other) const {
      return dtype == other.dtype &&
          index == other.index;
    }
  };

  struct BlockScanParametersHash {
    std::size_t operator()(const BlockScanParameters& key) const {
      if (auto prim_type = std::get_if<PrimDataType>(&key.dtype.type)) {
        return static_cast<std::size_t>(*prim_type) + 1;
      } else {
        return 0;
      }
    }
  };

  void registerArgsort(int64_t bdimx, int64_t items_per_thread, DataType dtype);

  void registerScan(int64_t bdimx, int64_t items_per_thread, DataType dtype);

  void registerTopK(int64_t bdimx, int64_t items_per_thread, DataType dtype);

  int64_t getTotalSizeInBytes() const;

 private:
  int64_t getArgsortTotalSizeInBytes() const;

  int64_t getScanTotalSizeInBytes() const;

  int64_t getTopKTotalSizeInBytes() const;

 private:
  int64_t max_bdimx_ = -1;
  std::unordered_set<BlockRadixSortParameters, BlockRadixSortParametersHash>
      argsort_calls_;
  int64_t scan_index_ = 0;
  std::unordered_set<BlockScanParameters, BlockScanParametersHash> scan_calls_;
  std::unordered_set<BlockRadixSortParameters, BlockRadixSortParametersHash>
      topk_calls_;
};

} // namespace scheduler_tools
} // namespace nvfuser
