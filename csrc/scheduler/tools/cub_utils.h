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

// Utility class to compute the size of the shared memory buffer used
// by CUB for operations like argsort
class CubSharedMemoryBuffer {
 public:
  void registerArgsort(int64_t bdimx, int64_t items_per_thread, DataType dtype);

  void registerScan(int64_t bdimx, int64_t items_per_thread, DataType dtype);

  void registerTopK(int64_t bdimx, int64_t items_per_thread, DataType dtype);

  int64_t getTotalSizeInBytes() const;

  int64_t getArgsortTotalSizeInBytes() const;

  int64_t getScanTotalSizeInBytes() const;

  int64_t getTopKTotalSizeInBytes() const;

 private:
  // Parameters affecting the buffer size of each call using block
  // radix sort. bdimx is common across all calls, so not included
  // here.
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

  // Parameters affecting the buffer size of each call using block
  // scan. bdimx is common across all calls, so not included
  // here.
  struct BlockScanParameters {
    DataType dtype;

    bool operator==(const BlockScanParameters& other) const {
      return dtype == other.dtype;
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

 private:
  int64_t max_bdimx_ = -1;

  // Keep track of argsort calls to compute the total size of the
  // shared memory buffers used for argsort
  std::unordered_set<BlockRadixSortParameters, BlockRadixSortParametersHash>
      argsort_calls_;

  // Keep track of scan calls to compute the total size of the
  // shared memory buffers used for scan. Note that each call seems to
  // be considered a distinctive separate call due to the lambda
  // parameter, and thus there's no reuse even for the same data
  // type. This should be fixed by using dynamically allocated buffers.
  std::vector<BlockScanParameters> scan_calls_;

  // Keep track of topk calls to compute the total size of the
  // shared memory buffers used for topk.
  std::unordered_set<BlockRadixSortParameters, BlockRadixSortParametersHash>
      topk_calls_;
};

} // namespace scheduler_tools
} // namespace nvfuser
