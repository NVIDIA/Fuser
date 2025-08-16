// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <ir/all_nodes.h>
#include <parallel_type_bitmap.h>
#include <visibility.h>

#include <unordered_map>

namespace nvfuser {

class SyncMap {
 public:
  //! Validates all tensors are consistently parallelized. Basically,
  //! when a producer axis is threaded, either with threadIdx or
  //! blockIdx, there must be a mapped consumer axis with the
  //! same ParallelType with some exceptions.
  //!
  //! Fills needs_raw_sync with output TVs if they need a raw sync if on smem or
  //! gmem. The second entry in this map is the parallel dimensions being
  //! communicated across.
  //!
  //! When error_on_failure is true, tensors requiring RAW sync are
  //! asserted such that they are placed in proper memory spaces.
  NVF_API SyncMap(Fusion* fusion, bool error_on_failure = true);

  std::string toString() const;

  bool needsAnyRawSync(TensorView* tv) const {
    auto it = needs_raw_sync_.find(tv);
    return it != needs_raw_sync_.end() && !it->second.none();
  }

  bool needsBlockRawSync(TensorView* tv) const {
    auto it = needs_raw_sync_.find(tv);
    return it != needs_raw_sync_.end() && it->second.hasTID();
  }

  bool needsGridRawSync(TensorView* tv) const {
    auto it = needs_raw_sync_.find(tv);
    return it != needs_raw_sync_.end() && it->second.hasBID();
  }

  ParallelTypeBitmap getRawSyncParallelTypes(TensorView* tv) const {
    if (auto it = needs_raw_sync_.find(tv); it != needs_raw_sync_.end()) {
      return it->second;
    } else {
      return ParallelTypeBitmap();
    }
  }

  const std::unordered_map<TensorView*, ParallelTypeBitmap>& map() const {
    return needs_raw_sync_;
  }

 private:
  std::unordered_map<TensorView*, ParallelTypeBitmap> needs_raw_sync_;
};

} // namespace nvfuser
