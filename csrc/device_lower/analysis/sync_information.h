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
  //! ComputeAtMap is already built as they are used to validate consistency.
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

  const std::unordered_map<
      TensorView*,
      std::unordered_map<TensorView*, ParallelTypeBitmap>>&
  producerConsumerRawSync() const {
    return producer_consumer_raw_sync_;
  }

 private:
  // RAW dependency parallel types of each tensor
  std::unordered_map<TensorView*, ParallelTypeBitmap> needs_raw_sync_;

  // Mappings of per-consumer dependecy parallel types. Maps from a
  // tensor to a consumer tensor and its RAW
  // dependency parallel types with respect to the consumer
  // tensor. Aggregating the parallel types of all consumers yields
  // the same parallel types as needs_raw_sync_
  std::unordered_map<
      TensorView*,
      std::unordered_map<TensorView*, ParallelTypeBitmap>>
      producer_consumer_raw_sync_;
};

} // namespace nvfuser
