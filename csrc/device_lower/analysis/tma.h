// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ostream>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <ir/all_nodes.h>
#include <val_graph.h>

namespace nvfuser {

namespace kir {
class TensorIndex;
} // namespace kir

// See doc/dev/tma.md for design

// All ValGroups are in the traversal graph of tensor indexer

struct TMADim {
  ValGroup partitioned;
  ValGroup box;
  ValGroup tile;
  ValGroup stride;
  Val* gmem_stride_bytes;

  Val* tensorSize() const {
    return partitioned->front()->as<IterDomain>()->extent();
  }
  Val* boxSize() const {
    return box ? box->front()->as<IterDomain>()->extent()
               : gmem_stride_bytes->fusion()->oneVal();
  }
  Val* tileSize() const {
    return tile ? tile->front()->as<IterDomain>()->extent()
                : gmem_stride_bytes->fusion()->oneVal();
  }
  Val* elementStride() const {
    return stride ? stride->front()->as<IterDomain>()->extent()
                  : gmem_stride_bytes->fusion()->oneVal();
  }
};

std::ostream& operator<<(std::ostream& os, const TMADim& d);

class TMAInfo {
  std::vector<TMADim> dims_;
  MmaInputSmemSwizzle swizzle_;
  TensorView* gmem_tv_;

 public:
  TMAInfo(
      std::vector<TMADim> dims,
      MmaInputSmemSwizzle swizzle,
      TensorView* gmem_tv)
      : dims_(std::move(dims)), swizzle_(swizzle), gmem_tv_(gmem_tv) {}

  const std::vector<TMADim>& dims() const {
    return dims_;
  }

  MmaInputSmemSwizzle swizzle() const {
    return swizzle_;
  }

  std::vector<ValGroup> getTMADomain() const {
    std::vector<ValGroup> result;
    std::transform(
        dims_.begin(),
        dims_.end(),
        std::back_inserter(result),
        [](const auto& d) { return d.partitioned; });
    return result;
  }

  Val* tileSizeBytes() const {
    int64_t itemsize = dataTypeSizeByte(gmem_tv_->dtype());
    Val* size = IrBuilder::create<Val>(itemsize, DataType::Index);
    for (const auto& d : dims_) {
      size = SimplifyingIrBuilder::mulExpr(size, d.tileSize());
    }
    return size;
  }

  Val* tensorMap() const;
};

std::unordered_map<TensorView*, const TMAInfo> getConsumerToTMAInfoMap(
    Fusion* fusion);

MmaInputSmemSwizzle getSwizzle(TensorView* tv);

// Contains information about batched non-circular-buffered TMA loads.
// This is populated during analysis and consumed by later passes
// (e.g. allocation / indexing / sync insertion).
//
// A TMA (Tensor Memory Accelerator) load is considered "batchable" if it meets
// all of the following criteria:
//
//  1. It is a CpAsyncBulk load operation (not circular-buffered)
//  2. Block dim X has at least 32 threads (required for elect sync)
//  3. The loaded TensorView has no thread-parallelized or serial dimensions
//
// When multiple batchable TMA loads exist (> 1) and they are all parallelized
// in the same way, the "batched TMA path" is used, which:
// - Allocates a shared array of mbarriers (one per TMA load)
// - Uses indexed mbarrier operations for synchronization
// - Waits for all TMA loads together after the last load completes
//
// This is more efficient than per-load mbarrier allocation when multiple
// non-circular-buffered TMA loads exist in the kernel.
//
class BatchedTmaInfo {
 public:
  explicit BatchedTmaInfo(Fusion* fusion);

  // Returns the set of batchable TMA load expressions
  const std::unordered_set<const Expr*>& batchableLoads() const {
    return batchable_tma_loads_;
  }

  // Returns the number of batchable TMA loads computed during construction
  int64_t numBatchableLoads() const {
    return batchable_tma_loads_.size();
  }

 private:
  std::unordered_set<const Expr*> batchable_tma_loads_;
};

} // namespace nvfuser
