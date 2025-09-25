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
#include <variant>
#include <vector>

#include <ir/all_nodes.h>
#include <val_graph.h>

namespace nvfuser {

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

} // namespace nvfuser
