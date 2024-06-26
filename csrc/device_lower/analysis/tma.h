// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <unordered_map>
#include <variant>
#include <vector>

#include <ir/all_nodes.h>
#include <val_graph.h>

namespace nvfuser {

// See doc/dev/tma.md for design

// All ValGroups are in the traversal graph of tensor indexer

class Box {
 public:
  virtual ~Box() = default;
  virtual Val* boxSize() const = 0;
  virtual Val* tileSize() const = 0;
  virtual Val* elementStride() const = 0;
};

class StridedBox : public Box {
  const ValGroup box;
  const ValGroup tile;
  const ValGroup stride;

 public:
  StridedBox(ValGroup box, ValGroup tile, ValGroup stride)
      : box(std::move(box)), tile(std::move(tile)), stride(std::move(stride)) {}

  Val* boxSize() const override {
    return box->front()->as<IterDomain>()->extent();
  }
  Val* tileSize() const override {
    return tile->front()->as<IterDomain>()->extent();
  }
  Val* elementStride() const override {
    return stride->front()->as<IterDomain>()->extent();
  }
};

class ContiguousBox : public Box {
 public:
  // There is no striding split, so box == tile
  ValGroups box_tile;

  ContiguousBox() = default;
  ContiguousBox(ValGroup g) : box_tile({std::move(g)}) {}

  Val* boxSize() const override {
    Val* size = nullptr;
    for (const auto& g : box_tile) {
      size = SimplifyingIrBuilder::mulExpr(
          size, g->front()->as<IterDomain>()->extent());
    }
    return size;
  }
  Val* tileSize() const override {
    return boxSize();
  }
  Val* elementStride() const override {
    return box_tile.front()->front()->fusion()->oneVal();
  }
};

class ImplicitSizeOneBox : public Box {
  Fusion* const fusion;

 public:
  ImplicitSizeOneBox(Fusion* fusion) : fusion(fusion) {}

  Val* boxSize() const override {
    return fusion->oneVal();
  }
  Val* tileSize() const override {
    return fusion->oneVal();
  }
  Val* elementStride() const override {
    return fusion->oneVal();
  }
};

struct TMADim {
  ValGroups partitioned;
  ValGroups coordinate;
  std::unique_ptr<Box> box;
  Val* gmem_stride_bytes;

  Val* tensorSize() const {
    Val* size = nullptr;
    for (const auto& g : partitioned) {
      size = SimplifyingIrBuilder::mulExpr(
          size, g->front()->as<IterDomain>()->extent());
    }
    return size;
  }
  Val* boxSize() const {
    return box->boxSize();
  }
  Val* tileSize() const {
    return box->tileSize();
  }
  Val* elementStride() const {
    return box->elementStride();
  }
};

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

  Val* tileSizeBytes() const {
    int64_t itemsize = dataTypeSize(gmem_tv_->dtype());
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

} // namespace nvfuser
