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
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>

namespace nvfuser {

IterDomain* getCircularBufferAxis(const TensorView* tv);

void validateCircularBufferedTensor(const TensorView* tv);

class CircularBufferInfo {
  // Lowering information of circular buffered tensors.
  struct TvInfo {
    IterDomain* circular_buffer_axis = nullptr;
    Val* original_alloc_size = nullptr;
  };

 public:
  void build(Fusion* fusion);

  void setCircularBufferTv(const TensorView* tv);

  IterDomain* getCircularBufferAxis(const TensorView* tv) const;

  //! Get all valid circular buffer TensorViews
  std::vector<const TensorView*> getCircularBufferTvs() const;

  //! Get a loop that matches with a given circular-buffer axis. If
  //! ignore_prologue is true, a matched loop is ignored if it's a
  //! prologue loop.
  static ForLoop* getCircularBufferLoop(
      IterDomain* axis,
      const std::vector<ForLoop*>& loops,
      bool ignore_prologue = false);

  //! Get a loop that matches with the circular-buffer axis of a given
  //! circular-buffered tensor. If ignore_prologue is true, a matched
  //! loop is ignored if it's a prologue loop.
  ForLoop* getCircularBufferLoop(
      const TensorView* tv,
      const std::vector<ForLoop*>& loops,
      bool ignore_prologue = false);

  //! Get the circular-buffered tensors for the given loop/axis.
  std::unordered_set<const TensorView*> getCircularBufferTvs(
      ForLoop* axis) const;
  std::unordered_set<const TensorView*> getCircularBufferTvs(
      IterDomain* axis) const;

  void setOriginalAllocSize(const TensorView* tv, Val* size);

  Val* getOriginalAllocSize(const TensorView* tv);

  //! Returns true if the iterdomain will be realized
  //!  as a circular buffer loop.
  bool isCircularBufferedIterDomain(IterDomain* id);

  //! Get the number of circular buffer stages and the prefetch distance for the
  //! given axis. The number of stages will be 2 in the case of double buffer
  //! loop.
  int64_t getStageDepthFor(IterDomain* circular_buffered_id) const;
  int64_t getPrefetchDistanceFor(IterDomain* circular_buffered_id) const;

  std::string toString() const;

 private:
  const TvInfo& getTvInfo(const TensorView* tv) const;

  TvInfo& getTvInfo(const TensorView* tv);

  //! Set the number of circular buffer stages and the prefetch distance for the
  //! given circular_buffered_id. Current code generation only supports one
  //! stage depth and prefetch distance per loop disjoint set, so this function
  //! will throw an error if trying to set different stage numbers to
  //! iterdomains that are loop mapped.
  void setStageDepthAndPrefetchDistance(
      IterDomain* circular_buffered_id,
      int64_t stage_depth,
      int64_t prefetch_distance);

 private:
  //! Keeps track of information for lowering circular buffered tensors
  std::unordered_map<const TensorView*, TvInfo> map_;

  //! Keeps track of which concrete loop map is realizing circular buffer
  //!  iterdomains.
  std::unordered_set<const IterDomain*> concrete_circular_buffered_loop_id_;

  //! Keeps track of circular buffer loop stage depth and prefetch distance.
  //! Currently for each disjoint set of loop mapped iterdomains,
  //! Only one stage depth and prefetch distance is supported, so that the loops
  //! can indeed shared with the same prolog extent and main loop offset.
  std::unordered_map<IterDomain*, CircularBufferOptions>
      circular_buffer_options_;

  //! Keeps track of circular buffer tvs for each disjoint set of loop mapped
  //! iterdomains.
  std::unordered_map<IterDomain*, std::unordered_set<const TensorView*>>
      circular_buffer_tvs_;
};

} // namespace nvfuser
