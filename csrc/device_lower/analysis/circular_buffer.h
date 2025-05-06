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
      bool ignore_prologue = false) const;

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
  //! Returns true if the fusion has warp specialized circular buffer
  const bool& hasWarpSpecialized() const {
    return has_warp_sepcialized_;
  };
  //! Get the circular buffer options for the given axis.
  const CircularBufferOptions& getCircularBufferOptionsFor(
      IterDomain* circular_buffered_id) const;

  //! Get the circular buffer insertion position for the given axis.
  int64_t getCircularBufferInsertionPosition(IterDomain* axis) const;

  //! Set the circular buffer insertion position for the given axis.
  void setCircularBufferInsertionPosition(
      const TensorView* circular_buffer_tv,
      IterDomain* circular_buffer_axis);

  //! Get the linearized index used for selecting the circular buffering stage
  //! and calculating mbarrier parity. The index includes all serial for-loops
  //! from outer-most to inner-most circular buffer axis. Assume the for_loop
  //! stack maps to the circular_buffer_tv's loop domain.
  Val* getLinearIndex(
      TensorView* circular_buffer_tv,
      const std::vector<ForLoop*>& loops) const;

  //! Get the linearized index used for selecting the circular buffering stage
  //! and calculating mbarrier parity. The index includes all serial for-loops
  //! from outer-most to inner-most circular buffer axis. Assume the for_loop
  //! stack can be anything to the left of the insertion position.
  Val* getLinearIndexRelativeForLoopStack(
      const std::vector<ForLoop*>& loops,
      int64_t insertion_position,
      int64_t start = 0) const;

  std::string toString() const;

 private:
  const TvInfo& getTvInfo(const TensorView* tv) const;

  TvInfo& getTvInfo(const TensorView* tv);

  //! Set the number of circular buffer options for the given
  //! circular_buffered_id. Current code generation only supports one option per
  //! loop disjoint set, so this function will throw an error if trying to set
  //! different options to iterdomains that are loop mapped.
  void setCircularBufferOptions(
      IterDomain* circular_buffered_id,
      const CircularBufferOptions& opt);

 private:
  //! Keeps track of information for lowering circular buffered tensors
  std::unordered_map<const TensorView*, TvInfo> map_;

  //! Keeps track of which concrete loop map is realizing circular buffer
  //!  iterdomains.
  std::unordered_set<const IterDomain*> concrete_circular_buffered_loop_id_;

  //! Keeps track of the circular buffer insertion position for each
  //! circular buffer loop.
  std::unordered_map<IterDomain*, int64_t> circular_buffer_insertion_position_;

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
  //! True if the fusion has warp specialized circular buffer
  bool has_warp_sepcialized_ = false;
};

} // namespace nvfuser
