// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>

#include <device_lower/pass/circular_buffer.h>
#include <ir/all_nodes.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>

namespace nvfuser {

// Warp specialization pads the CTA with 128 threads to support register
// sharing. This warp group is divided into four warps that run independently.
// These four warps are colloquially called AsyncWarp, and this struct tracks
// information for each individual warp.
//
// For example, take this Blackwell matrix multiplication kernel:
// if AsyncWarp:
//   decreaseRegisters()
//   if warp_id == 0:
//     mbarrier::wait(empty_operands)
//     mbarrier::arriveExpectTx(full_operands)
//     tma_load(A)
//     tma_load(b)
//   elif warp_id == 1:
//     mbarrier::wait(full_operands)
//     mbarrier::wait(empty_tmem_output)
//     tcgen05::utcmma(A_smem, B_smem)
//     mbarrier::arrive(full_tmem_output)
//   elif warp_id == 2:
//     mbarrier::wait(empty_epilogue_input)
//     mbarrier::arriveExpectTx(full_epilogue_input)
//     tma_load(epilogue_input)
//   elif warp_id == 3:
//     return;
//   return;
// else:  # ComputeWarp
//   increaseRegisters()
//   mbarrier::wait(full_tmem_output)
//   # load from tmem to registers
//   mbarrier::wait(epilogue_input)
//   # compute epilogue
//   # stmatrix to move from registers to shared memory
//   # tma_store to move from shared to global memory
//
// Three independent warps are active in the AsyncWarp warp group, so
// createAsyncWarps will filter all expressions to create three AsyncWarp
// structs.
//
// Here are the individual AsyncWarp structs and their constituent expressions:
//  1. AsyncWarp0: tma_load(A) and tma_load(B).
//  2. AsyncWarp1: tcgen05::utcmma
//  3. AsyncWarp2: tma_load(epilogue_input)
//  4. AsyncWarp3: Empty.
//
// Current implementation Details:
//  1. Only the mbarrier async operations in each warp is tracked in the
//  AsyncWarp struct. Consumer uses are not recorded.
//  2. Empty AsyncWarps are not tracked.
//  3. Only one AsyncWarp is supported.
struct AsyncWarp {
  // All the expressions in the AsyncWarp.
  std::vector<Expr*> exprs;
  // The corresponding output TensorView for all expressions.
  std::vector<TensorView*> tvs;
  // The common stage_slice_position for all TensorViews.
  int64_t stage_slice_position = -1;
};

// This helper function scans through all expressions, finds mbarrier async
// operations such as TMA Loads and Blackwell UTCMMA, and gather them into
// separate AsyncWarps.
std::vector<AsyncWarp> createAsyncWarps(const std::vector<Expr*>& exprs);

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

  void initializePingPongTracking(const TensorView* tv, IterDomain* cb_axis);

  void setCircularBufferTv(const TensorView* tv);

  IterDomain* getCircularBufferAxis(const TensorView* tv) const;

  //! Get all valid circular buffer TensorViews
  std::vector<const TensorView*> getCircularBufferTvs() const;

  //! Get a loop that matches with a given circular-buffer axis. If
  //! ignore_prologue is true, a matched loop is ignored if it's a
  //! prologue loop.
  static kir::ForLoop* getCircularBufferLoop(
      IterDomain* axis,
      const std::vector<kir::ForLoop*>& loops,
      bool ignore_prologue = false);

  //! Get a loop that matches with the circular-buffer axis of a given
  //! circular-buffered tensor. If ignore_prologue is true, a matched
  //! loop is ignored if it's a prologue loop.
  kir::ForLoop* getCircularBufferLoop(
      const TensorView* tv,
      const std::vector<kir::ForLoop*>& loops,
      bool ignore_prologue = false) const;

  //! Get the circular-buffered tensors for the given loop/axis.
  std::unordered_set<const TensorView*> getCircularBufferTvs(
      kir::ForLoop* axis) const;
  std::unordered_set<const TensorView*> getCircularBufferTvs(
      IterDomain* axis) const;

  void setOriginalAllocSize(const TensorView* tv, Val* size);

  Val* getOriginalAllocSize(const TensorView* tv);

  // Returns true if the warp groups run independently.
  bool hasIndependentComputeWarpGroups() const {
    return independent_compute_warp_groups_;
  }

  //! Returns true if the iterdomain will be realized
  //!  as a circular buffer loop.
  bool isCircularBufferedIterDomain(IterDomain* id);

  ParallelType getWarpSpecializedOn() const {
    return warp_specialized_on_;
  }

  //! Returns true if the fusion has warp specialized circular buffer
  bool hasWarpSpecialized() const {
    return warp_specialized_on_ != ParallelType::Serial;
  };

  //! Get the circular buffer options for the given axis.
  const CircularBufferOptions& getCircularBufferOptionsFor(
      IterDomain* circular_buffered_id) const;

  //! Get HopperPingPongMbarriers for the given axis.
  HopperPingPongMbarriers* getPingPongMbarriersFor(IterDomain* axis);

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
      const std::vector<kir::ForLoop*>& loops) const;

  //! Get the linearized index used for selecting the circular buffering stage
  //! and calculating mbarrier parity. The index includes all serial for-loops
  //! from outer-most to inner-most circular buffer axis. Assume the for_loop
  //! stack can be anything to the left of the insertion position.
  Val* getLinearIndexRelativeForLoopStack(
      const std::vector<kir::ForLoop*>& loops,
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

  //! Map IterDomain to HopperPingPongMbarriers
  std::unordered_map<IterDomain*, std::shared_ptr<HopperPingPongMbarriers>>
      ping_pong_mbarriers_;

  //! Keeps track of circular buffer tvs for each disjoint set of loop mapped
  //! iterdomains.
  std::unordered_map<IterDomain*, std::unordered_set<const TensorView*>>
      circular_buffer_tvs_;
  //! The warp specialized axis for circular buffering.
  //! Only one warp specialized axis for the fusion.
  ParallelType warp_specialized_on_ = ParallelType::Serial;
  //! If false, then the mbarrier in the ComputeWarp should be for all threads
  //! in ComputeWarp. Otherwise, it is per warp-group or 128 threads. It is True
  //! if the warp specialized axis is to the left of the stage_slice_position.
  bool independent_compute_warp_groups_ = false;
};

} // namespace nvfuser
