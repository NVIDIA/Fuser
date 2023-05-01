// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <c10/macros/Export.h>

#include <ir_all_nodes.h>
#include <lower_utils.h>
#include <parallel_type_bitmap.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace nvfuser {

//! Maps TensorViews to a { ParallelTypeBitmap, SourceMap } pair
//!
//! Map from TensorView to bit set represnting <BIDx, BIDy, BIDz, TIDx, TIDy,
//! TIDz> If any dependency of TV had a parallelized reduction, we will track
//! it here. This will be used for predicate generation to prevent
//! parallelization on that axis. This is important if we have a reduction on
//! for example TIDx, as the reduced value is only valid on threadIdx.x == 0
//! therefore if we use that value later in the kernel we have that predicate.
//! If we follow a reduction parallelized on TIDx with a broadcast on TIDx we
//! no longer need the predicate and can reset the bit accordingly
//!
//! In addition, if a parallel thread type is not used, it is
//! redundant to use all threads/blocks. That isn't a problem
//! generally although it can be inefficient, but when an aliased smem
//! buffer is used as an output, redundant writes can be invalid (see issue
//! #1110). PredicateInfo::redundant_types track which parallel types
//! are redundant for each tensor and is used to let only one
//! thread/block of a redundant type execute the expression for a
//! tensor.
class TORCH_CUDA_CU_API ThreadPredicateMap {
 public:
  using SourceMap =
      std::unordered_map<ParallelType, std::unordered_set<const TensorView*>>;

  //! Thread predicate information for each tensor
  struct PredicateInfo {
    // Parallel types where only one thread/block is valid.
    ParallelTypeBitmap limited_types;
    // Parallel types where only one thread/block is enough.
    ParallelTypeBitmap redundant_types;

    // when a leaf domain is merged from concretized broadcast root domain, only
    // part of thread/block do the write to gmem is enough.
    // e.g. [B1,I2,B3] is merged to [B1*I2*B3] and parallelized by blockIdx.x,
    // the write condition is: blockIdx.x == write_index,
    // where write_index = write_index_map[ParallelType::BIDx].
    // write_index is calculated by the following method:
    // (1) Find the root domains merged to this leaf domain, e.g. [B1, I2, B3]
    // (2) Calculate the stride if we index the leaf domain using its root
    // domains,
    //     e.g. extended_stride = [len(I2)*len(B3), len(B3) ,1],
    //     where len(Bx) is the length its concretized domain.
    // (3) Calculate the index of each dimension. e.g. linear_index =
    // blockIdx.x;
    //     index[i] = (linear_index / stride[i]); linear_index %= stride[i];
    // (4) index the leaf domain skipping the broadcasted dimensions.
    //     e.g. write_index = 0 * stride[0] + index[1] * stride[1] + 0 *
    //     stride[2]

    std::unordered_map<ParallelType, Val*> write_index_map;

    // Tracking use chain of redundant writes:
    //  [Redundant use chain]
    //  a parallel type is a `redundant_consumer_type` only
    //    if all of its propagation use chains terminate with
    //    a redundant write of this type.
    //  A propagation use chain is currently either a reg-to-reg
    //   chain for a shared mem tv, or a reg/smem-to-reg/smem chain
    //   for a global tv.
    // This is complementary information to `redundant_types`.
    //  If a tensor view is redundantly written and not redundantly
    //  used by all consumers, see FusionRedundantPredSync3,
    //  a RAW sync will need to be inserted before reading
    //  this redundantly written tensor.
    ParallelTypeBitmap redundant_use_types;
    bool operator==(const PredicateInfo& other) const {
      return limited_types == other.limited_types &&
          redundant_types == other.redundant_types &&
          redundant_use_types == other.redundant_use_types;
    }
  };

  using MapType = std::unordered_map<const TensorView*, PredicateInfo>;

  using const_iterator = MapType::const_iterator;

  //! Build a map from each tensor to PredicateInfo.
  void build(Fusion* fusion);

  //! Get a PredicateInfo for a given tensor. If it's an output of
  //! a parallel broadcast, unmask the limited_types_ bit of the
  //! corresponding parallel type since it must join the broadcast
  //! operation although the valid input is only available at one of
  //! the threads/blocks.
  PredicateInfo getPredicateInfo(const TensorView* tv) const;

  //! Returns a flag set that indicates which parallel types should be
  //! predicated.
  ParallelTypeBitmap getPredicatedParallelTypes(const TensorView* tv) const;

  //! Returns a Bool predicate for a given TensorView.
  Bool* getPredicate(
      const TensorView* tv,
      ParallelTypeBitmap mask = ParallelTypeBitmap().setAll()) const;

  //! Returns a ParallelTypeBitmap representing which domain needs
  //! blockBroadcast.
  //!
  //! Even when a domain is broadcast and parallelized, it does not need
  //! blockBroadcast unless it is predicated by limited_types_
  ParallelTypeBitmap getParallelBroadcastDomains(const TensorView* tv) const;

  //! Mark tv as updated so that rebuilding the map should recompute
  //! its predicates and those of its dependents.
  void markAsUpdated(const TensorView* tv);

  void print() const;

  //! Generate a Bool value from PredicateInfo.
  static Bool* getPredicateFromPredicateInfo(
      const ThreadPredicateMap::PredicateInfo& pred_info,
      const ParallelTypeBitmap& mask);

  //! Get the redundant use types of the given expr, see [Redundant use chain]
  ParallelTypeBitmap getRedundantConsumerType(Expr* expr) const;

 private:
  // Update the thread_predicates bitset based on provided Expr
  void updateBitSet(const Expr*);
  void avoidConcretizedBroadcastRedundantWrite(const TensorView* out_tv);
  const_iterator find(const TensorView* tv) const;
  const_iterator end() const;

  const PredicateInfo& at(const TensorView* tv) const;
  PredicateInfo& at(const TensorView* tv);

  //! Update a mapping
  bool update(
      const TensorView* tv,
      const ParallelTypeBitmap& limited_types,
      const ParallelTypeBitmap& redundant_types);

  //! Update a mapping
  bool update(const TensorView* tv, const PredicateInfo& pred_and_src);

  //! Backward populate redundant use chain info once the redundant
  //!  parallel writes have been identified.
  void populateRedundantUseMap(Fusion* fusion);

 private:
  MapType thread_predicates_;
  //! Keep track of updated tensors that need predicates to be computed
  std::unordered_set<const TensorView*> updated_tvs_;
};

} // namespace nvfuser
