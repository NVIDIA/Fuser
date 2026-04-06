// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nvfuser {

class TensorIndexer;
namespace kir {
class ForLoop;
}

//! Information about a predicate. By default, it corresponds to a
//! single logical domain but may cover multiple logial domains due to
//! contigous indexing.
class PredicateInfo {
  friend class Index;
  friend class TensorIndexer;

 public:
  const auto& startPredicate() const {
    return start_predicate_;
  }

  auto& startPredicate() {
    return start_predicate_;
  }

  const auto& startOffset() const {
    return start_offset_;
  }

  const auto& stopPredicate() const {
    return stop_predicate_;
  }

  auto& stopPredicate() {
    return stop_predicate_;
  }

  const auto& stopOffset() const {
    return stop_offset_;
  }

  const auto& predicatedDomains() const {
    return predicated_domains_;
  }

  const auto& loopDomains() const {
    return loop_domains_;
  }

  CircularBufferLoopStage loopStage() const {
    return loop_stage_;
  }

  //! Return a false RootPredicateInfo, i.e., both start and stop
  //! predicates are false.
  static PredicateInfo getFalseInfo();

 private:
  // prdicate for lower end
  Val* start_predicate_ = nullptr;
  // prdicate for upper end
  Val* stop_predicate_ = nullptr;
  // Offset of the start predicate
  Val* start_offset_ = nullptr;
  // Offset of the stop predicate
  Val* stop_offset_ = nullptr;
  // Track which domains are covered by the generated predicates
  std::unordered_set<IterDomain*> predicated_domains_;
  // Loops domains used for the predicate domains
  std::unordered_set<IterDomain*> loop_domains_;
  // Circular buffer loop stage if applicable
  CircularBufferLoopStage loop_stage_ = CircularBufferLoopStage::NotApplicable;
};

// Simple interface for IndexCompute
class Index {
 public:
  // Indexing functions
  // Consumer = Producer
  // i.e. T0 = T1... -> T0 is the consumer, T1 is the producer
  // Producer indexing dispatch
  static kir::TensorIndex* getProducerIndex(
      TensorView* producer,
      const TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops,
      const std::unordered_map<IterDomain*, Val*>& override_index = {},
      bool generate_pointer = false,
      DataType as_type = DataType::Null,
      bool ld_st_matrix = false);

  // Consumer index dispatch
  static kir::TensorIndex* getConsumerIndex(
      TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops,
      const std::unordered_map<IterDomain*, Val*>& override_index = {},
      bool generate_pointer = false,
      DataType as_type = DataType::Null,
      bool ld_st_matrix = false);

  //! Returns the logical index linearized from a multi-dimension address into a
  //! linear memory address a consumer tensor. The returned index is intended to
  //! be used for the computation of some tensor factories, such as: iota and
  //! rand (for Philox pseudo random sequences)
  static Val* getLinearLogicalIndex(
      TensorView* consumer_tv,
      const std::vector<kir::ForLoop*>& loops);

  //! Returns a vector of logical indices mapped onto the logical
  //! domain of a consumer tensor. The returned index is intended
  //! to be used for the computation of some tensor factories, such as:
  //! eye
  static std::vector<Val*> getConsumerPerDimLogicalIndex(
      TensorView* consumer_tv,
      const std::vector<kir::ForLoop*>& loops);

  //! Returns a vector of logical indices mapped onto the logical
  //! domain of a producer tensor.
  static std::vector<Val*> getProducerPerDimLogicalIndex(
      TensorView* producer_tv,
      const TensorView* consumer_tv,
      const std::vector<kir::ForLoop*>& loops,
      const std::unordered_map<IterDomain*, Val*>& override_index = {});

  //! Compute the result for iota
  static Val* iota(
      TensorView* consumer_tv,
      const std::vector<kir::ForLoop*>& loops,
      Val* start,
      Val* step,
      DataType dtype);

  //! Compute the result for eye
  static Val* eye(
      TensorView* consumer_tv,
      const std::vector<kir::ForLoop*>& loops,
      DataType dtype);

  //! Compute the global index and the expected bytes for complete_tx mechanism
  //! for CpAsyncBulk.
  static std::pair<Val*, Val*> getCpAsyncBulkGmemIndex(
      const LoadStoreOp* ldst,
      Val* mbarrier,
      const std::vector<kir::ForLoop*>& loops);
};

struct PredicateDomainInfo {
 public:
  // Iteration domain to predicate
  IterDomain* id = nullptr;
  // The set of iteration domains that make up the id. If this is for
  // a non-divisible split, the set only contains the id itself. This
  // set is used to remove redundant predicates when gathering
  // unswitch predicates.
  std::unordered_set<IterDomain*> covered_ids;
  // True if this predicate is for an intermediate domain. Examples
  // include domains with non-divisible split and resized domains.
  bool is_intermediate_domain = false;
};

} // namespace nvfuser
