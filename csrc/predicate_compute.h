// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <device_lower/analysis/thread_predicate.h>
#include <device_lower/utils.h>
#include <exceptions.h>
#include <index_compute.h>
#include <kernel_ir.h>
#include <root_domain_map.h>

namespace nvfuser {

class PredicateCompute {
 public:
  // ignore_internal_syncthread_ops will prevent creation of predicates on
  // block/grid broadcast/reduce as these have syncthread calls within them
  // so all threads need to execute the function.
  static Val* getInlinePredicate(
      const Expr* expr,
      const std::vector<kir::ForLoop*>& loops,
      const std::unordered_set<kir::ForLoop*>& rotated_loops,
      Val* thread_pred,
      PredicateType pred_type);
};

//! Parallelized domains may need to be predicated with threading
//! indices and IterDomain extents. For example, if a domain is
//! parallelized by TIDx, when TIDx is not exact, i.e., it can be
//! larger than the extents of domains parallelized by TIDx,
//! threadIdx.x may be larger than the IterDomain extent. This can be
//! harmless for Local tensors, however, for it would
//! result in out-of-bounds access for Shared tensors as they are
//! allocated based on tensor shapes rather than threading
//! dimensions.
class ParallelizedDomainPredicate {
 public:
  //! Predicate information for parallelized domains
  class PredicateInfo {
   public:
    explicit PredicateInfo(ParallelType pt) : pt_(pt) {}

    //! Adds a domain that is parallized by the same paralell type
    bool addDomain(IterDomain* id);

    const std::vector<IterDomain*>& ids() const {
      return ids_;
    }

    //! Generates a predicate Val from predicate information
    Val* getPredicate() const;

   private:
    ParallelType pt_;
    //! Domains parallelized by the same parallel type
    std::vector<IterDomain*> ids_;
  };

  //! Returns a predicate Val for parallelied domains of an expression.
  static Val* getPredicate(
      const Expr* expr,
      const std::vector<kir::ForLoop*>& loops);

  //! Returns predicate information for parallelied domains of an
  //! expression.
  static std::unordered_map<ParallelType, PredicateInfo> getPredicateMap(
      const Expr* expr,
      const std::vector<kir::ForLoop*>& loops,
      kir::ForLoop* unswitched_loop = nullptr);
};

//! Keys to identify unique unswitch predicates. Just consists of a
//! predicated concrete domain if not parallelized. If parallelized,
//! pick one for each different parallelization. When the same
//! parallel type is used for different concrete domains, they are
//! considered different predicates and are included in the unswitch
//! condition lists.
class UnswitchPredicateKey {
 public:
  UnswitchPredicateKey();

  UnswitchPredicateKey(
      IterDomain* predicated_consumer_id,
      TensorView* consumer_tv,
      IterDomain* predicated_concrete_id);

  bool operator==(const UnswitchPredicateKey& other) const {
    return predicated_concrete_id_ == other.predicated_concrete_id_ &&
        parallel_concrete_ids_ == other.parallel_concrete_ids_;
  }

  const auto& predicatedId() const {
    return predicated_concrete_id_;
  }

  const auto& parallelConcreteIds() const {
    return parallel_concrete_ids_;
  }

  IterDomain* parallelId(ParallelType pt) const {
    auto it = parallelConcreteIds().find(pt);
    if (it == parallelConcreteIds().end()) {
      return nullptr;
    } else {
      return it->second;
    }
  }

  std::string toString() const;

 private:
  //! Predicated concrete domain
  IterDomain* predicated_concrete_id_ = nullptr;
  //! Store parallelized concrete domains
  std::unordered_map<ParallelType, IterDomain*> parallel_concrete_ids_;
};

struct UnswitchPredicateKeyHash {
  std::size_t operator()(const UnswitchPredicateKey& key) const;
};

class UnswitchPredicate {
 public:
  static Val* get(
      const std::vector<kir::ForLoop*>& outer_loops,
      kir::ForLoop* unrolled_loop);

 private:
  //! Predicate information for each UnswitchPredicateKey.
  struct MergedPredicates {
    //! Predicate information for the start and stop predicates.
    struct Info {
      //! Most restrictive static predicate. Nullptr if no static
      //! predicate found.
      Val* static_pred = nullptr;
      //! The offset value of static_pred
      PolymorphicValue static_offset = 0L;
      //! List of dynamic predicates.
      std::vector<Val*> dynamic_preds;
    };
    UnswitchPredicateKey predicate_key;
    Info start;
    Info stop;
  };

  UnswitchPredicate(
      std::vector<kir::ForLoop*> outer_loops,
      kir::ForLoop* unrolled_loop);

  void predicateOn(Expr*);

  void openLoop(kir::ForLoop*);

  void openIte(kir::IfThenElse*);

  //! Generates the final predicates from the predicated_keys map
  void finalize();

  //! Merge predicates as much as possible. If a predicate offset is
  //! static, only pick the most restrictive one, e.g., the one with the
  //! minimum offset for the start predication.
  void mergeUnswitchPredicateOffsets(
      Val* predicate,
      Val* offset,
      MergedPredicates::Info& merged_predicate_info,
      bool is_start);

  //! Adds new predicates for parallelized domains
  void addParallelizedDomainPredicates(Expr*);

 private:
  //! Track which iter domains have been predicated
  std::unordered_set<UnswitchPredicateKey, UnswitchPredicateKeyHash>
      predicated_keys_;

  //! The predicates that have been recorded but not yet finalized
  std::vector<MergedPredicates> pending_predicates_;

  //! Track which parallelized domains have been predicated
  std::unordered_map<ParallelType, ParallelizedDomainPredicate::PredicateInfo>
      parallelized_dom_predicates_;

  //! The predicates that have been generated.
  std::vector<Val*> predicates_;

  std::vector<kir::ForLoop*> for_loops_;

  // Keep track of the loop in which the currently visiting expr is a rotated.
  std::unordered_set<kir::ForLoop*> rotated_loop_;

  kir::ForLoop* unrolled_loop_;
};

} // namespace nvfuser
