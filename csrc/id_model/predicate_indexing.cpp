// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/utils.h>
#include <id_model/indexing_utils.h>
#include <id_model/predicate_indexing.h>

namespace nvfuser {

std::vector<IterDomain*> getPredicateDomains(
    TensorView* consumer_tv,
    const Expr* expr) {
  // Logical domains should be the domains to predicate as they define
  // the logical shape of a tensor. However, in the case of rfactored
  // reductions, rfactor splits may not be divisible, thus root
  // domains need to be predicated. Note that the non-divisible split
  // info does not seem to cover non-divisible reduction rfactor
  // splits.
  std::vector<IterDomain*> predicate_domains = consumer_tv->hasReduction()
      ? consumer_tv->getMaybeRootDomain()
      : consumer_tv->getLogicalDomain();

  // Broadcast domains should not need to be predicated. Note that
  // unlike indexing for TensorIndex, reduction doamins do need to be
  // indexed to guard the access to the producer tensor
  predicate_domains.erase(
      std::remove_if(
          predicate_domains.begin(),
          predicate_domains.end(),
          [](IterDomain* id) -> bool { return id->isBroadcast(); }),
      predicate_domains.end());

  // If this is an expr initializing a buffer for a reduction, the
  // reduction domains do not need to be predicated. In fact, if it's
  // a Local tensor, no predicate is necessary at all
  if (lower_utils::isReductionInitExpr(expr)) {
    if (consumer_tv->getMemoryType() == MemoryType::Local) {
      return {};
    } else {
      predicate_domains.erase(
          std::remove_if(
              predicate_domains.begin(),
              predicate_domains.end(),
              [](IterDomain* id) -> bool { return id->isReduction(); }),
          predicate_domains.end());
    }
  }

  return predicate_domains;
}

std::unordered_map<Val*, Val*> getPredicateIndexReplacementMap(
    TensorView* tv,
    const std::vector<ForLoop*>& for_loops,
    const std::unordered_map<ValGroup, Val*>& index_map,
    const ValGraph& traversal_graph,
    const IdModel& id_model,
    bool is_start_predicate,
    ForLoop* unswitched_loop) {
  std::unordered_map<Val*, Val*> replacement_map;

  // For an iter domain of index i, it is valid to use N-1 instead of
  // i, where N is the extent of the iter domain if either of the
  // following conditions is satisfied:
  //
  // - Vectorized
  // - predicateAtEnd returns true
  // - Within an unswitch/unroll loop
  //
  // Use N-1 instead of i but not when it's thread parallelized so
  // that each thread or block can take different paths. This may not
  // be optimal for TID, though, as it might result in thread
  // divergence.
  //
  // Also in the case of vectorization, instead of N-1, it's also
  // valid to use 0 since the splits involved to create the iter
  // domain are all guaranteed to be divisible.
  auto predicate_at_end =
      [&](ForLoop* fl, IterDomain* loop_id, bool within_unswitch) -> Val* {
    // Don't replace thread indices even when unswitched
    if (!fl->iter_domain()->isThread() &&
        (fl->iter_domain()->getParallelType() == ParallelType::Vectorize ||
         within_unswitch || lower_utils::predicateAtEnd(fl))) {
      return is_start_predicate
          ? fl->fusion()->zeroVal()
          : SimplifyingIrBuilder::subExpr(
                fl->simplifiedStop(), fl->fusion()->oneVal());
    } else {
      return nullptr;
    }
  };

  auto replace_for_circular_buffering =
      [&](ForLoop* fl, Val* original_index, bool within_unswitch) -> Val* {
    auto circular_buffer_axis =
        GpuLower::current()->circularBufferInfo().getCircularBufferAxis(tv);
    if (circular_buffer_axis == nullptr ||
        !id_model.idGraph(IdMappingMode::LOOP)
             .disjointValSets()
             .strictAreMapped(fl->iter_domain(), circular_buffer_axis)) {
      return nullptr;
    }

    // The prologue loop does not need to be changed
    if (fl->circularBufferLoopStage() == CircularBufferLoopStage::Prolog) {
      return nullptr;
    } else {
      auto stage_depth =
          (int64_t)GpuLower::current()->circularBufferInfo().getStageDepthFor(
              fl->iter_domain());
      return SimplifyingIrBuilder::addExpr(
          original_index,
          SimplifyingIrBuilder::create<Val>(stage_depth - 1L, DataType::Index));
    }
  };

  // Inspect the for-loops from outer to inner and keep track of
  // unswitching since it affects all inner loops
  bool within_unswitch = false;
  for (const auto fl : for_loops) {
    auto parallel_type = fl->iter_domain()->getParallelType();

    // Note that unswitched_loop may be a vectorized loop
    if (fl == unswitched_loop && parallel_type != ParallelType::Vectorize) {
      within_unswitch = true;
    }

    auto loop_id =
        indexing_utils::getLoopPromotion(fl->iter_domain(), id_model);

    NVF_ERROR(
        !loop_id->maybePartial(),
        "Partial loop not supported: ",
        fl->toString());

    auto loop_index_it = index_map.find(traversal_graph.toGroup(loop_id));

    if (loop_index_it == index_map.end()) {
      // The index map is built from the tensor loop domains. There
      // can be for-loops that are not part of this tensor, e.g, a
      // tensor inlined into a higher dimensional tensor.
      continue;
    }

    Val* loop_index = loop_index_it->second;

    // If it's already const scalar, no replacment should be necessary
    if (loop_index->isConst()) {
      continue;
    }

    Val* replacement = loop_index;

    // Trivial loop. Note that not all trivial loops should just use
    // the start index for predication. For example, a vectorized loop
    // is trivial, but its predicate should use `vec_factor - 1` as
    // its index. This is taken care after this.
    if (fl->isTrivial()) {
      replacement = fl->start();
    }

    if (auto idx = predicate_at_end(fl, loop_id, within_unswitch)) {
      replacement = idx;
    }

    // Adjustment for circular buffering
    if (auto circular_buffer_index =
            replace_for_circular_buffering(fl, replacement, within_unswitch)) {
      replacement = circular_buffer_index;
    }

    if (replacement != loop_index) {
      auto inserted = replacement_map.emplace(loop_index, replacement).second;
      NVF_ERROR(
          inserted, "Duplicate replacement attempted: ", loop_id->toString());
    }
  }

  return replacement_map;
}

} // namespace nvfuser
