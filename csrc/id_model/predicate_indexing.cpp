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
#include <id_model/utils.h>

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

  if (expr->isA<ScatterOp>()) {
    auto index_input = expr->as<ScatterOp>()->index();
    if (index_input->isA<kir::TensorIndex>()) {
      index_input = index_input->as<kir::TensorIndex>()->view();
    }
    predicate_domains = index_input->as<TensorView>()->getLogicalDomain();
  }

  // Broadcast domains should not need to be predicated. Note that
  // unlike indexing for TensorIndex, reduction domains do need to be
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

namespace {

// Recall that when a loop domain is unswitched, the corresponding
// unswitch predicate needs to ensure that all the iteration values of the
// loop domain are covered. In principle, the maximum iteration value
// can be used to guard against the upper bound. However, when
// indexing propagtes backward a Merge expr, since the index of the
// inner input is calculated as (output_idx % inner_extent), the
// maximumness property may not be guaranteed. For example, given a 2D
// tensor of [I0, I1], suppose it's scheduled as:
//
// merge -> [I0*I1]
// split by 4 -> [ceilDiv(I0*I1, 4), 4]
// unswitch > [ceilDiv(I0*I1, 4), US(4)]
//
// In this case, we send a symbolic loop index of i for the outer loop
// domain and 3 for the unswitched inner domain. Suppose the actual
// extent of I1 is 3, the index of the I1 logical domain will be:
//
// (i * 4 + 3) % 3
//
// For example, when i is zero, the index is just zero. However,
// that's not the maximum possible index for the domain. Instead of
// assigning 3 to the inner loop domain of extent 4, it should use 2,
// which results in 2 for the inner logical domain.
//
// In the above example, it doesn't matter if the inner logical domain
// gets the actual maximum index because it's guaranteed to be less
// than the extent of the domain. However, that's not always the case,
// e.g.:
//
// split by 4 -> [I0, ceilDiv(I1, 4), 4]
// merge -> [I0*ceilDiv(I1, 4), 4]
// split by 1 -> [ceilDiv(I0*ceilDiv(I1, 4), 1), 1, 4]
// unswitch -> [ceilDiv(I0*ceilDiv(I1, 4), 1), US(1), 4]
//
// In this case, unless the maximumness property is guaranteed for the
// merge inner domain, the propagation through the backward split of
// the inner logical domain, I1, will not be able to guarantee the
// maximum index of the I1 domain. For a concret fusion example, see
// issue #681 as well as
// PredicateIndexingTest.UnswitchPredicateIssueRepro681.
void ensurePropagationOfMinMaxPredicates(
    TensorView* tv,
    const ValGroups& unswitched_loops,
    const std::unordered_map<ValGroup, Val*>& index_map,
    const ValGraph& traversal_graph,
    const ExprPath<ExprGroup>& traversal_path,
    const IdModel& id_model,
    bool is_start_predicate,
    std::unordered_map<Val*, Val*>& replacement_map) {
  // ID groups of the traversal graph
  std::unordered_set<ValGroup> unswitched_domains;

  // Gather all unswitched groups from the loop domains of this tensor
  for (auto loop_domain : tv->getLoopDomain()) {
    const auto& loop_group =
        id_model.idGraph(IdMappingMode::LOOP).toGroup(loop_domain);
    if (unswitched_loops.has(loop_group)) {
      unswitched_domains.emplace(traversal_graph.toGroup(loop_domain));
    }
  }

  if (unswitched_domains.empty()) {
    return;
  }

  // Propagate unswitching and assign a min or max index when necessary
  for (const auto& [expr_group, direction] : traversal_path) {
    // If any of inputs is unswitched, all outputs are considered unswitched
    const auto inputs = direction == Direction::Forward
        ? traversal_graph.inputGroups(expr_group)
        : traversal_graph.outputGroups(expr_group);

    if (std::any_of(
            inputs.begin(), inputs.end(), [&](const ValGroup& input) -> bool {
              return unswitched_domains.find(input) != unswitched_domains.end();
            })) {
      const auto outputs = direction == Direction::Forward
          ? traversal_graph.outputGroups(expr_group)
          : traversal_graph.inputGroups(expr_group);
      unswitched_domains.insert(outputs.begin(), outputs.end());
    }

    // The propagation issue happens when modulo is used, i.e.,
    // - inner domain of backward merge
    // - outer domain of forward split
    Expr* expr = expr_group->front();
    IterDomain* replacement_domain = nullptr;
    if (auto split = dynamic_cast<Split*>(expr);
        split != nullptr && direction == Direction::Forward) {
      replacement_domain = split->inner();
    } else if (auto merge = dynamic_cast<Merge*>(expr);
               merge != nullptr && direction == Direction::Backward) {
      replacement_domain = merge->inner();
    }

    if (replacement_domain == nullptr) {
      continue;
    }

    const auto& replacement_group = traversal_graph.toGroup(replacement_domain);
    if (unswitched_domains.find(replacement_group) ==
        unswitched_domains.end()) {
      continue;
    }

    auto index_it = index_map.find(replacement_group);
    NVF_ERROR(index_it != index_map.end());
    Val* current_idx = index_it->second;
    // Conservatively use either zero or max for start and stop,
    // respectively
    Val* replacement_idx = is_start_predicate
        ? current_idx->fusion()->zeroVal()
        : SimplifyingIrBuilder::subExpr(
              replacement_domain->extent(), current_idx->fusion()->oneVal());

    NVF_ERROR(
        replacement_map.emplace(current_idx, replacement_idx).second,
        "Attempted to register double replacement");
  }
}

} // namespace

std::unordered_map<Val*, Val*> getPredicateIndexReplacementMap(
    TensorView* tv,
    const std::vector<ForLoop*>& for_loops,
    const std::unordered_map<ValGroup, Val*>& index_map,
    const ValGraph& traversal_graph,
    const ExprPath<ExprGroup>& traversal_path,
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

  // If the tensor is circular buffered and the given for-loop is the
  // main loop of circular buffering, increment the index by
  // (number_of_stages - 1) since the main loop has a read that is
  // (number_of_stages - 1) elements ahead.
  // Only required for pipelined circular buffering, for warp specialized
  // circular buffering, there is no prologue or epilog loop.
  auto replace_for_circular_buffering = [&](ForLoop* fl,
                                            Val* original_index) -> Val* {
    auto circular_buffer_axis =
        GpuLower::current()->circularBufferInfo().getCircularBufferAxis(tv);
    if (circular_buffer_axis == nullptr ||
        !id_model.idGraph(IdMappingMode::LOOP)
             .disjointValSets()
             .strictAreMapped(fl->iter_domain(), circular_buffer_axis) ||
        lower_utils::isWarpSpecializedLoop(fl)) {
      return nullptr;
    }

    // Epilog should not hit this part since tv must be a circular
    // buffer tensor. Since predication is done based on a consumer
    // tensor, this tensor is a circular buffer tensor appearing as a
    // consumer tensor. Since no circular buffer tensor should appear
    // as a consumer in the epilog loop, the loop stage here must not
    // be epilog.
    NVF_ERROR(fl->circularBufferLoopStage() != CircularBufferLoopStage::Epilog);

    // The prologue loop does not need to be changed
    if (fl->circularBufferLoopStage() == CircularBufferLoopStage::Prolog) {
      return nullptr;
    } else {
      auto prefetch_distance =
          GpuLower::current()
              ->circularBufferInfo()
              .getCircularBufferOptionsFor(fl->iter_domain())
              .prefetch;
      return SimplifyingIrBuilder::addExpr(
          original_index,
          SimplifyingIrBuilder::create<Val>(
              prefetch_distance, DataType::Index));
    }
  };

  // Inspect the for-loops from outer to inner and keep track of
  // unswitching since it affects all inner loops
  bool within_unswitch = false;
  ValGroups unswitched_loops;
  for (const auto fl : for_loops) {
    auto parallel_type = fl->iter_domain()->getParallelType();

    // Note that unswitched_loop may be a vectorized loop
    if (fl == unswitched_loop && parallel_type != ParallelType::Vectorize) {
      within_unswitch = true;
    }

    if (within_unswitch && indexing_utils::isEffectiveUnswitchLoop(fl)) {
      unswitched_loops.pushBack(
          id_model.idGraph(IdMappingMode::LOOP).toGroup(fl->iter_domain()));
    }

    auto loop_id = getLoopPromotion(fl->iter_domain(), id_model);

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

    if (auto circular_buffer_index =
            replace_for_circular_buffering(fl, replacement)) {
      replacement = circular_buffer_index;
    }

    if (replacement != loop_index) {
      auto inserted = replacement_map.emplace(loop_index, replacement).second;
      NVF_ERROR(
          inserted, "Duplicate replacement attempted: ", loop_id->toString());
    }
  }

  ensurePropagationOfMinMaxPredicates(
      tv,
      unswitched_loops,
      index_map,
      traversal_graph,
      traversal_path,
      id_model,
      is_start_predicate,
      replacement_map);

  return replacement_map;
}

} // namespace nvfuser
