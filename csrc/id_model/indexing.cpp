// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <device_lower/analysis/index_compute.h>
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <expr_simplifier.h>
#include <id_model/circular_buffer_indexing.h>
#include <id_model/contiguity.h>
#include <id_model/id_model_index_compute.h>
#include <id_model/indexing.h>
#include <id_model/indexing_traversal.h>
#include <id_model/indexing_utils.h>
#include <id_model/predicate_indexing.h>
#include <id_model/to_string.h>
#include <id_model/utils.h>
#include <index_compute.h>
#include <ir/builder.h>
#include <ir/graphviz.h>
#include <ir/utils.h>
#include <kernel_ir_dispatch.h>
#include <swizzle.h>
#include <val_graph_visitor.h>

#include <algorithm>
#include <fstream>

namespace nvfuser {

TensorIndexer::TensorIndexer(IdModel& id_model) : id_model_(id_model) {
  buildLoopIndexMap();

  if (isDebugDumpEnabled(DebugDumpOption::IndexingVerbose)) {
    traversalGraph().dumpGraphvizDotGraph("indexing_traversal_graph.dot");
  }
}

void TensorIndexer::buildLoopIndexMap() {
  if (id_model_.empty()) {
    return;
  }

  Fusion* fusion = id_model_.fusion();
  FusionGuard fg(fusion);

  for (auto expr : fusion->exprs()) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }
    // It's assumed that all sibling outputs share the same for-loops,
    // thus only one of the outputs is considered.
    auto tv_output = ir_utils::getTvOutput(expr);
    for (auto loop_id : tv_output->getLoopDomain()) {
      const ValGroup& loop_group =
          id_model_.idGraph(IdMappingMode::LOOP).toGroup(loop_id);

      if (loop_index_map_.find(loop_group) != loop_index_map_.end()) {
        // Index already assigned
        continue;
      }

      Val* loop_index = nullptr;

      if (shouldUseZeroIndex(loop_group, id_model_)) {
        loop_index = fusion->zeroVal();
      } else {
        loop_index = GpuLower::current()->getLoopIndexVariable(loop_id);
      }

      loop_index_map_[loop_group] = loop_index;
      indexing_utils::verbose()
          << "Loop index map: " << nvfuser::toString(loop_group) << " -> "
          << loop_index->toInlineString() << std::endl;
    }
  }
}

const AllocationDomainInfo& TensorIndexer::getIndexAllocationInfo(
    TensorView* tv) const {
  return GpuLower::current()->getAllocationInfo(tv);
}

Val* TensorIndexer::getLoopIndex(
    IterDomain* loop_id,
    const std::vector<ForLoop*>& for_loops) const {
  // loop_id must be a loop domain.
  const auto& loop_group =
      id_model_.idGraph(IdMappingMode::LOOP).toGroup(loop_id);
  auto loop_index_map_it = loop_index_map_.find(loop_group);
  NVF_ERROR(
      loop_index_map_it != loop_index_map_.end(),
      "No loop index found for ",
      loop_id->toString());

  Val* loop_index = loop_index_map_it->second;

  // War for circular buffering
  if (auto circular_buffer_loop_index =
          getLoopIndexOfCircularBufferLoop(loop_id, for_loops, id_model_)) {
    loop_index = circular_buffer_loop_index;
  }

  return loop_index;
}

std::unordered_map<ValGroup, Val*> TensorIndexer::getInitialIndexMap(
    const std::vector<IterDomain*>& loop_domains,
    const std::vector<ForLoop*>& for_loops) const {
  std::unordered_map<ValGroup, Val*> initial_index_map;

  // For a given list of the loop domains, assign its corresponding
  // index Val.
  for (IterDomain* loop_id : loop_domains) {
    Val* initial_index = getLoopIndex(loop_id, for_loops);
    const auto& almost_exact_group = traversalGraph().toGroup(loop_id);

    if (initial_index_map.find(almost_exact_group) != initial_index_map.end()) {
      // Initial index already set. This can happen as this is an
      // almost exact group. It should be just size-1 domain.
      NVF_ERROR(
          initial_index->isZeroInt(),
          "Unexpected initial index: ",
          initial_index->toInlineString());
      auto existing_index = initial_index_map.at(almost_exact_group);
      NVF_ERROR(
          existing_index->isZeroInt(),
          "Unexpected initial index: ",
          existing_index->toInlineString());
      continue;
    }

    initial_index_map.emplace(almost_exact_group, initial_index);
  }

  return initial_index_map;
}

std::vector<Val*> TensorIndexer::getIndexFor(
    const Expr* expr,
    bool as_consumer,
    const std::vector<IterDomain*>& index_ids,
    const std::vector<ForLoop*>& for_loops) const {
  auto info = computeIndex(expr, index_ids, for_loops);
  const auto& replacement_map = getIndexReplacementMap(
      expr, as_consumer, info.loop_ids, for_loops, info.index_map);

  // Note that IDs of index_ids may be mapped as the traversal graph
  // is the AlmostExact graph.

  std::vector<Val*> result;
  result.reserve(index_ids.size());
  for (IterDomain* index_id : index_ids) {
    const auto& index_group = traversalGraph().toGroup(index_id);
    auto it = info.index_map.find(index_group);
    NVF_ERROR(
        it != info.index_map.end(),
        "Index not found for ",
        index_id->toString());
    result.push_back(
        ir_utils::replaceValRecursively(it->second, replacement_map));
  }
  return result;
}

Val* TensorIndexer::getLinearIndex(
    TensorView* tv,
    const Expr* expr,
    const std::vector<ForLoop*>& for_loops,
    const std::unordered_map<IterDomain*, Val*>& override_index) const {
  NVF_ERROR(tv != nullptr);
  NVF_ERROR(expr != nullptr);
  NVF_ERROR(
      (std::find(expr->inputs().begin(), expr->inputs().end(), tv) !=
       expr->inputs().end()) ||
          (std::find(expr->outputs().begin(), expr->outputs().end(), tv) !=
           expr->outputs().end()),
      "Inconsistent tensor and expr. Tensor, ",
      tv->toString(),
      " not found in ",
      expr->toString());

  const bool as_consumer =
      std::find(expr->outputs().begin(), expr->outputs().end(), tv) !=
      expr->outputs().end();

  indexing_utils::verbose() << "getLinearIndex of " << tv->toString() << " as "
                            << (as_consumer ? "consumer" : "producer") << " in "
                            << expr->toString() << std::endl;

  const auto alloc_info = getIndexAllocationInfo(tv);

  indexing_utils::verbose()
      << "Allocation domains: " << toDelimitedString(alloc_info.ids)
      << std::endl;

  const auto [contig_indices, contig_strides] = getContigIndexFor(
      tv, expr, as_consumer, alloc_info, for_loops, override_index);

  // Linearize the indices with strides.
  Val* linear_index = tv->fusion()->zeroVal();
  for (const auto i : c10::irange(contig_indices.size())) {
    Val* stride = contig_strides.at(i);
    linear_index = SimplifyingIrBuilder::addExpr(
        linear_index,
        SimplifyingIrBuilder::mulExpr(contig_indices.at(i), stride));
  }

  // If a tensor is circular buffered, it also requires indexing of
  // the circular buffer itself
  if (tv->isCircularBuffered()) {
    auto circular_buffer_offset =
        getOffsetForCircularBufferTensor(tv, as_consumer, for_loops);
    linear_index =
        SimplifyingIrBuilder::addExpr(linear_index, circular_buffer_offset);
  }

  indexing_utils::verbose()
      << "Final index: " << linear_index->toInlineString() << std::endl;
  return linear_index;
}

// Get the loop domains of a given expr, which are (potentially
// promoted) loop domains of the consumer tensor.
std::vector<IterDomain*> TensorIndexer::getLoopDomains(const Expr* expr) const {
  // Assume consumer-based indexing. Needs to revisit for ops like
  // scatter
  auto loop_domains = ir_utils::getTvOutput(expr)->getLoopDomain();

  // If this is an expr initializing a buffer for a reduction, there
  // should be no loops for reduction domains
  if (lower_utils::isReductionInitExpr(expr)) {
    std::erase_if(
        loop_domains, [](IterDomain* id) -> bool { return id->isReduction(); });
  }

  for (auto& loop_id : loop_domains) {
    loop_id = getLoopPromotion(loop_id, id_model_);
  }

  return loop_domains;
}

IndexingInfo TensorIndexer::computeIndex(
    const Expr* expr,
    const std::vector<IterDomain*>& index_ids,
    const std::vector<ForLoop*>& for_loops) const {
  const auto loop_ids = getLoopIds(expr, id_model_);
  const ExprPath<ExprGroup> traversal_path = getIndexingPath(expr, index_ids);
  const std::unordered_map<ValGroup, Val*> initial_index_map =
      getInitialIndexMap(loop_ids, for_loops);

  IdGraphIndexCompute index_compute(traversalGraph(), initial_index_map);

  // In addition to indices themselves, keep track of the
  // dependency from each domain to loop domains. This dependency is
  // represented as a map from ValGroup of the traversal graph to
  // ValGroup of the LOOP graph.
  std::unordered_map<ValGroup, ValGroups> loop_group_dependencies;

  // Initialize the loop dependency mappings
  for (const auto& loop_domain : loop_ids) {
    const auto& traversal_graph_group = traversalGraph().toGroup(loop_domain);
    const auto& loop_graph_group =
        id_model_.idGraph(IdMappingMode::LOOP).toGroup(loop_domain);
    loop_group_dependencies[traversal_graph_group].pushBack(loop_graph_group);
  }

  for (const auto& [expr_group, direction] : traversal_path) {
    index_compute.propagate(expr_group, direction);

    // Propagate loop dependencies from inputs to outputs
    const auto input_groups = direction == Direction::Forward
        ? traversalGraph().inputGroups(expr_group)
        : traversalGraph().outputGroups(expr_group);
    const auto output_groups = direction == Direction::Forward
        ? traversalGraph().outputGroups(expr_group)
        : traversalGraph().inputGroups(expr_group);
    for (const auto& output : output_groups) {
      for (const auto& input : input_groups) {
        const auto& input_loop_groups = loop_group_dependencies.at(input);
        loop_group_dependencies[output].pushBack(input_loop_groups);
      }
    }
  }

  // Fill in broadcast index groups by zero
  auto index_map = index_compute.indexMap();
  for (const auto index_id : index_ids) {
    if (index_id->isBroadcast()) {
      index_map[traversalGraph().toGroup(index_id)] =
          index_id->fusion()->zeroVal();
    }
  }

  IndexingInfo info{
      loop_ids, index_ids, traversal_path, index_map, loop_group_dependencies};
  return info;
}

std::unordered_map<Val*, Val*> TensorIndexer::getIndexReplacementMap(
    const Expr* expr,
    bool as_consumer,
    const std::vector<IterDomain*>& loop_domains,
    const std::vector<ForLoop*>& for_loops,
    const std::unordered_map<ValGroup, Val*>& index_map) const {
  std::unordered_map<Val*, Val*> replacement_map;

  for (const auto loop_id : loop_domains) {
    Val* cur_index = getLoopIndex(loop_id, for_loops);

    Val* replacement_index = nullptr;
    // Replace the index of a vectorized/bulk domain with zero. Note that
    // vectorized domains may need to use N-1, where N is the extent
    // of the domain, for predication, so the replacement is not
    // always done with zero.
    if (loop_id->getParallelType() == ParallelType::Vectorize ||
        loop_id->getParallelType() == ParallelType::Bulk ||
        loop_id->getParallelType() == ParallelType::Mma) {
      replacement_index = loop_id->fusion()->zeroVal();
    } else {
      ForLoop* for_loop = indexing_utils::getForLoop(
          loop_id, for_loops, id_model_.idGraph(IdMappingMode::LOOP));

      // for_loop is nullptr if no matching loop is found, which
      // happens when loop_id is a reduction domain and this loop-nest
      // is for initializing the reduction buffer.
      if (for_loop != nullptr) {
        // Replace circular buffer index with zero value if for-loop is trivial
        if (for_loop->circularBufferLoopStage() !=
            CircularBufferLoopStage::NotApplicable) {
          Val* base_index =
              replacement_index != nullptr ? replacement_index : cur_index;
          replacement_index =
              for_loop->isTrivial() ? for_loop->start() : base_index;
        }

        // If this for-loop is a circular buffer loop, the loop index
        // may need to have an additional offset.
        if (!as_consumer) {
          if (auto circular_buffer_offset =
                  getLoopIndexOffsetForProducerOfCircularBuffer(
                      expr, for_loop, id_model_)) {
            replacement_index = SimplifyingIrBuilder::addExpr(
                replacement_index, circular_buffer_offset);
          }
        }
      }
    }

    if (replacement_index == nullptr || replacement_index == cur_index) {
      continue;
    }

    replacement_map.emplace(cur_index, replacement_index);
  }

  return replacement_map;
}

std::vector<PredicateInfo> TensorIndexer::getPredicates(
    TensorView* tv,
    const Expr* expr,
    const std::vector<ForLoop*>& for_loops,
    ForLoop* unswitched_loop) const {
  const auto& zero_val = tv->fusion()->zeroVal();

  const std::vector<IterDomain*>& predicate_domains =
      getPredicateDomains(tv, expr);

  if (predicate_domains.empty()) {
    return {};
  }

  const IndexingInfo& index_info =
      computeIndex(expr, predicate_domains, for_loops);

  const auto& index_map = index_info.index_map;

  const std::unordered_map<Val*, Val*> replacement_map_start =
      getPredicateIndexReplacementMap(
          tv,
          for_loops,
          index_map,
          traversalGraph(),
          index_info.traversal_path,
          id_model_,
          /*is_start_predicate=*/true,
          /*unswitched_loop=*/unswitched_loop);

  const std::unordered_map<Val*, Val*> replacement_map_stop =
      getPredicateIndexReplacementMap(
          tv,
          for_loops,
          index_map,
          traversalGraph(),
          index_info.traversal_path,
          id_model_,
          /*is_start_predicate=*/false,
          /*unswitched_loop=*/unswitched_loop);

  const std::unordered_map<IterDomain*, ValGroup> contig_domains =
      isContigIndexingEnabled()
      ? getContigDomains(
            predicate_domains,
            std::vector<bool>(predicate_domains.size(), true),
            reverse(index_info.traversal_path),
            traversalGraph(),
            /*is_predicate_pass=*/true)
      : std::unordered_map<IterDomain*, ValGroup>{};

  auto getCoveredPredicatedDomains =
      [&predicate_domains, &contig_domains](const ValGroup& contig_group) {
        std::unordered_set<IterDomain*> covered_domains;
        for (const auto& predicate_domain : predicate_domains) {
          auto contig_domains_it = contig_domains.find(predicate_domain);
          NVF_ERROR(contig_domains_it != contig_domains.end());
          if (contig_group == contig_domains_it->second) {
            covered_domains.emplace(predicate_domain);
          }
        }
        return covered_domains;
      };

  const CircularBufferLoopStage loop_stage = getCircularBufferLoopStage(
      tv, for_loops, id_model_.idGraph(IdMappingMode::LOOP));

  std::vector<PredicateInfo> info_vec;
  info_vec.reserve(predicate_domains.size());

  std::unordered_set<ValGroup> already_indexed_domains;

  // Follow the same approach as Index::getReferenceRootPredicates.
  for (const auto& predicate_domain : predicate_domains) {
    const auto& predicate_domain_group =
        traversalGraph().toGroup(predicate_domain);
    IterDomain* actual_predicate_domain = predicate_domain;
    ValGroup actual_predicate_domain_group = predicate_domain_group;
    std::unordered_set<IterDomain*> actual_predicate_domains = {
        predicate_domain};

    if (isContigIndexingEnabled()) {
      auto contig_domains_it = contig_domains.find(predicate_domain);
      NVF_ERROR(
          contig_domains_it != contig_domains.end(),
          "No contig domain mapping found for ",
          predicate_domain->toString());
      const ValGroup& contig_domain_group = contig_domains_it->second;
      if (already_indexed_domains.find(contig_domain_group) !=
          already_indexed_domains.end()) {
        indexing_utils::verbose()
            << "Already indexed: " << predicate_domain->toString() << std::endl;
        continue;
      }
      already_indexed_domains.emplace(contig_domain_group);

      actual_predicate_domain_group = contig_domain_group;
      actual_predicate_domain =
          actual_predicate_domain_group->front()->as<IterDomain>();
      actual_predicate_domains =
          getCoveredPredicatedDomains(contig_domain_group);
    }

    auto idx_it = index_map.find(actual_predicate_domain_group);
    NVF_ERROR(
        idx_it != index_map.end(),
        "Index not found for ",
        nvfuser::toString(actual_predicate_domain_group));

    Val* idx = idx_it->second;
    Val* start_idx =
        ir_utils::replaceValRecursively(idx, replacement_map_start);
    Val* stop_idx = ir_utils::replaceValRecursively(idx, replacement_map_stop);

    // Generate predicates as follows:
    //
    // (start_idx + start_offset) >= 0 &&
    // (stop_idx + stop_offset) < extent.

    PredicateInfo info;
    // For now, just set zero for both start and stop offsets by
    // assuming the domain is not partial.
    NVF_ERROR(!predicate_domain->maybePartial());
    info.start_offset_ = tv->fusion()->zeroVal();
    info.stop_offset_ = tv->fusion()->zeroVal();
    info.loop_stage_ = loop_stage;

    info.start_predicate_ = SimplifyingIrBuilder::geExpr(
        SimplifyingIrBuilder::addExpr(start_idx, info.start_offset_), zero_val);

    info.stop_predicate_ = SimplifyingIrBuilder::ltExpr(
        SimplifyingIrBuilder::addExpr(stop_idx, info.stop_offset_),
        actual_predicate_domain->extent());

    info.predicated_domains_ = actual_predicate_domains;

    // Set the used loop ID groups for this predicated domain
    const ValGroups& loop_deps =
        index_info.loop_group_dependencies.at(actual_predicate_domain_group);
    for (const auto& loop_dep : loop_deps) {
      info.loop_domains_.insert(loop_dep->front()->as<IterDomain>());
    }

    info_vec.emplace_back(info);
  }

  // Add predicates for non-divisible splits.
  // If this is a reduction init expr, then no need to take care of
  // non divisible splits
  if (!lower_utils::isReductionInitExpr(expr)) {
    for (const auto& [eg, direction] : index_info.traversal_path) {
      // NOTE: Fundamentally, the problem of non divisiblity should be
      // checked while traversing the indexing path. Currently, it uses
      // the information gathered in a tensor-by-tensor basis. This
      // should be fine currently, but may not work if, e.g., the
      // indexing path involved both backward and forward traversals.
      if (!isNonDivisibleSplit(eg)) {
        continue;
      }

      NVF_ERROR(eg->front()->isA<Split>());
      auto split_to_predicate = eg->front()->as<Split>();

      IterDomain* non_divisible_domain = split_to_predicate->in();
      const auto& non_divisible_domain_group =
          traversalGraph().toGroup(non_divisible_domain);

      PredicateInfo info;
      info.loop_stage_ = loop_stage;
      // The start predicate should always be true
      info.start_offset_ = zero_val;
      info.start_predicate_ = non_divisible_domain->fusion()->trueVal();

      info.stop_offset_ = zero_val;

      auto idx_it = index_map.find(non_divisible_domain_group);
      NVF_ERROR(
          idx_it != index_map.end(),
          "Index not found for non-divisible split domain: ",
          non_divisible_domain->toString());

      auto idx =
          ir_utils::replaceValRecursively(idx_it->second, replacement_map_stop);
      info.stop_predicate_ =
          SimplifyingIrBuilder::ltExpr(idx, non_divisible_domain->extent());
      info.predicated_domains_ = {non_divisible_domain};

      const ValGroups& loop_deps =
          index_info.loop_group_dependencies.at(non_divisible_domain_group);
      for (const auto& loop_dep : loop_deps) {
        info.loop_domains_.insert(loop_dep->front()->as<IterDomain>());
      }

      info_vec.emplace_back(info);
    }
  }

  return info_vec;
}

ExprPath<ExprGroup> TensorIndexer::getIndexingPath(
    const Expr* expr,
    const std::vector<IterDomain*>& index_ids) const {
  // Exclude broadcast IDs as their indices should always be zero
  // and they may not be reachable from the loop domain
  std::vector<IterDomain*> non_broadcast_index_ids;
  for (const auto index_id : index_ids) {
    if (!index_id->isBroadcast()) {
      non_broadcast_index_ids.push_back(index_id);
    }
  }

  return IndexingTraversal::getExprsBetween(
      expr,
      traversalGraph(),
      getLoopIds(expr, id_model_),
      non_broadcast_index_ids);
}

std::pair<std::vector<ValGroup>, std::vector<Val*>> TensorIndexer::
    getContigDomainsAndStrides(
        const AllocationDomainInfo& alloc_info,
        const ExprPath<ExprGroup>& traversal_path) const {
  const std::unordered_map<IterDomain*, ValGroup>& contig_domains =
      getContigDomains(
          alloc_info.ids,
          alloc_info.contiguity,
          reverse(traversal_path),
          traversalGraph(),
          /*is_predicate_pass=*/false);

  // Find contiguous domains to index
  std::unordered_set<ValGroup> already_indexed_domains;
  std::deque<ValGroup> contig_alloc_groups;
  std::deque<Val*> contig_strides;
  for (const auto i : c10::irange(alloc_info.ids.size())) {
    // Traverse back from the innermost domains so that the right
    // stride val is picked up for each contiguous domain
    auto i1 = alloc_info.ids.size() - 1 - i;
    IterDomain* allocation_domain = alloc_info.ids.at(i1);
    auto contig_domains_it = contig_domains.find(allocation_domain);
    NVF_ERROR(
        contig_domains_it != contig_domains.end(),
        "No contig domain mapping found for ",
        allocation_domain->toString());

    const ValGroup& contig_domain_group = contig_domains_it->second;
    if (already_indexed_domains.find(contig_domain_group) !=
        already_indexed_domains.end()) {
      continue;
    }
    already_indexed_domains.emplace(contig_domain_group);

    contig_alloc_groups.push_front(contig_domain_group);
    contig_strides.push_front(alloc_info.strides.at(i1));
  }

  return {
      {contig_alloc_groups.begin(), contig_alloc_groups.end()},
      {contig_strides.begin(), contig_strides.end()}};
}

ValGroups TensorIndexer::getUsedLoopGroups(
    const IndexingInfo& index_info) const {
  ValGroups used_loop_groups;
  for (const auto& index_id : index_info.index_ids) {
    const ValGroups& loop_groups = index_info.loop_group_dependencies.at(
        traversalGraph().toGroup(index_id));
    used_loop_groups.pushBack(loop_groups);
  }
  return used_loop_groups;
}

void TensorIndexer::ensureStaticIndexing(
    const std::vector<ForLoop*>& for_loops,
    const IndexingInfo& index_info) const {
  const ValGroups used_loop_groups = getUsedLoopGroups(index_info);

  for (auto for_loop : for_loops) {
    if (used_loop_groups.has(id_model_.idGraph(IdMappingMode::LOOP)
                                 .toGroup(for_loop->iter_domain()))) {
      for_loop->requireUnroll();
    }
  }
}

std::pair<std::vector<Val*>, std::vector<Val*>> TensorIndexer::
    getContigIndexFor(
        TensorView* tv,
        const Expr* expr,
        bool as_consumer,
        const AllocationDomainInfo& alloc_info,
        const std::vector<ForLoop*>& for_loops,
        const std::unordered_map<IterDomain*, Val*>& override_index) const {
  std::vector<IterDomain*> indexed_ids;
  indexed_ids.reserve(alloc_info.ids.size());
  for (const auto& id : alloc_info.ids) {
    if (!override_index.count(id)) {
      indexed_ids.push_back(id);
    }
  }
  auto index_info = computeIndex(expr, indexed_ids, for_loops);
  for (const auto& [indexed_id, index] : override_index) {
    index_info.index_map.emplace(traversalGraph().toGroup(indexed_id), index);
  }
  const auto& index_map = index_info.index_map;
  const auto& replacement_map = getIndexReplacementMap(
      expr, as_consumer, index_info.loop_ids, for_loops, index_map);

  std::vector<ValGroup> contig_alloc_groups;
  std::vector<Val*> contig_strides;

  if (isContigIndexingEnabled()) {
    const auto& contig_alloc_strides =
        getContigDomainsAndStrides(alloc_info, index_info.traversal_path);
    contig_alloc_groups = contig_alloc_strides.first;
    contig_strides = contig_alloc_strides.second;
  } else {
    std::transform(
        alloc_info.ids.begin(),
        alloc_info.ids.end(),
        std::back_inserter(contig_alloc_groups),
        [&](IterDomain* allocation_domain) {
          return traversalGraph().toGroup(allocation_domain);
        });
    contig_strides = {alloc_info.strides.begin(), alloc_info.strides.end()};
  }

  std::vector<Val*> result;
  result.reserve(contig_alloc_groups.size());

  for (const auto i : c10::irange(contig_alloc_groups.size())) {
    const auto& contig_domain_group = contig_alloc_groups.at(i);
    auto idx_it = index_map.find(contig_domain_group);
    NVF_ERROR(
        idx_it != index_map.end(),
        "Index not found for ",
        contig_domain_group->front()->toString());
    Val* idx = idx_it->second;
    Val* replaced_idx = ir_utils::replaceValRecursively(idx, replacement_map);
    result.push_back(replaced_idx);
  }

  // It's a bit confusing for the function named as "getContigIndexFor"
  // to change the property of ForLoops, but the local variable of
  // index_info is needed.
  if (tv->getMemoryType() == MemoryType::Local) {
    ensureStaticIndexing(for_loops, index_info);
  }

  return {result, contig_strides};
}

bool TensorIndexer::isSupported(Fusion* fusion) {
  const auto all_tvs = fusion->allTvs();

  auto warn = [](const std::string& reason) -> void {
#ifndef NDEBUG
    TORCH_WARN("TensorIndexer disabled due to: ", reason);
#endif // NDEBUG
  };

  // The following conditions are those that are known to be
  // unsupported. It may not be a complete list.

  if (fusion->hasManaged("loop_rotation")) {
    warn("loop rotation is not supported");
    return false;
  }

  for (const auto& tv : all_tvs) {
    std::stringstream reason;

    if (auto gather = dynamic_cast<GatherOp*>(tv->definition());
        gather != nullptr && !gather->exactSizes()) {
      // take_along_axis is supported but generic gather is not
      reason << "Non-exact gather not supported: " << gather->toString();
    } else if (tv->hasComputeWith()) {
      reason << "computeWith not supported: " << tv->toString();
    } else {
      for (const auto& id : tv->domain()->allIDs()) {
        if (auto swizzle2d = dynamic_cast<Swizzle2D*>(id->definition())) {
          reason << "Swizzle2D not supported: " << swizzle2d->toString();
          break;
        } else if (ir_utils::isIndexedConsumerID(tv, id)) {
          reason << "Indirect indexing of consumer ID not supported: "
                 << tv->toString() << ", " << id->toString() << ", "
                 << tv->definition()->toString();
          break;
        }
      }
    }

    if (!reason.str().empty()) {
      warn(reason.str());
      return false;
    }
  }

  return true;
}

} // namespace nvfuser
