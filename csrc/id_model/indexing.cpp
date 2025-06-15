// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <device_lower/analysis/index_compute.h>
#include <device_lower/analysis/non_divisible_split.h>
#include <device_lower/lower2device.h>
#include <device_lower/pass/magic_zero.h>
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
    }

    if (!tv_output->getAlternateLoopDomain().has_value()) {
      continue;
    }

    const std::vector<IterDomain*>& alternate_loop_domain =
        tv_output->getAlternateLoopDomain().value();
    const std::vector<IterDomain*>& loop_domain = tv_output->getLoopDomain();
    // NOTE For scheduling ldmatrix and stmatrix, the assumption is the original
    // and alternate loop domains have the same number of iterDomains. This
    // assertion may not be strictly necessary.
    NVF_ERROR(alternate_loop_domain.size() == loop_domain.size());
    for (auto&& [alt_loop_id, loop_id] :
         zip(alternate_loop_domain, loop_domain)) {
      const ValGroup& alt_loop_group =
          id_model_.idGraph(IdMappingMode::LOOP).toGroup(alt_loop_id);
      if (loop_index_map_.find(alt_loop_group) != loop_index_map_.end()) {
        // Index already assigned for alternate loop iterDomain
        continue;
      }
      // Map alternate loop iterDomain to the index variable for the original
      // loop iterDomain.
      const ValGroup& loop_group =
          id_model_.idGraph(IdMappingMode::LOOP).toGroup(loop_id);
      auto loop_index_iter = loop_index_map_.find(loop_group);
      NVF_ERROR(loop_index_iter != loop_index_map_.end());
      loop_index_map_[alt_loop_group] = loop_index_iter->second;
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
    const std::vector<ForLoop*>& for_loops,
    bool use_magic_zero) const {
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

  if (use_magic_zero) {
    result = protectIndicesWithMagicZero(result, for_loops);
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

  const auto& alloc_info = getIndexAllocationInfo(tv);

  const auto [contig_indices, contig_strides] = getContigIndexFor(
      tv, expr, as_consumer, alloc_info, for_loops, override_index);

  // Linearize the indices with strides.
  Val* linear_index = tv->fusion()->zeroVal();
  for (const auto i : arange(contig_indices.size())) {
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

  if (tv->getMemoryType() == MemoryType::Global) {
    linear_index = protectIndicesWithMagicZero({linear_index}, for_loops).at(0);
  }

  if (tv->getMemoryType() == MemoryType::Local) {
    ensureStaticIndexing(for_loops, linear_index);
  }

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
    const std::vector<ForLoop*>& for_loops,
    bool use_alternate_loop_domain) const {
  const auto loop_ids = getLoopIds(expr, id_model_, use_alternate_loop_domain);
  const ExprPath<ExprGroup> traversal_path =
      getIndexingPath(expr, index_ids, use_alternate_loop_domain);
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

ValGroups TensorIndexer::getNonDivisibleIdsToPredicate(
    TensorView* tv,
    const IndexingInfo& index_info) const {
  const auto& non_div_info = GpuLower::current()->nonDivisiblePredicateInfo();

  ValGroups ids_to_predicate;

  if (auto it = non_div_info.idsToPredicate().find(tv);
      it != non_div_info.idsToPredicate().end()) {
    ids_to_predicate = it->second;
  }

  // Make sure all IDs have indices
  for (const auto& id_group : ids_to_predicate) {
    NVF_ERROR(
        index_info.index_map.contains(id_group),
        "Index not found for non-divisible predicate: ",
        nvfuser::toString(id_group));
  }

  // In addition to splits in the indexing traversal path, there can
  // be additional splits that need to be predicated. For example,
  //
  // auto tv0 = makeContigConcreteTensor({8});
  // fusion.addInput(tv0);
  // auto tv1 = sum(tv0, {0});
  // fusion.addOutput(tv1);
  //
  // [r0(8)]
  // tv1->split(0, 1);
  // [r1(8), r2(1)]
  // tv1->split(1, 4);
  // [r1(8), r3(1), r4(4)]
  //
  // The predicate of tv1 is generated by the index of r0, which is
  // mapped with r1. So, the predicate would just be `i0 < 8`, where
  // i0 is the loop index of the outermost loop. However, this is not
  // enough because of the innermost loop of size 4. Suppose its loop
  // index is i2, the tensor also needs to be predicated with `i2 <
  // 1`. This is because the split of r2 is not divisible. This
  // non-divisible split would be automatically included if indexing
  // were done using the exact graph since it'd be included in the
  // indexing traversal path. However, since indexing uses the
  // almost-exact graph, no traversal is actually necessary as r0 is
  // mapped with one of the loop IDs, r1.
  //
  // The reason certain non-divisible splits are missed is because
  // size-one IDs, like r2 in the above example, may not need to be
  // traversed in the almost-exact graph. In order to find such IDs,
  // we visit each ID group on the path and see if it has a pattern
  // like r0, that is, it is used by a split of a factor 1 and one of
  // its outputs is in the same group in the almost-exact graph. For
  // each of the splits, the size-one output ID is potentially
  // problematic as it may be further split in a non-divisible way and
  // the ID is not in the indexing path, those non-divisible splits
  // are never detected by NonDivisibleSplitInfo.
  //
  // See PredicateIndexingTest.AdditionalNonDivisibleSplit and
  // PredicateIndexingTest.AdditionalNonDivisibleSplitAfterDivisibleSplit
  // for concrete examples.

  // The first step here is to find this pattern and gather all
  // potentially problematic IDs.

  // Grab all involved ID groups.
  auto from_groups = traversalGraph().toGroups(index_info.loop_ids);
  auto to_groups = traversalGraph().toGroups(index_info.index_ids);
  std::vector<ValGroup> all_visited_groups = getValsBetween<ValGraphBFS>(
      index_info.traversal_path,
      from_groups.vector(),
      to_groups.vector(),
      traversalGraph());

  // For each of the visited groups, look for the pattern like the
  // above split of r0 to r1 and r2. Specifically, we want to find an
  // ID that is split and one of the outputs is mapped with the split
  // input in the AlmostExact graph.

  // It's conceptually easier to use the Exact graph.
  const auto& exact_graph = id_model_.idGraph(IdMappingMode::EXACT);

  // All potentially problematic IDs
  ValGroups exact_groups_to_check;

  for (const auto& almost_exact_group : all_visited_groups) {
    // Find all Exact groups included in the AlmostExact group
    ValGroups covered_exact_groups;
    for (const auto& val : *almost_exact_group) {
      // Additional IDs may be created without getting added to the
      // exact graph, e.g., IDs for TMA and TMem, so
      // exact_graph.toGroup may fail. Should be safe to ignore them.
      if (exact_graph.hasGroup(val)) {
        covered_exact_groups.pushBack(exact_graph.toGroup(val));
      }
    }

    // If all of the IDs are exact mapped, this node should not need
    // to be examined further
    if (covered_exact_groups.size() == 1) {
      continue;
    }

    // Look for an exact ID group that is used by a split and one of
    // the outputs is mapped in the almost-exact graph
    for (const auto& covered_exact_group : covered_exact_groups) {
      for (const auto& use_eg : exact_graph.getUses(covered_exact_group)) {
        auto split = dynamic_cast<Split*>(use_eg->front());
        if (split == nullptr) {
          continue;
        }
        bool inner_mapped = almost_exact_group->has(split->inner());
        bool outer_mapped = almost_exact_group->has(split->outer());

        NVF_ERROR(
            !inner_mapped || !outer_mapped,
            "Both outputs of a split are mapped with the input");

        if (!inner_mapped && !outer_mapped) {
          continue;
        }

        // This corresponds to r2 in the above example
        IterDomain* unmapped_output = inner_mapped
            ? split->outer()->as<IterDomain>()
            : split->inner()->as<IterDomain>();

        // The unmapped output should be size one.
        NVF_ERROR(unmapped_output->extent()->isOneInt());

        // If there's no use, there's nothing to predicate
        if (exact_graph.getUses(exact_graph.toGroup(unmapped_output)).empty()) {
          continue;
        }

        exact_groups_to_check.pushBack(exact_graph.toGroup(unmapped_output));
      }
    }
  }

  // For each r2-like ID, check if there's any non-divisible split
  // in the path from the loop IDs. Not all of them may need to be
  // predicated. Similar to what NonDivisiblePredicateInfo does, we
  // should be able to minimize IDs to predicate, but here for
  // simplicity all of non-divisible splits are predicated
  for (const auto& exact_group_to_check : exact_groups_to_check) {
    const auto path = ValGraphPermissiveBFS::getExprGroupsBetween(
                          exact_graph,
                          {exact_group_to_check},
                          exact_graph.toGroups(index_info.loop_ids),
                          /*require_all_to_visited=*/false)
                          .first;

    for (const auto& [expr_g, dir] : path) {
      auto split = dynamic_cast<Split*>(expr_g->front());
      if (split == nullptr) {
        continue;
      }

      // Note that the above traversal is from group_to_check to the
      // loop domain, so the actual indexing direction is the opposite
      if (dir == Direction::Backward) {
        continue;
      }

      if (GpuLower::current()->divisibleSplitSet().contains(split) ||
          simplifyExpr(split->isDivisible())->isTrue()) {
        continue;
      }

      ids_to_predicate.pushBack(traversalGraph().toGroup(split->in()));
    }
  }

  return ids_to_predicate;
}

void TensorIndexer::updateIndexInfoForNonDivisibleSplits(
    const Expr* expr,
    const std::vector<ForLoop*>& for_loops,
    const ValGroups& non_divisible_ids,
    IndexingInfo& index_info) const {
  // Non-divisible split input IDs need to be predicated but may not
  // be included in the traversal path. Grab all IDs that are
  // currently missing indices and do another traversal

  if (non_divisible_ids.empty()) {
    return;
  }

  auto& index_map = index_info.index_map;

  std::vector<IterDomain*> additional_ids_to_predicate;
  for (const auto& id_group : non_divisible_ids) {
    auto idx_it = index_map.find(id_group);
    if (idx_it != index_map.end()) {
      continue;
    }

    additional_ids_to_predicate.push_back(id_group->front()->as<IterDomain>());
  }

  const auto additional_index_info =
      computeIndex(expr, additional_ids_to_predicate, for_loops);

  // Merge additional_index_info.index_map to index_info.index_map
  index_map.insert(
      additional_index_info.index_map.begin(),
      additional_index_info.index_map.end());

  index_info.loop_group_dependencies.insert(
      additional_index_info.loop_group_dependencies.begin(),
      additional_index_info.loop_group_dependencies.end());
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

  IndexingInfo index_info = computeIndex(expr, predicate_domains, for_loops);

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

  auto protectPredicatesWithMagicZero = [&](PredicateInfo& info) {
    if (info.startPredicate() != nullptr) {
      info.startPredicate() =
          protectIndicesWithMagicZero({info.startPredicate()}, for_loops).at(0);
    }
    if (info.stopPredicate() != nullptr) {
      info.stopPredicate() =
          protectIndicesWithMagicZero({info.stopPredicate()}, for_loops).at(0);
    }
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

    protectPredicatesWithMagicZero(info);

    info_vec.emplace_back(info);
  }

  // Add predicates for non-divisible splits.
  // If this is a reduction init expr, then no need to take care of
  // non divisible splits
  if (!lower_utils::isReductionInitExpr(expr)) {
    const auto non_divisible_ids =
        getNonDivisibleIdsToPredicate(tv, index_info);

    updateIndexInfoForNonDivisibleSplits(
        expr, for_loops, non_divisible_ids, index_info);

    for (const ValGroup& id_group_to_predicate : non_divisible_ids) {
      IterDomain* id_to_predicate =
          id_group_to_predicate->front()->as<IterDomain>();
      PredicateInfo info;
      info.loop_stage_ = loop_stage;
      // The start predicate should always be true
      info.start_offset_ = zero_val;
      info.start_predicate_ = id_to_predicate->fusion()->trueVal();

      info.stop_offset_ = zero_val;

      auto idx_it = index_map.find(id_group_to_predicate);
      NVF_ERROR(
          idx_it != index_map.end(),
          "Index not found for non-divisible ID group: ",
          id_group_to_predicate->front()->toString());

      auto idx =
          ir_utils::replaceValRecursively(idx_it->second, replacement_map_stop);
      info.stop_predicate_ =
          SimplifyingIrBuilder::ltExpr(idx, id_to_predicate->extent());
      info.predicated_domains_ = {id_to_predicate};

      const ValGroups& loop_deps =
          index_info.loop_group_dependencies.at(id_group_to_predicate);
      for (const auto& loop_dep : loop_deps) {
        info.loop_domains_.insert(loop_dep->front()->as<IterDomain>());
      }

      protectPredicatesWithMagicZero(info);

      info_vec.emplace_back(info);
    }
  }

  return info_vec;
}

ExprPath<ExprGroup> TensorIndexer::getIndexingPath(
    const Expr* expr,
    const std::vector<IterDomain*>& index_ids,
    bool use_alternate_loop_domain) const {
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
      getLoopIds(expr, id_model_, use_alternate_loop_domain),
      non_broadcast_index_ids);
}

ExprPath<ExprGroup> TensorIndexer::getPredicateIndexingPath(
    TensorView* tv,
    const Expr* expr) const {
  const std::vector<IterDomain*>& predicate_domains =
      getPredicateDomains(tv, expr);
  return getIndexingPath(expr, predicate_domains);
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
  for (const auto i : arange(alloc_info.ids.size())) {
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

std::vector<ForLoop*> TensorIndexer::getUsedForLoopsOf(
    const std::vector<Val*>& indices,
    const std::vector<ForLoop*>& for_loops) const {
  // Grab the loop indices
  std::vector<Val*> loop_indices;
  loop_indices.reserve(for_loops.size());
  for (auto for_loop : for_loops) {
    Val* initial_loop_index = getLoopIndex(for_loop->iter_domain(), for_loops);
    loop_indices.push_back(initial_loop_index);
  }

  // Figure out which loop indices are used in index
  const auto dep_vals = DependencyCheck::getAllValsBetween(
      {loop_indices.begin(), loop_indices.end()}, indices);

  std::vector<ForLoop*> dep_loops;
  for (auto [i, for_loop] : enumerate(for_loops)) {
    auto initial_loop_index = loop_indices.at(i);
    if (std::find(dep_vals.begin(), dep_vals.end(), initial_loop_index) !=
        dep_vals.end()) {
      dep_loops.push_back(for_loop);
    }
  }

  return dep_loops;
}

void TensorIndexer::ensureStaticIndexing(
    const std::vector<ForLoop*>& for_loops,
    Val* index) const {
  for (auto for_loop : getUsedForLoopsOf({index}, for_loops)) {
    for_loop->requireUnroll();
  }
}

namespace {

// Use alternate loop domain for the shared memory tensor for ldmatrix and
// stmatrix.
bool isSharedMemoryTvForLdStMatrix(TensorView* tv, const Expr* expr) {
  // short-circuit: not (ldmatrix or stmatrix)
  if (!ir_utils::isLdMatrixOp(expr) && !ir_utils::isStMatrixOp(expr)) {
    return false;
  }
  // short-circuit: only the shared memory TensorView uses alternate loop
  // domain. For ldmatrix, it is the input TensorView. For stmatrix, it is the
  // output TensorView.
  if (tv->getMemoryType() != MemoryType::Shared) {
    return false;
  }

  TensorView* output_tv = ir_utils::getTvOutput(expr);
  NVF_ERROR(output_tv != nullptr);

  // alternate_loop_domain is optional for now.
  return output_tv->getAlternateLoopDomain().has_value();
}

} // namespace

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
  auto index_info = computeIndex(
      expr, indexed_ids, for_loops, isSharedMemoryTvForLdStMatrix(tv, expr));
  for (const auto& [indexed_id, index] : override_index) {
    index_info.index_map.emplace(traversalGraph().toGroup(indexed_id), index);
  }
  const auto& index_map = index_info.index_map;
  auto replacement_map = getIndexReplacementMap(
      expr, as_consumer, index_info.loop_ids, for_loops, index_map);

  // War for MmaOp. The allocation domain may involve parallelized
  // IDs, either directly or by traversal. Ideally, we should set the
  // right allocation domain, but this seems to be a good enough WAR.
  if (expr->isA<MmaOp>() && tv->getMemoryType() == MemoryType::Local &&
      !as_consumer) {
    // Replace the indices of parallelized loop IDs with zero
    for (const auto loop_id : index_info.loop_ids) {
      if (isParallelTypeThread(loop_id->getParallelType())) {
        Val* loop_index = getLoopIndex(loop_id, for_loops);
        replacement_map.emplace(loop_index, expr->fusion()->zeroVal());
      }
    }
  }

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

  for (const auto i : arange(contig_alloc_groups.size())) {
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

  return {result, contig_strides};
}

std::vector<Val*> TensorIndexer::protectIndicesWithMagicZero(
    const std::vector<Val*>& indices,
    const std::vector<ForLoop*>& for_loops) const {
  if (!GpuLower::current()->isNvFuserZeroEnabled()) {
    return indices;
  }

  auto used_for_loops = getUsedForLoopsOf(indices, for_loops);

  for (const auto for_loop : used_for_loops | std::views::reverse) {
    Val* initial_loop_index = getLoopIndex(for_loop->iter_domain(), for_loops);

    if (!needsMagicZero(
            for_loop, for_loop->iter_domain(), initial_loop_index)) {
      continue;
    }

    std::unordered_map<Val*, Val*> replacement_map;
    replacement_map.emplace(
        initial_loop_index,
        SimplifyingIrBuilder::addExpr(
            initial_loop_index, GpuLower::current()->kernel()->magicZeroVal()));

    std::vector<Val*> protected_indices;
    protected_indices.reserve(indices.size());
    for (const auto index : indices) {
      auto protected_index =
          ir_utils::replaceValRecursively(index, replacement_map);
      protected_indices.push_back(protected_index);
    }
    return protected_indices;
  }

  return indices;
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
