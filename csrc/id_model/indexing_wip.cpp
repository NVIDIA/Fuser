// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/analysis/index_compute.h>
#include <device_lower/lower2device.h>
#include <id_model/circular_buffer_indexing.h>
#include <id_model/contiguity.h>
#include <id_model/indexing.h>
#include <id_model/indexing_utils.h>
#include <id_model/to_string.h>
#include <id_model/utils.h>
#include <ir/utils.h>

namespace nvfuser {

namespace {

// This is a temporary duplicate
// Get the promotion domain of a given loop domain.
IterDomain* getLoopPromotion(IterDomain* loop_id, const IdModel& id_model) {
  const auto& loop_graph = id_model.idGraph(IdMappingMode::LOOP);
  const auto& loop_promotion_map = id_model.loopPromotionMap();
  const auto& loop_group = loop_graph.toGroup(loop_id);

  auto loop_promotion_map_it = loop_promotion_map.find(loop_group);
  NVF_ERROR(
      loop_promotion_map_it != loop_promotion_map.end(),
      "No loop promotion found: ",
      loop_id->toString(),
      ". Loop group: ",
      nvfuser::toString(loop_group));

  return loop_promotion_map_it->second;
}

}

std::pair<std::deque<ValGroup>, std::deque<Val*>> TensorIndexer::
    getContigDomainsAndStrides(
        const std::vector<IterDomain*>& allocation_domains,
        const std::vector<Val*>& strides,
        const std::vector<bool>& contiguity,
        const ExprPath& traversal_path) const {
  const std::unordered_map<IterDomain*, ValGroup>& contig_domains =
      getContigDomains(
          allocation_domains,
          contiguity,
          reverse(traversal_path),
          traversalGraph(),
          concrete_info_,
          false);

  // Find contiguous domains to index
  std::unordered_set<ValGroup> already_indexed_domains;
  std::deque<ValGroup> contig_alloc_groups;
  std::deque<Val*> contig_strides;
  for (const auto i : c10::irange(allocation_domains.size())) {
    // Traverse back from the innermost domains so that the right
    // stride val is picked up for each contiguous domain
    auto i1 = allocation_domains.size() - 1 - i;
    IterDomain* allocation_domain = allocation_domains.at(i1);
    auto contig_domains_it = contig_domains.find(allocation_domain);
    NVF_ERROR(
        contig_domains_it != contig_domains.end(),
        "No contig domain mapping found for ",
        allocation_domain->toString());

    const ValGroup& contig_domain_group = contig_domains_it->second;
    if (already_indexed_domains.find(contig_domain_group) !=
        already_indexed_domains.end()) {
      VERBOSE() << "Already indexed: " << allocation_domain->toString()
                << std::endl;
      continue;
    }
    already_indexed_domains.emplace(contig_domain_group);

    if (!contig_domain_group->has(allocation_domain)) {
      VERBOSE() << "Contig indexing: "
                << contig_domain_group->front()->toString() << " instead of "
                << allocation_domain->toString() << std::endl;
    } else {
      VERBOSE() << "Non contig indexing: " << allocation_domain->toString()
                << std::endl;
    }

    VERBOSE() << "Stride: " << strides.at(i1)->toInlineString() << std::endl;

    contig_alloc_groups.push_front(contig_domain_group);
    contig_strides.push_front(strides.at(i1));
  }

  return {contig_alloc_groups, contig_strides};
}

std::unordered_map<Val*, Val*> TensorIndexer::getPredicateIndexReplacementMap(
    TensorView* tv,
    const std::vector<ForLoop*>& for_loops,
    bool is_start_predicate,
    bool is_unswitch,
    const std::unordered_map<ValGroup, Val*>& index_map,
    const ValGraph& traversal_graph) const {
  std::unordered_map<Val*, Val*> replacement_map;

  auto replace_for_unswitch =
      [&](ForLoop* fl, IterDomain* loop_id, bool within_unswitch) -> Val* {
    // Don't replace thread indices even when unswitched
    if (fl->iter_domain()->isThread() ||
        (fl->iter_domain()->getParallelType() != ParallelType::Vectorize &&
         !within_unswitch && !predicateAtEnd(fl))) {
      return nullptr;
    } else {
      return is_start_predicate
          ? fl->fusion()->zeroVal()
          : SimplifyingIrBuilder::subExpr(
                fl->simplifiedStop(), fl->fusion()->oneVal());
    }
  };

  auto replace_for_double_buffering = [&](ForLoop* fl,
                                          Val* original_index) -> Val* {
    auto db_axis =
        GpuLower::current()->circularBufferInfo().getCircularBufferAxis(tv);
    if (db_axis == nullptr ||
        !id_model_.idGraph(IdMappingMode::LOOP)
             .disjointValSets()
             .strictAreMapped(fl->iter_domain(), db_axis)) {
      return nullptr;
    }

    // The prologue loop does not need to be changed
    if (fl->circularBufferLoopStage() == CircularBufferLoopStage::Prolog) {
      return nullptr;
    }

    auto stage_depth =
        (int64_t)GpuLower::current()->circularBufferInfo().getStageDepthFor(
            fl->iter_domain());
    return SimplifyingIrBuilder::addExpr(
        original_index,
        SimplifyingIrBuilder::create<Val>(stage_depth - 1L, DataType::Index));
  };

  bool within_unswitch = false;

  for (const auto fl : for_loops) {
    auto parallel_type = fl->iter_domain()->getParallelType();

    if (parallel_type == ParallelType::Unswitch ||
        parallel_type == ParallelType::Unroll) {
      within_unswitch = is_unswitch;
    }

    auto loop_id = getLoopPromotion(fl->iter_domain(), id_model_);
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

    auto unswitched_index = replace_for_unswitch(fl, loop_id, within_unswitch);
    if (unswitched_index != nullptr) {
      replacement = unswitched_index;
    }

    // Adjustment for double buffering
    auto db_index = replace_for_double_buffering(fl, replacement);
    if (db_index != nullptr) {
      replacement = db_index;
    }

    if (replacement != loop_index) {
      auto inserted = replacement_map.emplace(loop_index, replacement).second;
      NVF_ERROR(
          inserted, "Duplicate replacement attempted: ", loop_id->toString());
      VERBOSE() << "Replacing initial index: " << loop_index->toInlineString()
                << " with " << replacement->toInlineString() << std::endl;
    }
  }

  return replacement_map;
}

namespace {

std::vector<IterDomain*> getPredicateDomains(
    TensorView* tv,
    const Expr* expr,
    const IdModel& id_model) {
  // TODO: Contig merged indexing

  // Rfactor domains should be the domains to predicate as they define
  // the logical shape of a tensor. However, in the case of rfactored
  // reductions, rfactor splits may not be divisible, thus root
  // domains need to be predicated. Note that the non-divisible split
  // info does not seem to cover non-divisible reduction rfactor
  // splits.
  std::vector<IterDomain*> predicate_domains =
      tv->hasReduction() ? tv->getMaybeRootDomain() : tv->getLogicalDomain();

  // Broadcast domains should not be predicated
  predicate_domains.erase(
      std::remove_if(
          predicate_domains.begin(),
          predicate_domains.end(),
          [](IterDomain* id) -> bool { return id->isBroadcast(); }),
      predicate_domains.end());

  // If this is an expr initializing a buffer for a reduction, the
  // reduction domains do not need to be predicated. In fact, if it's
  // a Local or Shared memory, no predicate is necessary
  if (lower_utils::isReductionInitExpr(expr)) {
    VERBOSE() << "Reduction init expr: " << expr->toString();
    if (isAllocationBasedOnLeaf(tv)) {
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
} // namespace

std::vector<RootPredicateInfo> TensorIndexer::getPredicates(
    TensorView* tv,
    const Expr* expr,
    const std::vector<ForLoop*>& for_loops,
    bool is_unswitch) {
  if (is_unswitch) {
    VERBOSE() << "get unswitch predicates of " << tv->toString() << " in "
              << expr->toString();
  } else {
    VERBOSE() << "get inline predicates of " << tv->toString() << " in "
              << expr->toString();
  }

  // For a double buffered tensor, use the predicate from the main
  // loop only. The prologue loop is only for the first element, so it
  // should be ignored. Double buffering may or may not create the
  // eplogue loop, but irrespective of that we can just use the
  // predicate of the main loop.
  if (is_unswitch) {
    if (auto loop_stage = getCircularBufferLoopStage(
            tv, for_loops, id_model_.idGraph(IdMappingMode::LOOP));
        loop_stage.has_value() &&
        loop_stage.value() != CircularBufferLoopStage::Main) {
      return {};
    }
  }

  const auto& traversal_graph = id_model_.idGraph(IdMappingMode::ALMOSTEXACT);
  const auto zero_val = tv->fusion()->zeroVal();

  const auto& predicate_domains = getPredicateDomains(tv, expr, id_model_);

  VERBOSE() << "Predicate domains: " << toDelimitedString(predicate_domains)
            << std::endl;

  const auto& index_info = computeIndex(
      expr,
      for_loops,
      traversalGraph().toGroups(predicate_domains),
      true,
      is_unswitch);
  const auto& index_map = index_info.index_map;

  auto replacement_map_start = getPredicateIndexReplacementMap(
      tv, for_loops, true, is_unswitch, index_map, traversal_graph);

  auto replacement_map_stop = getPredicateIndexReplacementMap(
      tv, for_loops, false, is_unswitch, index_map, traversal_graph);

  auto non_divisible_splits = getNonDivisibleConsumerDomainsToPredicate(tv);

  const std::unordered_map<IterDomain*, ValGroup>& contig_domains =
      getContigDomains(
          predicate_domains,
          std::vector<bool>(predicate_domains.size(), true),
          reverse(index_info.traversal_path),
          traversal_graph,
          concrete_info_,
          true);

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

  std::vector<RootPredicateInfo> info_vec;
  info_vec.reserve(predicate_domains.size() + non_divisible_splits.size());
  std::unordered_set<ValGroup> already_indexed_domains;

  for (const auto& predicate_domain : predicate_domains) {
    auto contig_domains_it = contig_domains.find(predicate_domain);
    NVF_ERROR(
        contig_domains_it != contig_domains.end(),
        "No contig domain mapping found for ",
        predicate_domain->toString());
    const ValGroup& contig_domain_group = contig_domains_it->second;

    VERBOSE() << "Predicate domain: " << predicate_domain->toString()
              << ", contig domain: " << contig_domain_group->front()->toString()
              << std::endl;

    auto idx_it = index_map.find(traversal_graph.toGroup(predicate_domain));
    if (!getenv("DISABLE_CONTIG_INDEXING")) {
      if (already_indexed_domains.find(contig_domain_group) !=
          already_indexed_domains.end()) {
        VERBOSE() << "Already indexed: " << predicate_domain->toString()
                  << std::endl;
        continue;
      }
      already_indexed_domains.emplace(contig_domain_group);

      if (!contig_domain_group->has(predicate_domain)) {
        VERBOSE() << "Contig predication: "
                  << contig_domain_group->front()->toString() << " instead of "
                  << predicate_domain->toString()
                  << ". Tensor: " << tv->toString() << std::endl;
      }

      // auto idx_it =
      // index_map.find(traversal_graph.toGroup(predicate_domain));
      idx_it = index_map.find(contig_domain_group);
    }
    NVF_ERROR(
        idx_it != index_map.end(),
        "Index not found for ",
        contig_domain_group->front()->toString());
    Val* idx = idx_it->second;
    VERBOSE() << "Predicate index of " << predicate_domain->toString() << ": "
              << idx->toInlineString() << ", unswitch? : " << is_unswitch
              << std::endl;

    RootPredicateInfo info;
    // For now, just set zero for both start and stop offsets
    info.start_offset_ = zero_val;
    info.stop_offset_ = zero_val;

    // Use the same index for start and stop
    auto start_idx =
        ir_utils::replaceValRecursively(idx, replacement_map_start);
    info.start_predicate_ = SimplifyingIrBuilder::geExpr(
        SimplifyingIrBuilder::addExpr(start_idx, info.start_offset_), zero_val);

    // TODO: predicate elimination
    auto stop_idx = ir_utils::replaceValRecursively(idx, replacement_map_stop);
    VERBOSE() << "Before replacement: " << idx->toInlineString()
              << " after: " << stop_idx->toInlineString() << std::endl;

    if (getenv("DISABLE_CONTIG_INDEXING")) {
      info.stop_predicate_ = SimplifyingIrBuilder::ltExpr(
          SimplifyingIrBuilder::addExpr(stop_idx, info.stop_offset_),
          predicate_domain->extent());
      info.root_ids_ = {predicate_domain};
    } else {
      info.stop_predicate_ = SimplifyingIrBuilder::ltExpr(
          SimplifyingIrBuilder::addExpr(stop_idx, info.stop_offset_),
          contig_domain_group->front()->as<IterDomain>()->extent());
      info.root_ids_ = getCoveredPredicatedDomains(contig_domain_group);
      VERBOSE() << "Contig covered root: " << toDelimitedString(info.root_ids_)
                << std::endl;
    }

    info_vec.emplace_back(info);
  }

  // If this is a reduction init expr, then no need to take care of
  // non divisible splits
  if (!lower_utils::isReductionInitExpr(expr)) {
    for (const auto& [eg, direction] : index_info.traversal_path) {
      if (!isNonDivisibleSplit(eg)) {
        continue;
      }

      NVF_ERROR(eg->front()->isA<Split>());
      auto split_to_predicate = eg->front()->as<Split>();
      VERBOSE() << "Non-divisible predicate: "
                << split_to_predicate->toString();

      IterDomain* non_divisible_domain = split_to_predicate->in();

      RootPredicateInfo info;
      info.start_offset_ = zero_val;
      info.start_predicate_ = non_divisible_domain->fusion()->trueVal();
      info.stop_offset_ = zero_val;

      auto idx_it =
          index_map.find(traversal_graph.toGroup(non_divisible_domain));
      NVF_ERROR(
          idx_it != index_map.end(),
          "Index not found for non-divisible split domain: ",
          non_divisible_domain->toString());

      auto idx =
          ir_utils::replaceValRecursively(idx_it->second, replacement_map_stop);
      info.stop_predicate_ =
          SimplifyingIrBuilder::ltExpr(idx, non_divisible_domain->extent());
      VERBOSE() << "Precicate: " << info.stop_predicate_->toInlineString()
                << std::endl;
      info.root_ids_ = {non_divisible_domain};
      info_vec.emplace_back(info);
    }
  }

  return info_vec;
}

// TODO: Drop the tv parameter. It's only for double buffering, which
// I believe should be done as a separate step after indexing
std::vector<Val*> TensorIndexer::getPerDimIndex(
    TensorView* tv,
    const std::vector<IterDomain*>& index_domains,
    const Expr* expr,
    const std::vector<ForLoop*>& for_loops) {
  const auto& traversal_graph = id_model_.idGraph(IdMappingMode::ALMOSTEXACT);

  VERBOSE() << "getPerDimIndex of " << toDelimitedString(index_domains)
            << " in " << expr->toString() << std::endl;

  const auto& index_info = computeIndex(
      expr, for_loops, traversalGraph().toGroups(index_domains), false, false);

  const auto& index_map = index_info.index_map;

  std::vector<Val*> indices;
  indices.reserve(index_domains.size());

  for (const auto i : c10::irange(index_domains.size())) {
    auto index_domain = index_domains.at(i);

    if (index_domain->isBroadcast() || index_domain->isReduction()) {
      indices.push_back(index_domain->fusion()->zeroVal());
      continue;
    }

    auto idx_it = index_map.find(traversal_graph.toGroup(index_domain));
    NVF_ERROR(
        idx_it != index_map.end(),
        "Index not found for ",
        index_domain->toString());
    Val* idx = idx_it->second;
    VERBOSE() << "Index of " << index_domain->toString() << ": "
              << idx->toInlineString() << std::endl;

    indices.push_back(idx);
  }

  return indices;
}

bool TensorIndexer::isSupported(Fusion* fusion) {
  const auto all_tvs = ir_utils::allTvs(fusion);

  auto printReason = [](const std::string& reason) -> void {
    VERBOSE() << "TensorIndexer disabled due to: " << reason << std::endl;
  };

  if (fusion->hasManaged("loop_rotation")) {
    printReason("loop rotation is not supported");
    return false;
  }

  for (const auto& tv : all_tvs) {
    std::stringstream reason;

    if (auto loadstore = dynamic_cast<LoadStoreOp*>(tv->definition());
        loadstore != nullptr &&
        (loadstore->opType() == LoadStoreOpType::LdMatrix)) {
      // loadstore->opType() == LoadStoreOpType::CpAsync ||
      // loadstore->opType() == LoadStoreOpType::CpAsyncBulkTensorTile)) {
      reason << "LoadStoreOp not supported: " << loadstore->toString();
    } else {
      for (const auto& id : ir_utils::allIDsOf(tv)) {
        if (id->getParallelType() == ParallelType::MisalignedVectorize) {
          reason << "MialignedVectorize is used: " << id->toString();
          break;
        } else if (auto swizzle = dynamic_cast<Swizzle*>(id->definition())) {
          reason << "Swizzle not supported: " << swizzle->toString();
          break;
        } else if (
            auto swizzle2d = dynamic_cast<Swizzle2D*>(id->definition())) {
          reason << "Swizzle2D not supported: " << swizzle2d->toString();
          break;
        } else if (ir_utils::isIndexedID(tv, id)) {
          reason << "Index ops such as select not supported: "
                 << tv->toString();
          break;
        }
      }
    }

    if (!reason.str().empty()) {
      printReason(reason.str());
      return false;
    }
  }

  return true;
}

} // namespace nvfuser
