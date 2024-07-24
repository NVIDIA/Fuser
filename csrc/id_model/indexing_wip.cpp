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
#include <id_model/predicate_indexing.h>
#include <id_model/to_string.h>
#include <id_model/utils.h>
#include <ir/utils.h>

namespace nvfuser {

using namespace indexing_utils;

std::pair<std::deque<ValGroup>, std::deque<Val*>> TensorIndexer::
    getContigDomainsAndStrides(
        const std::vector<IterDomain*>& allocation_domains,
        const std::vector<Val*>& strides,
        const std::vector<bool>& contiguity,
        const ExprPath<ExprGroup>& traversal_path) const {
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

std::vector<PredicateInfo> TensorIndexer::getPredicatesWIP(
    TensorView* tv,
    const Expr* expr,
    const std::vector<ForLoop*>& for_loops,
    ForLoop* unswitched_loop) const {
  bool is_unswitch = unswitched_loop != nullptr &&
      (unswitched_loop->iter_domain()->getParallelType() ==
           ParallelType::Unswitch ||
       unswitched_loop->iter_domain()->getParallelType() ==
           ParallelType::Unroll);

  if (is_unswitch) {
    VERBOSE() << "get unswitch predicates of " << tv->toString() << " in "
              << expr->toString();
  } else {
    VERBOSE() << "get inline predicates of " << tv->toString() << " in "
              << expr->toString();
  }

  const auto zero_val = tv->fusion()->zeroVal();

  const std::vector<IterDomain*>& predicate_domains =
      getPredicateDomains(tv, expr);

  // TODO: It should be safe to exit if predicate_domains is empty

  VERBOSE() << "Predicate domains: " << toDelimitedString(predicate_domains)
            << std::endl;

  const IndexingInfo& index_info = computeIndex(
      expr,
      traversalGraph().toGroups(predicate_domains),
      for_loops,
      is_unswitch);

  const std::unordered_map<ValGroup, Val*>& index_map = index_info.index_map;

  const std::unordered_map<Val*, Val*> replacement_map_start =
      getPredicateIndexReplacementMap(
          tv,
          for_loops,
          index_map,
          traversalGraph(),
          id_model_,
          /*is_start_predicate=*/true,
          /*unswitched_loop=*/unswitched_loop);

  const std::unordered_map<Val*, Val*> replacement_map_stop =
      getPredicateIndexReplacementMap(
          tv,
          for_loops,
          index_map,
          traversalGraph(),
          id_model_,
          /*is_start_predicate=*/false,
          /*unswitched_loop=*/unswitched_loop);

  auto non_divisible_splits = getNonDivisibleConsumerDomainsToPredicate(tv);

  const std::unordered_map<IterDomain*, ValGroup>& contig_domains =
      getContigDomains(
          predicate_domains,
          std::vector<bool>(predicate_domains.size(), true),
          reverse(index_info.traversal_path),
          traversalGraph(),
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

  const CircularBufferLoopStage loop_stage = getCircularBufferLoopStage(
      tv, for_loops, id_model_.idGraph(IdMappingMode::LOOP));

  std::vector<PredicateInfo> info_vec;
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

    auto idx_it = index_map.find(traversalGraph().toGroup(predicate_domain));
    if (enableContigIndexing()) {
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

    PredicateInfo info;
    // For now, just set zero for both start and stop offsets
    info.start_offset_ = zero_val;
    info.stop_offset_ = zero_val;
    info.loop_stage_ = loop_stage;

    // Use the same index for start and stop
    auto start_idx =
        ir_utils::replaceValRecursively(idx, replacement_map_start);
    info.start_predicate_ = SimplifyingIrBuilder::geExpr(
        SimplifyingIrBuilder::addExpr(start_idx, info.start_offset_), zero_val);

    auto stop_idx = ir_utils::replaceValRecursively(idx, replacement_map_stop);
    VERBOSE() << "Before replacement: " << idx->toInlineString()
              << " after: " << stop_idx->toInlineString() << std::endl;

    if (!enableContigIndexing()) {
      info.stop_predicate_ = SimplifyingIrBuilder::ltExpr(
          SimplifyingIrBuilder::addExpr(stop_idx, info.stop_offset_),
          predicate_domain->extent());
      info.predicated_domains_ = {predicate_domain};
    } else {
      info.stop_predicate_ = SimplifyingIrBuilder::ltExpr(
          SimplifyingIrBuilder::addExpr(stop_idx, info.stop_offset_),
          contig_domain_group->front()->as<IterDomain>()->extent());
      info.predicated_domains_ =
          getCoveredPredicatedDomains(contig_domain_group);
      VERBOSE() << "Contig covered root: "
                << toDelimitedString(info.predicated_domains_) << std::endl;
    }

    info_vec.emplace_back(info);
  }

  // Add predicates for non-divisible splits.
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

      PredicateInfo info;
      info.start_offset_ = zero_val;
      info.start_predicate_ = non_divisible_domain->fusion()->trueVal();
      info.stop_offset_ = zero_val;

      auto idx_it =
          index_map.find(traversalGraph().toGroup(non_divisible_domain));
      NVF_ERROR(
          idx_it != index_map.end(),
          "Index not found for non-divisible split domain: ",
          non_divisible_domain->toString());

      auto idx =
          ir_utils::replaceValRecursively(idx_it->second, replacement_map_stop);
      info.stop_predicate_ =
          SimplifyingIrBuilder::ltExpr(idx, non_divisible_domain->extent());
      info.predicated_domains_ = {non_divisible_domain};
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
      expr, traversalGraph().toGroups(index_domains), for_loops, false);

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
      for (const auto& id : tv->domain()->allIDs()) {
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
