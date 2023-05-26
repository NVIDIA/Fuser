// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/utils.h>
#include <ir/utils.h>
#include <scheduler/pointwise_utils.h>
#include <utils.h>

namespace nvfuser {
namespace pointwise_utils {

namespace {

// Grab all exact set mappings from consumer to producer domains of
// indexed accesses, e.g., index_select
std::unordered_map<
    std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>,
    std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
getIndexedConsumerToProducerMap(Fusion* fusion, const ComputeAtMap& ca_map) {
  std::unordered_map<
      std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>,
      std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      indexed_id_map;

  for (auto expr : fusion->exprs()) {
    if (auto gather = dynamic_cast<TorchGatherOp*>(expr)) {
      auto p_id = gather->getIndexedID();
      auto c_id = gather->getConsumerOfIndexedID();
      TORCH_INTERNAL_ASSERT(
          indexed_id_map
              .emplace(
                  ca_map.disjointSetOf(c_id, IdMappingMode::EXACT),
                  ca_map.disjointSetOf(p_id, IdMappingMode::EXACT))
              .second);
    } else if (auto index_select = dynamic_cast<IndexSelectOp*>(expr)) {
      auto p_id = index_select->getIndexedID();
      auto c_id = index_select->getConsumerOfIndexedID();
      TORCH_INTERNAL_ASSERT(
          indexed_id_map
              .emplace(
                  ca_map.disjointSetOf(c_id, IdMappingMode::EXACT),
                  ca_map.disjointSetOf(p_id, IdMappingMode::EXACT))
              .second);
    } else {
      // Note there's no consumer ID for select. This means we can't
      // just propagate from consumers to indexed producers. It seems
      // it's necessary to schedule producers and consumers separately
      // in those cases.
      continue;
    }
  }

  return indexed_id_map;
}

// Check if a root ID of a fusion input tensor that is indirectly
// accessed by ops such as torch_gather needs to be mapped with
// a reference tensor. Select has a similar effect as squeeze as the
// indexed domain is removed, so the domain does not need to be mapped
// as long as the tensor is a fusion input. Similarly, in index_select
// and torch_gather, if the output domain is a broadcast, it does not
// need to be mapped if not resolved.
bool canIgnoreIndexedInputDomainID(
    TensorView* input_tv,
    IterDomain* root_id,
    const ComputeAtMap& ca_map) {
  TORCH_INTERNAL_ASSERT(input_tv->isFusionInput());
  for (auto use : input_tv->uses()) {
    if (auto select = dynamic_cast<SelectOp*>(use)) {
      if (root_id != select->getIndexedID()) {
        return false;
      }
    } else if (auto index_select = dynamic_cast<IndexSelectOp*>(use)) {
      // If the root_id is an indexed ID, and the consumer ID may be a
      // broadcast. In that case, nothing needs to be mapped if the
      // consumer broadcast is not resolved
      if (root_id != index_select->getIndexedID() ||
          !ca_map
               .getConcreteMappedID(
                   index_select->getConsumerOfIndexedID(),
                   IdMappingMode::PERMISSIVE)
               ->isBroadcast()) {
        return false;
      }
    } else if (auto gather = dynamic_cast<TorchGatherOp*>(use)) {
      // TODO: Remove this. Once slice is used for torch_gather, this
      // should not be necessary. For now, it is necessary to not
      // break the existing torch_gather tests
      if (!gather->exactSizes()) {
        continue;
      }
      // If the root_id is an indexed ID, and the consumer ID may be a
      // broadcast. In that case, nothing needs to be mapped if the
      // consumer broadcast is not resolved
      if (root_id != gather->getIndexedID() ||
          !ca_map
               .getConcreteMappedID(
                   gather->getConsumerOfIndexedID(), IdMappingMode::PERMISSIVE)
               ->isBroadcast()) {
        return false;
      }
    } else {
      // If the input TV is used by any other ops
      return false;
    }
  }

  return true;
}

} // namespace

DomainMap::DomainMap(Fusion* fusion) : fusion_(fusion), ca_map_(fusion) {
  tvs_with_rfactor_ = scheduler_utils::getTVsWithNonReductionRFactor(fusion);
}

// Determine if all IterDomains in input are mapped to the given tensor
bool DomainMap::areAllInputIdsMappedTo(TensorView* input_tv, TensorView* tv)
    const {
  // Get concrete IDs for input root or rfactor domain
  std::unordered_set<IterDomain*> in_concrete_ids;
  for (auto in_id : input_tv->getMaybeRFactorDomain()) {
    if (canIgnoreIndexedInputDomainID(input_tv, in_id, ca_map_)) {
      continue;
    }

    // Permissive map is required for the transpose scheduler to support cases
    // like T0[I0, b] + T1[b, I1]
    auto concrete =
        ca_map_.getConcreteMappedID(in_id, IdMappingMode::PERMISSIVE);

    if (!concrete->isBroadcast() && !in_id->isReduction()) {
      in_concrete_ids.insert(concrete);
    }
  }

  // Erase all input concrete IDs mapped to the output domain
  // Ignore unresolved broadcast dimensions
  eraseifInputMappedThroughRFactorDomainAndIndexing(
      in_concrete_ids, tv->getMaybeRFactorDomain());

  return in_concrete_ids.empty();
}

// Erase input concrete ID if it is mapped to output ID
bool DomainMap::eraseIfMapped(
    std::unordered_set<IterDomain*>& in_concrete_ids,
    IterDomain* out_id) const {
  auto out_concrete_id =
      ca_map_.getConcreteMappedID(out_id, IdMappingMode::PERMISSIVE);
  auto in_concrete_id_iter = in_concrete_ids.find(out_concrete_id);
  bool found_match = in_concrete_id_iter != in_concrete_ids.end();
  if (found_match) {
    in_concrete_ids.erase(in_concrete_id_iter);
  }
  return found_match;
}

void DomainMap::eraseifInputMappedThroughRFactorDomainAndIndexing(
    std::unordered_set<IterDomain*>& in_ids,
    const std::vector<IterDomain*>& ids) const {
  // Use ComputeAtMap::getAllDisjointSetProducers to grab all producer
  // IDs through rfactor exprs
  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      exact_sets;
  std::for_each(ids.begin(), ids.end(), [&](IterDomain* id) {
    exact_sets.pushBack(ca_map_.disjointSetOf(id, IdMappingMode::EXACT));
  });

  // Traverse through indexed domains.
  const auto indexed_id_map = getIndexedConsumerToProducerMap(fusion_, ca_map_);

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      all_exact_sets_covered;

  // Back traverses through the exact map and indexed
  // producer-consumer pairs
  for (auto current_sets = exact_sets; !current_sets.empty();) {
    auto producer_sets = ca_map_.getAllDisjointSetProducers(current_sets);
    all_exact_sets_covered.pushBack(producer_sets);

    current_sets.clear();

    // Further traversal if any of the new producer sets is a producer
    // of indexed domains
    for (const auto& producer_set : producer_sets) {
      auto producer_of_producer_it = indexed_id_map.find(producer_set);
      if (producer_of_producer_it != indexed_id_map.end()) {
        current_sets.pushBack(producer_of_producer_it->second);
      }
    }
  }

  for (const auto& exact_set_ptr : all_exact_sets_covered) {
    auto exact_concrete_id = ca_map_.getConcreteMappedID(
        exact_set_ptr->front(), IdMappingMode::EXACT);
    eraseIfMapped(in_ids, exact_concrete_id);
  }
}

// Find any id in domain that maps with target id
IterDomain* DomainMap::anyMapped(
    const std::vector<IterDomain*>& domain,
    IterDomain* target) const {
  for (auto id : domain) {
    if (ca_map_.areMapped(id, target, IdMappingMode::EXACT)) {
      return id;
    }
  }
  return nullptr;
}

// Determine if output TensorView is a valid reference tensor for this fusion.
// The reference tensor must map to all the iterDomains in each input.
bool DomainMap::isValidReference(TensorView* tv) const {
  for (auto input_tv : ir_utils::filterByType<TensorView>(fusion_->inputs())) {
    if (input_tv->uses().empty()) {
      continue;
    }
    // TODO: Same backward traversal from tv is done for all input
    // tvs. Consider doing the analysis one for all inputs
    if (!areAllInputIdsMappedTo(input_tv, tv)) {
      return false;
    }
  }
  return true;
}

} // namespace pointwise_utils
} // namespace nvfuser
