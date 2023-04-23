// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir_utils.h>
#include <lower_utils.h>
#include <scheduler/pointwise_utils.h>
#include <utils.h>

namespace nvfuser {
namespace pointwise_utils {

namespace {

// Grab all exact set pairs of producer and consumer domains of
// indexed accesses, e.g., index_select
std::vector<std::pair<
    std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>,
    std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>>
getIndexedProducerConsumerPairs(Fusion* fusion, const ComputeAtMap& ca_map) {
  std::vector<std::pair<
      std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>,
      std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>>
      indexed_ids;

  for (auto expr : fusion->exprs()) {
    if (auto gather = dynamic_cast<TorchGatherOp*>(expr)) {
      auto producer_indexe_id = gather->getIndexedProducerDomain();
      auto out_tv = ir_utils::getTvOutput(expr);
      PairwiseRootDomainMap p2c(gather->lookupTv(), out_tv);
      p2c.mapDifferentExtents(true).mapIndexedDomains(true);
      for (const auto& [p_id, c_id] : p2c.mapProducerToConsumer(
               gather->lookupTv()->domain(), out_tv->domain())) {
        if ((gather->isTakeAlongAxis() && p_id == producer_indexe_id) ||
            !gather->isTakeAlongAxis()) {
          indexed_ids.emplace_back(
              ca_map.disjointSetOf(p_id, IdMappingMode::EXACT),
              ca_map.disjointSetOf(c_id, IdMappingMode::EXACT));
        }
      }
    } else if (auto index_select = dynamic_cast<IndexSelectOp*>(expr)) {
      auto p_id = index_select->getIndexedProducerDomain();
      auto c_id = index_select->getIndexedConsumerDomain();
      indexed_ids.emplace_back(
          ca_map.disjointSetOf(p_id, IdMappingMode::EXACT),
          ca_map.disjointSetOf(c_id, IdMappingMode::EXACT));
    } else {
      // Note there's no consumer ID for select. This means we can't
      // just propagate from consumers to indexed producers. It seems
      // it's necessary to schedule producers and consumers separately
      // in those cases.
      continue;
    }
  }

  return indexed_ids;
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
    // Permissive map is required for the transpose scheduler to support cases
    // like T0[I0, b] + T1[b, I1]
    auto concrete =
        ca_map_.getConcreteMappedID(in_id, IdMappingMode::PERMISSIVE);

    bool skip = true;
    for (auto use : input_tv->uses()) {
      if (auto select = dynamic_cast<SelectOp*>(use)) {
        if (in_id == select->getIndexedProducerDomain()) {
          // effectively same as squeeze
          continue;
        }
      } else if (auto index_select = dynamic_cast<IndexSelectOp*>(use)) {
        // The consumer ID may be a broadcast. In that case, nothing
        // needs to be mapped with it
        if (in_id == index_select->getIndexedProducerDomain() &&
            index_select->getIndexedConsumerDomain()->isBroadcast()) {
          continue;
        } else {
          skip = false;
          break;
        }
      } else if (auto gather = dynamic_cast<TorchGatherOp*>(use)) {
        // The consumer ID may be a broadcast. In that case, nothing
        // needs to be mapped with it
        if (in_id == gather->getIndexedProducerDomain() &&
            gather->getIndexedConsumerDomain()->isBroadcast()) {
          continue;
        } else {
          skip = false;
          break;
        }
      } else {
        skip = false;
        break;
      }
    }

    if (!concrete->isBroadcast() && !in_id->isReduction() && !skip) {
      in_concrete_ids.insert(concrete);
    }
  }

  // Erase all input concrete IDs mapped to the output domain
  // Ignore unresolved broadcast dimensions
  eraseIfInputMappedThroughRFactorDomain(
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

void DomainMap::eraseIfInputMappedThroughRFactorDomain(
    std::unordered_set<IterDomain*>& in_ids,
    const std::vector<IterDomain*>& ids) const {
  // Use ComputeAtMap::getAllDisjointSetProducers to grab all producer
  // IDs through rfactor exprs
  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      exact_sets;
  std::for_each(ids.begin(), ids.end(), [&](IterDomain* id) {
    exact_sets.pushBack(ca_map_.disjointSetOf(id, IdMappingMode::EXACT));
  });

  // Traverse through indexed domains. Don't care if indexed or
  // not. Maybe it should be configurable.
  const auto indexed_sets = getIndexedProducerConsumerPairs(fusion_, ca_map_);

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      all_exact_sets_covered;

  // Back traverses through the exact map and indexed
  // producer-consumer pairs
  for (auto current_sets = exact_sets; !current_sets.empty();) {
    auto producer_sets = ca_map_.getAllDisjointSetProducers(current_sets);
    all_exact_sets_covered.pushBack(producer_sets);

    current_sets.clear();

    for (const auto& exact_set : producer_sets) {
      auto indexed_it = std::find_if(
          indexed_sets.begin(), indexed_sets.end(), [&](const auto& pair) {
            return pair.second == exact_set;
          });
      if (indexed_it != indexed_sets.end()) {
        current_sets.pushBack(indexed_it->first);
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
    if (!areAllInputIdsMappedTo(input_tv, tv)) {
      return false;
    }
  }
  return true;
}

} // namespace pointwise_utils
} // namespace nvfuser
