// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir_utils.h>
#include <scheduler/pointwise_utils.h>

namespace nvfuser {
namespace pointwise_utils {

DomainMap::DomainMap(Fusion* fusion) : fusion_(fusion), ca_map_(fusion) {
  tvs_with_rfactor_ = scheduler_utils::getTVsWithNonReductionRFactor(fusion);
  for (auto select : ir_utils::getSelectOps(fusion)) {
    select_ids_.emplace(select->getIndexedID());
  }
  for (auto select : ir_utils::getIndexSelectOps(fusion)) {
    select_ids_.emplace(select->getIndexedID());
  }
  for (auto select : ir_utils::getTorchGatherOps(fusion)) {
    TORCH_INTERNAL_ASSERT(select->input(0)->isA<TensorView>());
    auto inp = select->input(0)->as<TensorView>();
    auto inp_domain = TensorDomain::noReductions(inp->getMaybeRFactorDomain());
    for (auto id : inp_domain) {
      select_ids_.emplace(id);
    }
  }
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
    if (!concrete->isBroadcast() && !in_id->isReduction() &&
        !isSelectId(in_id)) {
      in_concrete_ids.insert(concrete);
    }
  }

  // Erase all input concrete IDs mapped to the output domain
  // Ignore unresolved broadcast dimensions
  eraseIfInputMappedThroughRFactorDomain(
      in_concrete_ids, tv->getMaybeRFactorDomain());

  // Erase input concrete IDs mapped to any reduction domains
  for (auto id :
       ir_utils::filterByType<IterDomain>(input_tv->container()->vals())) {
    if (id->isReduction()) {
      auto concrete =
          ca_map_.getConcreteMappedID(id, IdMappingMode::PERMISSIVE);
      auto conc_it = in_concrete_ids.find(concrete);
      if (conc_it != in_concrete_ids.end()) {
        in_concrete_ids.erase(conc_it);
      }
    }
  }

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

  // This technically traverses exprs that are not rfactor
  // transformations but when this function is used, there should be
  // no non-rfactor IterDomain exprs
  auto all_exact_sets_covered = ca_map_.getAllDisjointSetProducers(exact_sets);

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
