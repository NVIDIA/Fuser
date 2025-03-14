// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <id_model/id_model.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <options.h>
#include <preseg_passes/exact_mapped_extent_substitution.h>
namespace nvfuser::preseg_passes {

namespace {
// Skip broadcast without expanded extent
// Skip derived domains, e.g. iS11{( i0 * i2 )}rf
// Skip domain whose extent is derived e.g. iS12{( i0 * i2 )}
// e.g. in this set { iS11{( i0 * i2 )}rf; iS12{( i0 * i2 )}; iS14{i3} } from
// NVFuserTest.SymbolicSqueeze, we can't substitute {i0 * i2} with {i3},
// otherwise, validateDomainEquivalence fails. If we really want to substitute,
// we may need to skip or modify validateDomainEquivalence.
inline bool isNonSubstitutableID(const IterDomain* id) {
  return id == nullptr || (id->isBroadcast() && !id->hasExpandedExtent()) ||
      id->definition() || id->extent()->definition();
}

// Build disjoint set of extents from  disjoint set of ids
auto buildExtentSetFromIdSets(const DisjointSets<Val*>& id_sets) {
  DisjointSets<Val*> extent_sets;
  // Loop over each id set
  for (const auto& id_set_ptr : id_sets.disjointSets()) {
    // If one of the extent in this set is already in the extent_sets, then
    // map all other extents to the same set, otherwise create a new set.
    DisjointSets<Val*>::DisjointSet current_extent_set = nullptr;

    // First substitutable id in this set, used to create a new set when no
    // extent is mapped
    IterDomain* first_substitutable_id = nullptr;

    // First loop over the set, to check if one of the extent is already mapped
    for (auto id_set_val : *id_set_ptr) {
      auto id = dynamic_cast<IterDomain*>(id_set_val);
      if (isNonSubstitutableID(id)) {
        continue;
      }
      if (extent_sets.mappingExists(id->extent())) {
        current_extent_set = extent_sets.disjointSetMap().at(id->extent());
        break;
      }
      if (first_substitutable_id == nullptr) {
        first_substitutable_id = id;
      }
    }

    // Create a new set if no extent is mapped
    // if no substitutable id in this set, no need to create a new set, continue
    // to the next id set
    if (current_extent_set == nullptr) {
      if (first_substitutable_id == nullptr) {
        continue;
      }
      auto extent = first_substitutable_id->extent();
      auto it = extent_sets.initializeSet(extent).first;
      current_extent_set = it->second;
    }

    // Second loop over the ID set, to map all extents to the same extent set.
    for (auto id_set_val : *id_set_ptr) {
      auto id = dynamic_cast<IterDomain*>(id_set_val);
      if (isNonSubstitutableID(id)) {
        continue;
      }
      // Here extent is used instead of expanded exent since a bcast dim may
      // be expanded to different extents, e.g. issue-3227.
      auto extent = id->extent();
      if (!extent_sets.mappingExists(extent)) {
        extent_sets.appendToSet(extent, current_extent_set);
      }
    }
  }

  return extent_sets;
}

void exactMappedExtentSubstitution(Fusion* fusion) {
  // map non-const extents to const extents
  std::unordered_map<Val*, Val*> replacement_map;

  // Build the exact graph
  IdModel id_model(fusion);
  id_model.buildExactGraph();
  const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);
  const DisjointSets<Val*>& id_sets = exact_graph.disjointValSets();
  const auto& extent_set = buildExtentSetFromIdSets(id_sets);

  // Loop over each set of extents
  for (const auto& set_ptr : extent_set.disjointSets()) {
    // (1) pick a const extent
    // (2) if no const extent, pick the var with the lowest name()
    Val* const_extent = nullptr;
    Val* lowest_val = nullptr;
    for (auto extent : *set_ptr) {
      // find the const extent, if already seen, check if they are the same.
      if (extent->isConstScalar()) {
        if (const_extent) {
          NVF_CHECK(
              const_extent->sameAs(extent),
              "Found two different const extents in the same set: ",
              set_ptr->toString());
        } else {
          const_extent = extent;
        }
      }
      // find the lowest name
      if (!lowest_val || extent->name() < lowest_val->name()) {
        lowest_val = extent;
      }
    }
    // replace with const extents.
    // if no const extents, replace with the one with the lowest name.
    // avoid self-replacement
    auto replaced_to = const_extent ? const_extent : lowest_val;
    for (auto extent : *set_ptr) {
      if (extent != replaced_to) {
        replacement_map.emplace(extent, replaced_to);
      }
    }
  }

  // Replace non-const extents with const extents
  ir_utils::replaceValue(fusion, replacement_map);
}
} // namespace

void ExactMappedExtentSubstitutionPass::runPass(Fusion* fusion) {
  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "DisjointSets before " << name() << ":" << std::endl;
    IdModel id_model(fusion, /*build_graphs=*/false);
    id_model.buildExactGraph();
    const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);
    const DisjointSets<Val*>& id_sets = exact_graph.disjointValSets();
    debug() << id_sets.toString() << std::endl;
  }

  exactMappedExtentSubstitution(fusion);

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "ExactLogicalDomainMap after " << name() << ":" << std::endl;
    IdModel id_model(fusion, false, false, false);
    id_model.buildExactGraph();
    const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);
    const DisjointSets<Val*>& id_sets = exact_graph.disjointValSets();
    debug() << id_sets.toString() << std::endl;
  }
}

} // namespace nvfuser::preseg_passes
