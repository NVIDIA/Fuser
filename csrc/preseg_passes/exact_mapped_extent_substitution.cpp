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
  return (id->isBroadcast() && !id->hasExpandedExtent()) || id->definition() ||
      id->extent()->definition();
}

DisjointSets<Val*> buildExtentSetFromIdSets(const DisjointSets<Val*>& id_sets) {
  std::cout << "===============id set===============" << std::endl;
  std::cout << id_sets.toString() << std::endl;
  std::cout << "==============================" << std::endl;

  DisjointSets<Val*> extent_set;
  Val* extent_0 = nullptr;
  for (const auto& set_ptr : id_sets.disjointSets()) {
    std::cout << "\n============set_ptr==================" << std::endl;
    bool first = true;
    DisjointSets<Val*>::DisjointSet new_set;
    for (auto v : *set_ptr) {
      auto id = dynamic_cast<IterDomain*>(v);
      auto extent = id->extent();

      std::cout << "\nid: " << id->toString() << " extent: " << extent->toString()
                << std::endl;

      if (!extent_0) {
        extent_0 = extent;
      }

      std::cout << "extent_0: " << extent_0->toString()
                << " extent: " << extent->toString() << ", isSame "
                << extent->sameAs(extent_0) << std::endl;

      if (extent_set.mappingExists(extent)) {
        std::cout << " extent already exists " << std::endl;
        continue;
      }
      std::cout << " extent not exists " << std::endl;

      if (first) {
        auto it = extent_set.initializeSet(extent).first;
        new_set = it->second;
        first = false;
        std::cout << " initializeSet " << extent->toString() << std::endl;
      } else {
        std::cout << " appendToSet " << extent->toString() << std::endl;
        extent_set.appendToSet(extent, new_set);
      }
      
      std::cout << "-------------extent_set-------------" << std::endl;
      std::cout << extent_set.toString() << std::endl;
      std::cout << "-------------extent_set-------------" << std::endl;

    }
  }
  std::cout << "============extent_set==================" << std::endl;
  std::cout << extent_set.toString() << std::endl;
  std::cout << "==============================" << std::endl;
  return extent_set;
}

void exactMappedExtentSubstitution(Fusion* fusion) {
  // map non-const extents to const extents
  std::unordered_map<Val*, Val*> replacement_map;

  // Build the exact graph
  IdModel id_model(fusion, false, false, false);
  id_model.buildExactGraph();
  const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);
  const DisjointSets<Val*>& id_sets = exact_graph.disjointValSets();

  const auto& extent_set = buildExtentSetFromIdSets(id_sets);

  // Loop over each set of values
  for (const auto& set_ptr : id_sets.disjointSets()) {
    // (1) pick a const extent
    // (2) if no const extent, pick the var with the lowest name()
    Val* const_extent = nullptr;
    Val* lowest_val = nullptr;
    for (auto v : *set_ptr) {
      auto id = dynamic_cast<IterDomain*>(v);
      if (id == nullptr || isNonSubstitutableID(id)) {
        continue;
      }
      // find the const extent, if already seen, check if they are the same.
      // Here extent is used instead of expanded exent since a bcast dim may
      // be expanded to different extents, e.g. issue-3227.
      if (id->extent()->isConstScalar()) {
        if (const_extent) {
          NVF_CHECK(
              const_extent->sameAs(id->extent()),
              "Found two different const extents in the same set: ",
              set_ptr->toString());
        } else {
          const_extent = id->extent();
        }
      }
      // find the lowest name
      if (!lowest_val || id->extent()->name() < lowest_val->name()) {
        lowest_val = id->extent();
      }
    }
    // replace with const extents.
    // if no const extents, replace with the one with the lowest name.
    for (auto v : *set_ptr) {
      auto id = dynamic_cast<IterDomain*>(v);
      // No need to reaplce constant extent, they are same if mapped to
      // the same set.
      if (id == nullptr || isNonSubstitutableID(id)) {
        continue;
      }
      replacement_map.emplace(
          id->extent(), const_extent ? const_extent : lowest_val);
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
  fusion->printMath();
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
