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

void exactMappedExtentSubstitution(Fusion* fusion) {
  // map non-const extents to const extents
  std::unordered_map<Val*, Val*> replacement_map;

  // Build the exact graph
  IdModel id_model(fusion, false, false, false);
  id_model.buildExactGraph();
  const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);
  const DisjointSets<Val*>& val_sets = exact_graph.disjointValSets();

  // Loop over each set of values
  for (const auto& set_ptr : val_sets.disjointSets()) {
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
      if (id == nullptr || isNonSubstitutableID(id) ||
          id->extent()->isConstScalar()) {
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
    const DisjointSets<Val*>& val_sets = exact_graph.disjointValSets();
    debug() << val_sets.toString() << std::endl;
  }

  exactMappedExtentSubstitution(fusion);

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "ExactLogicalDomainMap after " << name() << ":" << std::endl;
    IdModel id_model(fusion, false, false, false);
    id_model.buildExactGraph();
    const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);
    const DisjointSets<Val*>& val_sets = exact_graph.disjointValSets();
    debug() << val_sets.toString() << std::endl;
  }
}

} // namespace nvfuser::preseg_passes
