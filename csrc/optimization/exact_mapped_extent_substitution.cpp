// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <ir/utils.h>
#include <optimization/exact_mapped_extent_substitution.h>
#include <options.h>
#include <root_domain_map.h>

namespace nvfuser::optimization {

namespace {
// Skip broadcast without expanded extent
// Skip derived domains, e.g. iS11{( i0 * i2 )}rf
// Skip domain whose extent is derived e.g. iS12{( i0 * i2 )}
// e.g. in this set { iS11{( i0 * i2 )}rf; iS12{( i0 * i2 )}; iS14{i3} } from
// NVFuserTest.SymbolicSqueeze, we can't substitute {i0 * i2} with {i3},
// otherwise, ValidateDomainEquivalence fails. If we really want to substitute,
// we may need to skip or modify ValidateDomainEquivalence.
inline bool isNonSubstitutableID(const IterDomain* id) {
  return (id->isBroadcast() && !id->hasExpandedExtent()) || id->definition() ||
      id->getMaybeExpandedExtent()->definition();
}

void exactMappedExtentSubstitution(Fusion* fusion) {
  // map non-const extents to const extents
  std::unordered_map<Val*, Val*> replacement_map;

  const auto mapped_sets = ExactRootDomainMap(fusion).getMappedSets();
  // Loop over each exact root domain set
  for (const auto& set_ptr : mapped_sets.disjointSets()) {
    // (1) pick a const extent
    // (2) if no const extent, pick the var with the lowest name()
    Val* const_extent = nullptr;
    Val* lowest_val = nullptr;
    for (auto id : *set_ptr) {
      if (isNonSubstitutableID(id)) {
        continue;
      }
      // find the const extent, if already seen, check if they are the same
      if (id->getMaybeExpandedExtent()->isConstScalar()) {
        if (const_extent) {
          NVF_CHECK(
              const_extent->sameAs(id->getMaybeExpandedExtent()),
              "Found two different const extents in the same set: ",
              set_ptr->toString());
        } else {
          const_extent = id->getMaybeExpandedExtent();
        }
      }
      // find the lowest name
      if (!lowest_val ||
          id->getMaybeExpandedExtent()->name() < lowest_val->name()) {
        lowest_val = id->getMaybeExpandedExtent();
      }
    }
    // replace with const extents.
    // if no const extents, replace with the one with the lowest name.
    for (auto id : *set_ptr) {
      if (isNonSubstitutableID(id)) {
        continue;
      }
      replacement_map.emplace(
          id->getMaybeExpandedExtent(),
          const_extent ? const_extent : lowest_val);
    }
  }

  // Replace non-const extents with const extents
  ir_utils::replaceValue(fusion, replacement_map);
}
} // namespace

void ExactMappedExtentSubstitutionPass::runPass(Fusion* fusion) {
  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion before exactMappedExtentSubstitutionPass:" << std::endl;
    fusion->printMath();
    debug() << "ExactRootDomainMap before exactMappedExtentSubstitutionPass:"
            << std::endl;
    const auto mapped_sets = ExactRootDomainMap(fusion).getMappedSets();
    debug() << mapped_sets.toString() << std::endl;
  }

  exactMappedExtentSubstitution(fusion);

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion after exactMappedExtentSubstitutionPass:" << std::endl;
    fusion->printMath();
    debug() << "ExactRootDomainMap after exactMappedExtentSubstitutionPass:"
            << std::endl;
    const auto mapped_sets = ExactRootDomainMap(fusion).getMappedSets();
    debug() << mapped_sets.toString() << std::endl;
  }
}

} // namespace nvfuser::optimization
