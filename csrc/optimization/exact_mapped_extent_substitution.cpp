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
    bool can_substitute = true;
    for (auto id : *set_ptr) {
      // If one of the domains is not a root domain, we cannot substitute.
      // e.g. in this set { iS11{( i0 * i2 )}rf; iS12{( i0 * i2 )}; iS14{i3} }
      // we can't substitute {i0 * i2} with {i3}, otherwise,
      // ValidateDomainEquivalence fails. If we really want to substitute, we
      // may need to skip ValidateDomainEquivalence.
      if (id->definition()) {
        can_substitute = false;
        break;
      }
      // Skip broadcast without expanded extent
      if (id->isBroadcast() && !id->hasExpandedExtent()) {
        continue;
      }
      // find the const extent, don't use break here as we may miss the
      // detection of a non-root domain.
      if (!const_extent && id->getMaybeExpandedExtent()->isConstScalar()) {
        const_extent = id->getMaybeExpandedExtent();
      }
      // find the lowest name
      if (!lowest_val ||
          id->getMaybeExpandedExtent()->name() < lowest_val->name()) {
        lowest_val = id->getMaybeExpandedExtent();
      }
    }
    // skip this set if we can't substitute
    if (!can_substitute) {
      continue;
    }
    // replace with const extents.
    // if no const extents, replace with the one with the lowest name.
    for (auto id : *set_ptr) {
      if (id->isBroadcast() && !id->hasExpandedExtent()) {
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
  }

  exactMappedExtentSubstitution(fusion);

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion after exactMappedExtentSubstitutionPass:" << std::endl;
    fusion->printMath();
  }
}

} // namespace nvfuser::optimization
