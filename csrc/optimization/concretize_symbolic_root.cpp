// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <ir/utils.h>
#include <optimization/concretize_symbolic_root.h>
#include <options.h>
#include <root_domain_map.h>

namespace nvfuser::optimization {

namespace {

void concretize_symbolic_root_domain(Fusion* fusion) {
  // map non-const extents to const extents
  std::unordered_map<Val*, Val*> replacement_map;

  const auto mapped_sets = ExactRootDomainMap(fusion).getMappedSets();
  // Loop over each exact root domain set
  for (const auto& set_ptr : mapped_sets.disjointSets()) {
    // For each set, find an extent that is a const scalar
    Val* const_extent = nullptr;
    for (auto id : *set_ptr) {
      if (id->isBroadcast()) {
        continue;
      }
      if (id->extent()->isConstScalar()) {
        const_extent = id->extent();
        break;
      }
    }

    // If no const extent was found, skip this set
    if (!const_extent) {
      continue;
    }

    // place non-const extents and target const_extent to the replacement map
    for (auto id : *set_ptr) {
      if (id->isBroadcast()) {
        continue;
      }
      if (!id->extent()->isConstScalar()) {
        replacement_map.emplace(id->extent(), const_extent);
      }
    }
  }

  // Replace non-const extents with const extents
  ir_utils::replaceValue(fusion, replacement_map);
}
} // namespace

void ConcretizeSymbolicRootDomainPass::runPass(Fusion* fusion) {
  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion before ConcretizeSymbolicRootDomainPass:" << std::endl;
    fusion->printMath();
  }

  concretize_symbolic_root_domain(fusion);

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion after ConcretizeSymbolicRootDomainPass:" << std::endl;
    fusion->printMath();
  }
}

} // namespace nvfuser::optimization
