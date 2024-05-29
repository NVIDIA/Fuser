// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <ir/utils.h>
#include <options.h>
#include <preseg_passes/remove_bcast_squeeze.h>
#include <root_domain_map.h>

namespace nvfuser::preseg_passes {

namespace {
// Remove broadcast-squeeze and squeeze-broadcast patterns
// TODO: still remove when have intermediate ops between broadcast and squeeze
void removeBcastSqueeze(Fusion* fusion) {
  std::unordered_map<Val*, Val*> replacement_map;
  for (auto expr : fusion->exprs()) {
    // step-1: find and remove broadcast + squeeze pattern
    // before: Y = broadcast(X); Z = squeeze(Y);  M = someOp(Z)
    // after : M = someOp(X)
    // conditions: (1) broadcast and squeeze have the same dim flags
    // special case: if Z is a fusion output, replace the output with X
    if (auto sop = dynamic_cast<SqueezeOp*>(expr)) {
      if (auto bcast = dynamic_cast<BroadcastOp*>(sop->in()->definition())) {
        if (bcast->getBroadcastDimFlags() == sop->getSqueezeDimFlags()) {
          if (sop->out()->isFusionOutput()) {
            fusion->replaceOutput(sop->out(), bcast->in());
          }
          // if already have a -> b in replacement map and comes a new pair of c
          // -> a, instead of add c -> a to the map, add c -> b to avoid two hop
          // replacement. see test BcastSqueezeBcastSqueeze.
          auto new_val = bcast->in();
          if (replacement_map.count(new_val)) {
            new_val = replacement_map[new_val];
          }
          replacement_map.insert({sop->out(), new_val});
        }
      }
    }

    // step-2: find and remove squeeze + broadcast pattern
    // before: Y = squeeze(X); Z = broadcast(Y);  M = someOp(Z)
    // after : M = someOp(X)
    // conditions: (1) broadcast and squeeze have the same dim flags
    // special case: if Z is a fusion output, replace the output with X
    if (auto bcast = dynamic_cast<BroadcastOp*>(expr)) {
      if (auto sop = dynamic_cast<SqueezeOp*>(bcast->in()->definition())) {
        if (bcast->getBroadcastDimFlags() == sop->getSqueezeDimFlags()) {
          if (bcast->out()->isFusionOutput()) {
            fusion->replaceOutput(bcast->out(), sop->in());
          }
          // if already have a -> b in replacement map and comes a new pair of c
          // -> a, instead of add c -> a to the map, add c -> b to avoid two hop
          // replacement. see test SqueezeBcastSqueezeBcast
          auto new_val = sop->in();
          if (replacement_map.count(new_val)) {
            new_val = replacement_map[new_val];
          }
          replacement_map.insert({bcast->out(), new_val});
        }
      }
    }
  }
  // Replace if there is any match
  if (!replacement_map.empty()) {
    ir_utils::replaceValue(fusion, replacement_map);
  }
}

} // namespace

void RemoveBcastSqueeze::runPass(Fusion* fusion) {
  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion before RemoveBcastSqueeze:" << std::endl;
    fusion->printMath();
  }

  removeBcastSqueeze(fusion);

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion after RemoveBcastSqueeze:" << std::endl;
    fusion->printMath();
  }
}

} // namespace nvfuser::preseg_passes
