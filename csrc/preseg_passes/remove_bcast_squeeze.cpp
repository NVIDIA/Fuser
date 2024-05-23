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
inline bool isFusionOutput(Fusion* fusion, Val* val) {
  const auto& outputs = fusion->outputs();
  return std::find(outputs.begin(), outputs.end(), val) != outputs.end();
}
void removeBcastSqueeze(Fusion* fusion) {
  // step-1: find and remove broadcast + squeeze pattern
  // before: Y = broadcast(X); Z = squeeze(Y);  M = someOp(Z)
  // after : M = someOp(X)
  // conditions: (1) broadcast and squeeze have the same dim flags
  //             (2) Y is only consumed by Z (has only one consumer)
  // special case: if Z is a fusion output, replace the output with X
  {
    std::unordered_map<Val*, Val*> replacement_map;
    for (auto expr : fusion->exprs()) {
      if (auto sop = dynamic_cast<SqueezeOp*>(expr)) {
        if (auto bcast = dynamic_cast<BroadcastOp*>(sop->in()->definition())) {
          if (bcast->getBroadcastDimFlags() == sop->getSqueezeDimFlags() &&
              bcast->out()->uses().size() == 1) {
            if (isFusionOutput(fusion, sop->out())) {
              fusion->replaceOutput(sop->out(), bcast->in());
            }
            replacement_map.insert({sop->out(), bcast->in()});
          }
        }
      }
    }
    // Replace if there is any match
    if (!replacement_map.empty()) {
      ir_utils::replaceValue(fusion, replacement_map);
    }
  }
  // step-2: find and remove squeeze + broadcast pattern
  // before: Y = squeeze(X); Z = broadcast(Y);  M = someOp(Z)
  // after : M = someOp(X)
  // conditions: (1) broadcast and squeeze have the same dim flags
  //             (2) Y is only consumed by Z (has only one consumer)
  // special case: if Z is a fusion output, replace the output with X
  {
    std::unordered_map<Val*, Val*> replacement_map;
    for (auto expr : fusion->exprs()) {
      if (auto bcast = dynamic_cast<BroadcastOp*>(expr)) {
        if (auto sop = dynamic_cast<SqueezeOp*>(bcast->in()->definition())) {
          if (bcast->getBroadcastDimFlags() == sop->getSqueezeDimFlags() &&
              sop->out()->uses().size() == 1) {
            if (isFusionOutput(fusion, bcast->out())) {
              fusion->replaceOutput(bcast->out(), sop->in());
            }
            replacement_map.insert({bcast->out(), sop->in()});
          }
        }
      }
    }
    // Replace if there is any match
    if (!replacement_map.empty()) {
      ir_utils::replaceValue(fusion, replacement_map);
    }
  }
}

} // namespace

void RemoveBcastSqueeze::runPass(Fusion* fusion) {
  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion before RemoveBcastSqueeze:" << std::endl;
  }
  fusion->printMath();

  removeBcastSqueeze(fusion);

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion after RemoveBcastSqueeze:" << std::endl;
  }
  fusion->printMath();
}

} // namespace nvfuser::preseg_passes
