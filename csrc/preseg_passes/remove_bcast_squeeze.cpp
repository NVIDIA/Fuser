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

void removeBcastSqueeze(Fusion* fusion) {

  std::unordered_map<Val*, Val*> replacement_map;
  // Find broadcast + squeeze pattern
  for (auto expr : fusion->exprs()) {
    auto sop = dynamic_cast<SqueezeOp*>(expr);
    if(!sop){
      continue;
    }
    // Y = squeeze(broadcast(X))
    auto def_sop_in = sop->in()->definition();
    if(auto bcast = dynamic_cast<BroadcastOp*>(def_sop_in)){
      replacement_map.insert({sop->out(), bcast->in()});
      std::cout << "Replace " << sop->out() << " with " << bcast->in() << std::endl;
    }else if (auto sop_out_tv = dynamic_cast<TensorView*>(sop->out())) {
        // Y = broadcast(squeeze(X))
        auto consumers = ir_utils::consumerTvsOf(sop_out_tv);
        for(auto consumer : consumers){
          std::cout << "Consumer: " << consumer->toString() << std::endl;
          if(auto bcast = dynamic_cast<BroadcastOp*>(consumer->definition())){
            replacement_map.insert({bcast->out(), sop->in()});
          }
        }
      }
  }
  // Replace non-const extents with const extents
  if(!replacement_map.empty()){
    ir_utils::replaceValue(fusion, replacement_map);
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
