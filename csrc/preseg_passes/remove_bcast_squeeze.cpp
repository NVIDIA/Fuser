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
#include <ops/alias.h>
namespace nvfuser::preseg_passes {

namespace {
std::deque<std::deque<TensorView*>> tvChains(
    std::deque<std::deque<Val*>> val_chains) {
  std::deque<std::deque<TensorView*>> tv_chains(val_chains.size());
  for (const auto i : c10::irange(val_chains.size())) {
    auto tv_iterable = ir_utils::filterByType<TensorView>(val_chains[i]);
    tv_chains[i] =
        std::deque<TensorView*>(tv_iterable.begin(), tv_iterable.end());
  }
  return tv_chains;
}

void removeBcastSqueeze(Fusion* fusion) {

  std::unordered_map<Val*, Val*> replacement_map;
  // Find broadcast + squeeze pattern
  BroadcastOp* bcast_op = nullptr;
  SqueezeOp* squeeze_op = nullptr;
  for (auto expr : fusion->exprs()) {
    if(auto sop = dynamic_cast<SqueezeOp*>(expr)){
      squeeze_op = sop;
    }else if(auto bcast = dynamic_cast<BroadcastOp*>(expr)){
      bcast_op = bcast;
    }
    if(!squeeze_op || !bcast_op || squeeze_op->getSqueezeDimFlags() != bcast_op->getBroadcastDimFlags()){
      continue;
    }
    // we have both broadcast and squeeze
    bool can_be_removed = true;
    bool is_squeeze_bcast = DependencyCheck::isDependencyOf(squeeze_op->out(), bcast_op->in());
    bool is_bcast_squeeze = DependencyCheck::isDependencyOf(bcast_op->out(), squeeze_op->in());
    std::cout << "is_squeeze_bcast: " << is_squeeze_bcast << " is_bcast_squeeze: " << is_bcast_squeeze << std::endl;
    if(is_bcast_squeeze){
      // const auto& all_vals = DependencyCheck::getAllValsBetween({bcast_op->out()},{squeeze_op->in()});
      auto forward_tv_chains = tvChains(DependencyCheck::getAllUseChains(bcast_op->out()));
      std::unordered_set<TensorView*> all_bcast_consumers;
      // check all uses of broadcast tv
      for (auto forward_tv_dep_chain : forward_tv_chains) {
        std::cout << "=========== forward_tv_dep_chain: " << std::endl;
        TensorView* forward_running_producer = nullptr;
        TensorView* forward_running_consumer = forward_tv_dep_chain.front();
        forward_tv_dep_chain.pop_front();
        while (!forward_tv_dep_chain.empty()) {
          all_bcast_consumers.insert(forward_running_consumer);
          forward_running_producer = forward_running_consumer;
          forward_running_consumer = forward_tv_dep_chain.front();
          forward_tv_dep_chain.pop_front();
          std::cout << "Producer: " << forward_running_producer->toString() << " Consumer: " << forward_running_consumer->toString() << std::endl;
          // if a tv in a use chain has multiple producers
          // e.g. `Z = broadcast(X) + Y`, if replace broadcast(X) with X, must ensure Y = f(broadcast(X))
          const auto& producers = ir_utils::producerTvsOf(forward_running_consumer);
          for(auto producer : producers){
            if(all_bcast_consumers.count(producer) == 0){
              can_be_removed = false;
              break;
            }
          }
        }
        if(!can_be_removed){
          break;
        }
      }
      std::cout << "Can be removed: " << can_be_removed << std::endl;
      if(can_be_removed){
        // for(auto expr : bcast_op->out()->uses()){
        //   std::cout << "Replace " << bcast_op->out()->toString() << " with " << bcast_op->in()->toString() << std::endl;
        //   std::cout << "Expr: " << expr->toString() << std::endl;
        //   ir_utils::replaceValInExprInputs(expr, bcast_op->out(), bcast_op->in());
        // }
        // for(auto expr : squeeze_op->out()->uses()){
        //   std::cout<< "Replace " << squeeze_op->out()->toString() << " with " << squeeze_op->in()->toString() << std::endl;
        //   std::cout << "Expr: " << expr->toString() << std::endl;
        //   ir_utils::replaceValInExprInputs(expr, squeeze_op->out(), squeeze_op->in());
        // }  
        for(auto tv : all_bcast_consumers){
          // TODO: same broadcast dim
          if(tv->hasBroadcast()){
            auto squeezed_tv = squeeze(tv, squeeze_op->getSqueezeDimFlags());
            std::cout << "Replace " << tv->toString() << " with " << squeezed_tv->toString() << std::endl;
            replacement_map.insert({tv, squeezed_tv});
          }
        }
      }
        // std::unordered_set<Val*> all_vals_set(all_vals.begin(), all_vals.end());
        // replacement_map.insert({bcast_op->out(), bcast_op->in()});

    }
    // // Y = squeeze(broadcast(X))
    // auto def_sop_in = sop->in()->definition();
    // if(auto bcast = dynamic_cast<BroadcastOp*>(def_sop_in)){..
    // 3
    //   replacement_map.insert({sop->out(), bcast->in()});
    //   std::cout << "Replace " << sop->out() << " with " << bcast->in() << std::endl;
    // }else if (auto sop_out_tv = dynamic_cast<TensorView*>(sop->out())) {
    //     // Y = broadcast(squeeze(X))
    //     auto consumers = ir_utils::consumerTvsOf(sop_out_tv);
    //     for(auto consumer : consumers){
    //       std::cout << "Consumer: " << consumer->toString() << std::endl;
    //       if(auto bcast = dynamic_cast<BroadcastOp*>(consumer->definition())){
    //         replacement_map.insert({bcast->out(), sop->in()});
    //       }
    //     }
    //   }
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
