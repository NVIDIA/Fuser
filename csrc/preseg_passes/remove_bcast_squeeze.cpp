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
// Remove broadcast-squeeze and squeeze-broadcast patterns
// TODO: still remove when have intermediate ops between broadcast and squeeze
void removeBcastSqueeze(Fusion* fusion) {
  // Iterate backwards over fusion expressions.
  // This will ensure we don't process expressions that are no longer valid
  // after replacement.
  SqueezeOp* squeeze = nullptr;
  BroadcastOp* bcast = nullptr;
  auto exprs = fusion->exprs();
  for (auto it = std::rbegin(exprs); it != std::rend(exprs); it++) {
    Expr* expr = *it;
    if (auto sop = dynamic_cast<SqueezeOp*>(expr)) {
      squeeze = sop;
    }
    if (auto bop = dynamic_cast<BroadcastOp*>(expr)) {
      bcast = bop;
    }
    if (!squeeze || !bcast) {
      continue;
    }
    bool is_squeeze_bcast =
        DependencyCheck::isDependencyOf(squeeze->out(), bcast->in());
    std::cout << "Found squeeze-broadcast pattern: " << is_squeeze_bcast
              << std::endl;
    // If dim mismatch, drop the downstream op and check the next pair
    if (bcast->getBroadcastDimFlags() != squeeze->getSqueezeDimFlags()) {
      bcast = is_squeeze_bcast ? nullptr : bcast;
      squeeze = is_squeeze_bcast ? squeeze : nullptr;
    }
    // Further check all the tvs between squeeze and broadcast
    bool can_be_removed = true;
    auto forward_tv_chains = tvChains(DependencyCheck::getAllUseChains(
        is_squeeze_bcast ? squeeze->in() : bcast->in()));
    std::unordered_set<TensorView*> all_consumers;
    for (auto forward_tv_dep_chain : forward_tv_chains) {
      std::cout << "=========== forward_tv_dep_chain: " << std::endl;
      TensorView* forward_running_producer = nullptr;
      TensorView* forward_running_consumer = forward_tv_dep_chain.front();
      forward_tv_dep_chain.pop_front();
      while (!forward_tv_dep_chain.empty()) {
        all_consumers.insert(forward_running_consumer);
        forward_running_producer = forward_running_consumer;
        forward_running_consumer = forward_tv_dep_chain.front();
        forward_tv_dep_chain.pop_front();
        // if a squeeze op is found, stop checking
        std::cout << "Producer: " << forward_running_producer->toString()
                  << " Consumer: " << forward_running_consumer->toString()
                  << std::endl;
        if (auto sop = dynamic_cast<SqueezeOp*>(
                forward_running_consumer->definition())) {
          std::cout << "Found SqueezeOp: " << sop->toString() << std::endl;
          break;
        }
        // if a tv in a use chain has multiple producers
        // e.g. `Z = broadcast(X) + Y`, if replace broadcast(X) with X, must
        // ensure Y = f(broadcast(X))
        const auto& producers =
            ir_utils::producerTvsOf(forward_running_consumer);
        for (auto producer : producers) {
          if (all_consumers.count(producer) == 0) {
            can_be_removed = false;
            std::cout << "invalid producer: " << producer->toString()
                      << std::endl;
            break;
          }
        }
        if (!can_be_removed) {
          break;
        }
      }
      if (!can_be_removed) {
        break;
      }
    }
    if (can_be_removed) {
      for (auto consumer : all_consumers) {
        std::cout << "loop over all_bcast_consumers: " << consumer->toString()
                  << std::endl;
        if (!consumer->hasBroadcast()) {
          continue;
        }
        std::cout << "\nClear broadcast iter domains for "
                  << consumer->toString() << std::endl;
        consumer->clearBroadcastIterDomains(squeeze->getSqueezeDimFlags());
        const auto& producers = ir_utils::producerTvsOf(consumer);
        std::cout << "check its producers2 " << producers.size() << std::endl;
        for (auto producer : producers) {
          std::cout << "producer " << producer->toString() << std::endl;
          if (all_consumers.count(producer) == 0 && producer->hasBroadcast()) {
            std::cout << "Producer: " << producer->toString()
                      << " is not in all_bcast_consumers" << std::endl;
            producer->clearBroadcastIterDomains(squeeze->getSqueezeDimFlags());
          } else {
            std::cout << "skip this producer,producer->hasBroadcast(): "
                      << producer->hasBroadcast()
                      << ", in all_bcast_consumers: "
                      << all_consumers.count(producer) << std::endl;
          }
        }
      }
      std::cout << "loop over all_bcast_consumers is done" << std::endl;

      // convert bcast & squeeze to set
      IrBuilder::create<LoadStoreOp>(
          LoadStoreOpType::Set, bcast->out(), bcast->in());
      IrBuilder::create<LoadStoreOp>(
          LoadStoreOpType::Set, squeeze->out(), squeeze->in());

      bcast = nullptr;
      squeeze = nullptr;
    } else {
      bcast = is_squeeze_bcast ? nullptr : bcast;
      squeeze = is_squeeze_bcast ? squeeze : nullptr;
    }
  }
}

} // namespace

void RemoveBcastSqueeze::runPass(Fusion* fusion) {
  fusion->printMath();
  removeBcastSqueeze(fusion);
  fusion->printMath();
}

} // namespace nvfuser::preseg_passes
