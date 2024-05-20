// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <ir/utils.h>
#include <ops/alias.h>
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

// replaces input to the bcast op that produes cast_output, return the new
// cast_output
Val* replaceInputInBroadcast(Val* cast_output, Val* new_input) {
  auto tmp_expr = cast_output->definition();
  // short-cut for cases when no substitution is needed;
  if (cast_output == new_input || new_input == tmp_expr->input(0)) {
    return cast_output;
  }
  auto new_expr = nvfuser::ir_utils::replaceValInExprInputs(
      tmp_expr, tmp_expr->input(0), new_input);
  return new_expr->output(0);
}

void removeBcastSqueeze(Fusion* fusion) {
  // std::unordered_map<Val*, Val*> replacement_map;
  // Find broadcast + squeeze pattern
  BroadcastOp* bcast_op = nullptr;
  SqueezeOp* squeeze_op = nullptr;
  const auto& exprs = fusion->exprs();
  for (auto expr : exprs) {
    std::cout << "Check Expr: " << expr->toString() << std::endl;
    if (auto sop = dynamic_cast<SqueezeOp*>(expr)) {
      squeeze_op = sop;
      std::cout << "SqueezeOp found" << std::endl;
    } else if (auto bcast = dynamic_cast<BroadcastOp*>(expr)) {
      bcast_op = bcast;
      std::cout << "BroadcastOp found" << std::endl;
    }
    if (!squeeze_op || !bcast_op) {
      std::cout << "continue" << std::endl;
      continue;
    }
    NVF_ERROR(squeeze_op != nullptr && bcast_op != nullptr,
              "Found a broadcast and squeeze op, but one of them is null");
    std::cout << "Found broadcast and squeeze op" << std::endl;
    // we have both broadcast and squeeze
    bool is_squeeze_bcast =
        DependencyCheck::isDependencyOf(squeeze_op->out(), bcast_op->in());
    bool is_bcast_squeeze =
        DependencyCheck::isDependencyOf(bcast_op->out(), squeeze_op->in());
    std::cout << "bcast_op: " << bcast_op->toString() << std::endl;
    std::cout << "squeeze_op: " << squeeze_op->toString() << std::endl;
    std::cout << "is_squeeze_bcast: " << is_squeeze_bcast
              << " is_bcast_squeeze: " << is_bcast_squeeze << std::endl;
    if (is_bcast_squeeze) {
      bool can_be_removed = true;
      auto forward_tv_chains =
          tvChains(DependencyCheck::getAllUseChains(bcast_op->out()));
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
            if (all_bcast_consumers.count(producer) == 0) {
              // if the producer is calculated from inputs, we can squeeze the
              // input
              std::cout << "Producer: " << producer->toString()
                        << " is not in all_bcast_consumers" << std::endl;
              for (auto pp : ir_utils::producerTvsOf(producer)) {
                if (!pp->isFusionInput()) {
                  std::cout << "Producer: " << producer->toString()
                            << " is not a fusion input" << std::endl;
                  can_be_removed = false;
                  break;
                }
              }
              if (!can_be_removed) {
                break;
              }
            }
          }
        }
        if (!can_be_removed) {
          break;
        }
      }
      std::cout << "Can be removed: " << can_be_removed << std::endl;
      if (can_be_removed) {
        for (auto consumer : all_bcast_consumers) {
          std::cout << "loop over all_bcast_consumers: " << consumer->toString() << std::endl;
          if (!consumer->hasBroadcast()) {
            continue;
          }
          std::cout << "\nClear broadcast iter domains for "
                    << consumer->toString() << std::endl;
          consumer->clearBroadcastIterDomains(squeeze_op->getSqueezeDimFlags());
          const auto& producers = ir_utils::producerTvsOf(consumer);
          std::cout << "check its producers2 " << producers.size() << std::endl;
          for (auto producer : producers) {
            std::cout << "producer " << producer->toString() << std::endl;
            if (all_bcast_consumers.count(producer) == 0 &&
                producer->hasBroadcast()) {
              std::cout << "Producer: " << producer->toString()
                        << " is not in all_bcast_consumers" << std::endl;
              producer->clearBroadcastIterDomains(
                  squeeze_op->getSqueezeDimFlags());
              // if the producer is calculated from inputs, we can squeeze the
              // input
              for (auto pp : ir_utils::producerTvsOf(producer)) {
                if (pp->isFusionInput()) {
                  std::cout << "Producer's producer: " << pp->toString()
                            << " is a fusion input" << std::endl;
                  auto pp_squeeze =
                      squeeze(pp, squeeze_op->getSqueezeDimFlags());
                  ir_utils::replaceValInExprInputs(
                      producer->definition(), pp, pp_squeeze);
                }
              }
            }else{
              std::cout << "skip this producer,producer->hasBroadcast(): " << producer->hasBroadcast() << ", in all_bcast_consumers: " << all_bcast_consumers.count(producer) << std::endl;
            }
          }
        }
        std::cout << "loop over all_bcast_consumers is done"  << std::endl;

        // convert bcast & squeeze to set
        IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, bcast_op->out(), bcast_op->in());
        IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, squeeze_op->out(), squeeze_op->in());

        bcast_op = nullptr;
        squeeze_op = nullptr;
      }
    }
  }
}

} // namespace

void RemoveBcastSqueeze::runPass(Fusion* fusion) {
  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
  }
  debug() << "\n========Fusion before RemoveBcastSqueeze:" << std::endl;
  fusion->printMath();

  removeBcastSqueeze(fusion);

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
  }
  debug() << "\n==========Fusion after RemoveBcastSqueeze:" << std::endl;
  fusion->printMath();
}

} // namespace nvfuser::preseg_passes
