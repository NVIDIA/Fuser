// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <options.h>
#include <preseg_passes/remove_bcast_squeeze.h>
namespace nvfuser::preseg_passes {

namespace {
  inline bool isMultipleConsumersOrProducers(TensorView* tv) {
  if(ir_utils::producerTvsOf(tv).size() > 1){
    return true;
  }
  if(ir_utils::consumerTvsOf(tv).size() > 1){
    return true;
  }
  return false;
}
// Remove broadcast-squeeze and squeeze-broadcast patterns
void removeBcastSqueeze(Fusion* fusion) {
  // set of exprs that are already processed
  std::unordered_set<Expr*> processed_exprs;
  // Iterate backwards over fusion expressions.
  // This will ensure we don't process expressions that are no longer valid
  // after replacement.
  auto exprs = fusion->exprs();
  for (auto it = std::rbegin(exprs); it != std::rend(exprs); it++) {
    Expr* expr = *it;
    if (processed_exprs.find(expr) != processed_exprs.end()) {
      std::cout << "skip " << expr->toString() << std::endl;
      continue;
    }
    std::cout << "visiting " << expr->toString() << std::endl;
    // step-1: find and remove broadcast + squeeze pattern
    // before: Y0 = broadcast(X); Yn = PointwiseOps(Y0), Z = squeeze(Yn);
    //         M = someOp(Z)
    // after : M = someOp(PointwiseOps(X))
    // broadcast Ids are removed from all tensors between broadcast and
    // squeeze
    if (auto squeeze = dynamic_cast<SqueezeOp*>(expr)) {
      // if (auto bcast =
      //         dynamic_cast<BroadcastOp*>(squeeze->in()->definition())) {
      //   if (bcast->getBroadcastDimFlags() == squeeze->getSqueezeDimFlags()) {
      //     ir_utils::replaceValInAllExprInputsAndFusionOutputs(
      //         squeeze->out(), bcast->in());
      //     std::cout << "\n after removing s+b" << std::endl;
      //     fusion->printMath();
      //     continue;
      //   }
      // }

      auto bcast_tv = squeeze->in()->as<TensorView>();
      // If this bcast tv has multiple consumers, don't remove the broadcast id
      if(isMultipleConsumersOrProducers(bcast_tv)){
        continue;
      }
      std::vector<TensorView*> tvs_between_bcast_squeeze{bcast_tv};
      // don't want to remove bcast Id from fusion outputs
      bool can_remove_bcast_id = !bcast_tv->isFusionOutput();
      // walk up the producer-consumer chain to find the broadcast op or an
      // input tv, all the tensors in between should has only one producer.
      // TODO: extend to allow multiple producers.
      while (!bcast_tv->definition()->isA<BroadcastOp>() &&
             !bcast_tv->isFusionInput() && can_remove_bcast_id) {
        const auto& producers = ir_utils::producerTvsOf(bcast_tv);
        const auto& consumers = ir_utils::consumerTvsOf(bcast_tv);
        std::cout << bcast_tv->toString() << " has " << producers.size()
                  << " producers and " << consumers.size() << " consumers"
                  << std::endl;
        if (producers.size() == 1 && consumers.size() == 1 &&
            !producers.at(0)->isFusionOutput()) {
          bcast_tv = producers.at(0);
          tvs_between_bcast_squeeze.push_back(bcast_tv);
        } else {
          can_remove_bcast_id = false;
          break;
        }
      }
      // if can't remove the broadcast id, e.g. output, continue to next expr
      if (!can_remove_bcast_id) {
        continue;
      }

      // For valid case, we can remove the broadcast id from all tensors
      // between broadcast and squeeze and replace bcast and squeeze with set
      if (auto bcast = dynamic_cast<BroadcastOp*>(bcast_tv->definition())) {
        if (bcast->getBroadcastDimFlags() == squeeze->getSqueezeDimFlags()) {
          for (auto tv : tvs_between_bcast_squeeze) {
            tv->clearBroadcastIterDomains(bcast->getBroadcastDimFlags());
          }
          // convert bcast & squeeze to set
          IrBuilder::create<LoadStoreOp>(
              LoadStoreOpType::Set, bcast->out(), bcast->in());
          IrBuilder::create<LoadStoreOp>(
              LoadStoreOpType::Set, squeeze->out(), squeeze->in());
        }
      }
    }

    // step-2: find and remove squeeze + broadcast pattern
    // before: Y = squeeze(X); Z = broadcast(Y);  M = someOp(Z)
    // after : M = someOp(X)
    // conditions: (1) broadcast and squeeze have the same dim flags
    if (auto bcast = dynamic_cast<BroadcastOp*>(expr)) {
      if (auto squeeze = dynamic_cast<SqueezeOp*>(bcast->in()->definition())) {
        if (bcast->getBroadcastDimFlags() == squeeze->getSqueezeDimFlags()) {
          ir_utils::replaceValInAllExprInputsAndFusionOutputs(
              bcast->out(), squeeze->in());
          // output of squeeze is no longer used
          processed_exprs.insert(squeeze);
          std::cout << "\n after removing s+b" << std::endl;
          fusion->printMath();
        }
      }
    }
  }
}

} // namespace

void RemoveBcastSqueeze::runPass(Fusion* fusion) {
  removeBcastSqueeze(fusion);
}

} // namespace nvfuser::preseg_passes
