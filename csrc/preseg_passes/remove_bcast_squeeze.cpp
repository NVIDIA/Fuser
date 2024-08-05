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
  if (ir_utils::producerTvsOf(tv).size() > 1) {
    return true;
  }
  if (ir_utils::consumerTvsOf(tv).size() > 1) {
    return true;
  }
  return false;
}
// Remove broadcast-squeeze and squeeze-broadcast patterns
// Limitation: only support one producer and one consumer for tensors between
// broadcast and squeeze
// TODO: extend to allow multiple producers and consumers.
void removeBcastSqueeze(Fusion* fusion) {
  bool is_removed = true;
  // Loop over all exprs in fusion until no more patterns are removed
  while (is_removed) {
    // init the flag to false, if any pattern is removed, set it to true
    is_removed = false;
    // re-fetch the exprs after removing the patterns
    auto exprs = fusion->exprs();
    // Iterate backwards over fusion expressions.
    // This will ensure we don't process expressions that are no longer valid
    // after replacement.
    for (auto it = std::rbegin(exprs); it != std::rend(exprs); it++) {
      Expr* expr = *it;
      // step-1: find and remove broadcast + squeeze pattern
      // before: Y0 = broadcast(X); Yn = PointwiseOps(Y0), Z = squeeze(Yn);
      //         M = someOp(Z)
      // after : M = someOp(PointwiseOps(X))
      // broadcast Ids are removed from all tensors between broadcast and
      // squeeze
      if (auto squeeze = dynamic_cast<SqueezeOp*>(expr)) {
        auto bcast_tv = squeeze->in()->as<TensorView>();
        // If this bcast tv has multiple consumers, don't remove the broadcast
        // id
        if (isMultipleConsumersOrProducers(bcast_tv)) {
          continue;
        }
        std::vector<TensorView*> tvs_between_bcast_squeeze{bcast_tv};
        bool can_remove_bcast_id = true;
        // walk up the producer-consumer chain to find the broadcast op or an
        // input tv, all the tensors in between should has only one producer.
        while (!bcast_tv->definition()->isA<BroadcastOp>() &&
               !bcast_tv->isFusionInput()) {
          if (!isMultipleConsumersOrProducers(bcast_tv)) {
            bcast_tv = ir_utils::getSoleProducerTv(bcast_tv);
            tvs_between_bcast_squeeze.push_back(bcast_tv);
          } else {
            can_remove_bcast_id = false;
            break;
          }
        }
        if (!can_remove_bcast_id) {
          continue;
        }

        // For valid case, we can remove the broadcast id from all tensors
        // between broadcast and squeeze and replace bcast and squeeze with set
        if (auto bcast = dynamic_cast<BroadcastOp*>(bcast_tv->definition())) {
          const auto bcast_dims = bcast->getBroadcastDimFlags();
          if (bcast_dims == squeeze->getSqueezeDimFlags()) {
            for (auto tv : tvs_between_bcast_squeeze) {
              // keep the bcast Id for output tvs, cacheFork creates a set op,
              // change to broadcast op to keep the bcast Ids.
              TensorView* forked_tv = nullptr;
              if (tv->isFusionOutput()) {
                forked_tv = tv->cacheFork();
              }
              tv->clearBroadcastIterDomains(bcast_dims);
              if (forked_tv) {
                IrBuilder::create<BroadcastOp>(forked_tv, tv, bcast_dims);
              }
            }
            // convert bcast & squeeze to set
            IrBuilder::create<LoadStoreOp>(
                LoadStoreOpType::Set, bcast->out(), bcast->in());
            IrBuilder::create<LoadStoreOp>(
                LoadStoreOpType::Set, squeeze->out(), squeeze->in());
            is_removed = true;
          }
        }
      }

      // step-2: find and remove squeeze + broadcast pattern
      // before: Y = squeeze(X); Z = broadcast(Y);  M = someOp(Z)
      // after : M = someOp(X)
      // conditions: (1) broadcast and squeeze have the same dim flags
      if (auto bcast = dynamic_cast<BroadcastOp*>(expr)) {
        if (auto squeeze =
                dynamic_cast<SqueezeOp*>(bcast->in()->definition())) {
          if (bcast->getBroadcastDimFlags() == squeeze->getSqueezeDimFlags()) {
            ir_utils::replaceValInAllExprInputsAndFusionOutputs(
                bcast->out(), squeeze->in());
            is_removed = true;
          }
        }
      }
      // break to re-fetch the exprs after removing the patterns
      if (is_removed) {
        break;
      }
    }
  }
}

} // namespace

void RemoveBcastSqueeze::runPass(Fusion* fusion) {
  removeBcastSqueeze(fusion);
}

} // namespace nvfuser::preseg_passes
