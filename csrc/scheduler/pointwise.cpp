// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <scheduler/pointwise.h>

#include <ATen/cuda/CUDAContext.h>
#include <instrumentation.h>
#include <scheduler/debug_utils.h>
#include <scheduler/pointwise_non_tma.h>
#include <scheduler/pointwise_tma.h>
#include <scheduler/pointwise_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>
#include <scheduler/utils.h>
#include <ranges>

namespace nvfuser {

namespace {

// This propagator checks that TransformPropagator will be able to properly
// schedule the entire Fusion. To do so, it tracks logical IterDomains that are
// introduced along the propagation path.
//
//  Example 1:
//
//    addInput(tv0)         // [ i0 ]
//    addInput(tv1)         // [ i0, i1 ]
//    tv2 = neg(tv0)        // [ i0 ]
//    tv3 = broadcast(tv2)  // [ i0, 1 ]
//    tv4 = mul(tv3, tv1)   // [ i0, i1 ]
//    addOutput(tv4)
//
//  In this example, the broadcast of tv2 is concretized when multiplying by
//  tv1, which is 2D. If we propagate from tv2, then we introduce that
//  broadcast dimension which is then concretized as i1, so we detect an
//  unscheduled concrete ID. Note that this should not happen as tv2 will not be
//  selected as a potential reference tensor.
//
//  Example 2:
//
//    addInput(tv0)         // [ i0, 1 ]
//    addInput(tv1)         // [ i0, i1 ]
//    tv2 = squeeze(tv0)    // [ i0 ]
//    tv3 = neg(tv2)        // [ i0 ]
//    tv4 = mul(tv0, tv1)   // [ i0, i1 ]
//    addOutput(tv3)
//    addOutput(tv4)
//
//  In this example, if we propagate from the output tv3, the backward
//  propagation from tv2 to tv0 picks up a broadcast ID. The broadcast is
//  concretized in the multiplication with tv1, so we have an unscheduled i1 ID
//  in the output tv4. Note that tv2 should not be chosen as a reference tensor
//  anyway as it does not have all concrete IDs.
//
//  Example 3:
//
//    addInput(tv0)         // [ i0 ]
//    addInput(tv1)         // [ i1 ]
//    tv2 = broadcast(tv0)  // [ i0, 1 ]
//    tv3 = broadcast(tv1)  // [ 1, i1 ]
//    tv4 = mul(tv2, tv3)   // [ i0, i1 ]
//    tv5 = broadcast(tv0)  // [ i0, 1 ]
//    tv6 = broadcast(tv1)  // [ 1, i1 ]
//    tv7 = add(tv5, tv6)   // [ i0, i1 ]
//    addOutput(tv4)
//    addOutput(tv7)
//
//  If we choose tv4 as the reference then a possible propagation path is
//      4->2->0->5->7->6
//      4->3->1
//  The 2->0 propagation loses the broadcast dimension and then subsequently
//  propagating 0->5->7 means we have concrete dimension i1 unscheduled in tv7,
//  even though there is a scheduled i1 axis in the reference tensor tv4.
//
//  Example 4:
//
//    Suppose instead of creating new broadcasts, we reuse the broadcasts tv2
//    and tv3:
//
//    addInput(tv0)         // [ i0 ]
//    addInput(tv1)         // [ i1 ]
//    tv2 = broadcast(tv0)  // [ i0, 1 ]
//    tv3 = broadcast(tv1)  // [ 1, i1 ]
//    tv4 = mul(tv2, tv3)   // [ i0, i1 ]
//    tv7 = add(tv2, tv3)   // [ i0, i1 ]
//    addOutput(tv4)
//    addOutput(tv7)
//
//  Now the propagation may look like this instead:
//      4->2->0
//         2->7
//      4->3->1
//  None of these steps loses an ID that later is needed for scheduling, so
//  hasUnscheduledConcreteIDs() returns false.
class CoveredDomainPropagator : public MaxInfoSpanningTree::Propagator {
 public:
  void propagateC2P(TensorView* from, TensorView* to) override {
    if (to->isFusionInput()) {
      // We are not concerned with scheduling fusion inputs during transform
      // propagation
      return;
    }
    std::unordered_map<IterDomain*, IterDomain*> c2p =
        PairwiseLogicalDomainMap(to, from)
            .mapBroadcast(true)
            .mapDifferentExtents(true)
            .mapIndexedDomains(true)
            .mapConsumerToProducer();
    check(from->getMaybeRootDomain(), to->getLogicalDomain(), c2p);
    if (to->hasRoot()) {
      // propagate untracked property through root->logical transforms
      for (Expr* e : std::ranges::views::reverse(StmtSort::getExprsBetween(
               {to->getMaybeRootDomain().begin(),
                to->getMaybeRootDomain().end()},
               {to->getLogicalDomain().begin(),
                to->getLogicalDomain().end()}))) {
        bool has_unscheduled_output = std::any_of(
            e->outputs().begin(), e->outputs().end(), [&](Val* out_val) {
              auto* id = dynamic_cast<IterDomain*>(out_val);
              return id && unscheduled_ids_.count(id);
            });
        if (has_unscheduled_output) {
          for (Val* in_val : e->inputs()) {
            unscheduled_ids_.insert(in_val->as<IterDomain>());
          }
        }
      }
    }
  }
  void propagateP2C(TensorView* from, TensorView* to) override {
    std::unordered_map<IterDomain*, IterDomain*> p2c =
        PairwiseLogicalDomainMap(from, to)
            .mapBroadcast(true)
            .mapDifferentExtents(true)
            .mapIndexedDomains(true)
            .mapProducerToConsumer();
    check(from->getLogicalDomain(), to->getMaybeRootDomain(), p2c);
    if (to->hasRoot()) {
      // propagate untracked property through root->logical transforms
      for (Expr* e : StmtSort::getExprsBetween(
               {to->getMaybeRootDomain().begin(),
                to->getMaybeRootDomain().end()},
               {to->getLogicalDomain().begin(),
                to->getLogicalDomain().end()})) {
        // TODO: should we exclude ID exprs other than Merge/Split here?
        bool has_unscheduled_input =
            std::ranges::any_of(e->inputs(), [&](Val* in_val) {
              auto* id = dynamic_cast<IterDomain*>(in_val);
              return id && unscheduled_ids_.count(id);
            });
        if (has_unscheduled_input) {
          for (Val* out_val : e->outputs()) {
            unscheduled_ids_.insert(out_val->as<IterDomain>());
          }
        }
      }
    }
  }
  void propagateSibling(TensorView* from, TensorView* to) override {
    // Siblings require no special consideration in this check
  }

  bool hasUnscheduledConcreteIDs() const {
    for (IterDomain* id : unscheduled_ids_) {
      if (!id->isBroadcast()) {
        return true;
      }
    }
    return false;
  }

 private:
  void check(
      const std::vector<IterDomain*>& from_domain,
      const std::vector<IterDomain*>& to_domain,
      const std::unordered_map<IterDomain*, IterDomain*>& f2t) {
    std::unordered_set<IterDomain*> seen_to;
    for (IterDomain* from_id : from_domain) {
      auto it = f2t.find(from_id);
      if (it != f2t.end()) {
        IterDomain* to_id = it->second;
        if (unscheduled_ids_.count(from_id)) {
          unscheduled_ids_.insert(to_id);
        }
        seen_to.insert(to_id);
      }
    }
    for (IterDomain* to_id : to_domain) {
      if (!seen_to.count(to_id)) {
        // This is a new ID introduced along the propagation path
        unscheduled_ids_.insert(to_id);
      }
    }
  }

 private:
  std::unordered_set<IterDomain*> unscheduled_ids_;
};

} // namespace

//! Utility for canSchedule interface to check if this fusion has
//!  a fully broadcasted reference tensor, which is necessary for
//!  the pointwise scheduler.
bool hasReferenceTensorView(Fusion* fusion) {
  TensorView* reference = pointwise_utils::getReferenceTensor(fusion);
  if (reference == nullptr) {
    return false;
  }

  // If we can find a reference TV, verify that propagation will not need to
  // schedule any IDs that were lost along the propagation path
  MaxLogicalDomainInfoSpanningTree tree(
      reference,
      /*selector=*/nullptr,
      /*propagate_through_resize=*/true);
  CoveredDomainPropagator propagator;
  tree.traverse(&propagator);
  return !propagator.hasUnscheduledConcreteIDs();
}

bool PointWiseScheduler::canScheduleCompileTime(Fusion* fusion) {
  if (scheduler_utils::isResharding(fusion)) {
    FUSER_PERF_SCOPE("PointWiseScheduler::canScheduleCompileTime");
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Fusion is resharding.");
    return false;
  }

  // Currently using the same path as the scheduler
  // to eliminate mismatch between canSchedule and
  // schedule pointwise.
  if (!hasReferenceTensorView(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "cannot find reference tensor");
    return false;
  }

  // Check that inputs of all select/gather-like ops are fusion inputs
  if (registry_utils::rejectScheduleForMemoryPromotion(
          fusion, schedulerType())) {
    return false;
  }

  if (!ir_utils::getReshapeOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "Fusion requires view being reversible.");
      return false;
    }
  }

  if (ir_utils::hasAnyReductionOps(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "no support for reduction ops");
    return false;
  }

  if (registry_utils::hasNonUniqueBcast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Broadcasting dimension might be broadcasting to multiple sizes.");
    return false;
  }

  // The block scales output of the Block Quantization Op
  // should be a segment output as it is written to the global
  // memory.
  if (registry_utils::hasNonTerminalBlockQuantizeOp(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "no support for block quantization where block scales is not a fusion "
        "output");
    return false;
  }

  return true;
}

bool PointWiseScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("PointWiseScheduler::canScheduleRunTime");
  // Check if the fusion has a Block Quantization Op
  // If so, ensure that the vectorization factor is at least 2
  // and that the grid y dimension is not split.
  // These are requirements of the current implementation of the
  // Block Quantization Op runtime function.

  auto has_block_quantization_ops =
      HeuristicDataCacheEntry<HeuristicCompileTime::HasBlockQuantizationOps>(
          data_cache,
          [fusion]() {
            return std::make_unique<bool>(
                !ir_utils::getOpsOfType<BlockQuantizationOp>(fusion).empty());
          })
          .get();

  if (has_block_quantization_ops) {
    auto heuristics = computeHeuristics(fusion, runtime_info, data_cache);
    auto pparams = static_cast<const PointwiseParams*>(heuristics.get());
    NVF_ERROR(pparams != nullptr);
    if (pparams->vectorization_factor < 2) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(),
          "Block Quantization Op requires vectorization factor to be at least "
          "2.");
      return false;
    }

    if (pparams->split_grid_y_dim) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(),
          "Block Quantization Op is not supported when splitting grid y "
          "dimension. This is because this will create a serial ID with an "
          "extent > 1. The runtime function implementing block quantization "
          "will currently not be able to handle that.");
      return false;
    }
  }
  return true;
}

namespace {

// TODO: Refine this function to check contiguity, broadcasts, reshapes, etc.
bool mayHaveTmaCompatibleInputs(
    const pointwise_utils::FusionRuntimeProperties& prop) {
  for (auto tv : prop.vectorizable_inputs_outputs) {
    if (!tv->isFusionInput()) {
      continue;
    }
    auto dtype_bits =
        dataTypeSizeBit(tv->getDataType().value(), prop.index_type);
    // Note: The actual element count should consider the breakpoint and be
    // computed individually for each input. Here, the largest output is used
    // as a conservative estimate. If the largest output fails these checks,
    // then no input is suitable for TMA since all inputs are smaller than or
    // equal to the largest output in a pointwise fusion.
    auto elem_count = prop.n_elems;

    // Condition 1: We only support 2D TMA, which requires at least 2 tiles in
    // the inner dimension, each with  at least 16 bytes. This imposes a minimum
    // inner TMA domain size of 2 * 16 bytes. Additionally, skip if the inner
    // TMA domain size equals the total element count, as this would mean the
    // outer TMA domain is 1, which is not a valid 2D TMA configuration.
    const int64_t min_inner_tma_domain_size = 2 * 128 / dtype_bits;
    if (elem_count % min_inner_tma_domain_size != 0 ||
        elem_count == min_inner_tma_domain_size) {
      continue;
    }

    // Condition 2: the input tensor must have cacheable uses
    if (scheduler_utils::getCacheableUses(tv).empty()) {
      continue;
    }

    // TODO: Add checks for reshape, contiguity, allocation domain, etc.
    // TODO: Add performance checks:
    //   - Skip if input size is too small
    //   - Skip if inner TMA domain size is too small

    // Passed all preliminary checks, may be suitable for TMA
    return true;
  }
  return false;
}

// Preliminary check to determine if TMA can be used for this fusion. This
// serves as a fast path to avoid computing full heuristics if TMA is clearly
// not applicable. Passing this check does not guarantee that TMA will be used;
// the final decision is made during heuristics computation.
bool mayUseTma(const pointwise_utils::FusionRuntimeProperties& prop) {
  // Hardware requirement: Don't use TMA for pre-Hopper GPUs
  if (at::cuda::getCurrentDeviceProperties()->major < 9) {
    return false;
  }
  // Check if there are TMA-compatible inputs
  if (!mayHaveTmaCompatibleInputs(prop)) {
    return false;
  }
  return true;
}
} // namespace

std::unique_ptr<HeuristicParams> PointWiseScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("PointWiseScheduler::computeHeuristics");

  auto prop_opt = pointwise_utils::getFusionRuntimeProperties(
      fusion, runtime_info, data_cache);
  // Return default parameters if the fusion is zero-dimensional or zero-size
  if (!prop_opt.has_value()) {
    auto pwise_params = std::make_unique<PointwiseParams>();
    pwise_params->tag = "Pointwise heuristics";
    pwise_params->cparams.index_type = runtime_info.getIndexType();
    return pwise_params;
  }
  const auto& prop = prop_opt.value();

  bool use_tma = mayUseTma(prop) && isOptionEnabled(EnableOption::TmaPointwise);
  std::unique_ptr<HeuristicParams> pparams = nullptr;
  if (use_tma) {
    pparams = pointwise::tma::getPointwiseHeuristics(
        fusion, runtime_info, data_cache, prop);
  }
  // Fallback to non-TMA scheduler if TMA is not applicable
  if (pparams == nullptr) {
    pparams = pointwise::non_tma::getPointwiseHeuristics(
        fusion, runtime_info, data_cache, prop);
  }
  NVF_ERROR(pparams != nullptr);
  return pparams;
}

void PointWiseScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("PointWiseScheduler::schedule");
  auto pparams = dynamic_cast<const PointwiseParams*>(params);
  NVF_ERROR(
      pparams != nullptr,
      "Incorrect parameters sent to PointWiseScheduler::schedule",
      params);
  if (pparams->use_tma_load) {
    pointwise::tma::schedulePointwise(fusion, pparams);
  } else {
    NVF_ERROR(
        !pparams->use_tma_store,
        "Use TMA store without use TMA load is not supported");
    pointwise::non_tma::schedulePointwise(fusion, pparams);
  }
}

} // namespace nvfuser
