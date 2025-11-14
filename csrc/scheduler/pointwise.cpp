// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <scheduler/pointwise.h>

#include <instrumentation.h>
#include <scheduler/debug_utils.h>
#include <scheduler/pointwise_non_tma.h>
#include <scheduler/pointwise_tma.h>
#include <scheduler/registry_utils.h>
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

  return true;
}

<<<<<<< HEAD
// TODO: Inline intermediate operations (avoid inlining unrolled/vectorized
// input/output caches)
void schedulePointwise(Fusion* fusion, const PointwiseParams* pparams) {
  FusionGuard fg(fusion);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  // Cache inputs
  auto cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  auto cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, true);

  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  refineCachePolicy(fusion);

  std::vector<TensorView*> input_tvs;
  {
    auto filtered_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    // Remove hanging tensor views
    for (auto tv : filtered_tvs) {
      if (tv->uses().empty()) {
        continue;
      }
      input_tvs.push_back(tv);
    }
  }
  auto output_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());

  int64_t max_dims = 0;
  for (auto inp : input_tvs) {
    max_dims = std::max(scheduler_utils::nLogicalDims(inp), max_dims);
  }

  for (auto out : output_tvs) {
    max_dims = std::max(scheduler_utils::nLogicalDims(out), max_dims);
  }

  // If everything is zero dim tensors, just return.
  if (max_dims == 0) {
    return;
  }

  TensorView* reference_tv = pointwise_utils::getReferenceTensor(fusion);
  NVF_ERROR(
      reference_tv != nullptr,
      "Could not find a fully broadcasted output to reference schedule on.");
  std::vector<IterDomain*> ref_orig_loop = reference_tv->getLoopDomain();

  scheduler_utils::moveNonConcretizedBroadcastInnermost(fusion, {reference_tv});

  int64_t num_device_dims = numDeviceDims(reference_tv);
  int64_t device_aware_break_point = pparams->break_point + num_device_dims;

  // Positions of rhs and lhs after merging all dimensions.
  int64_t rhs_i = -1;
  int64_t lhs_i = -1;

  if (!ir_utils::getReshapeOps(fusion).empty()) {
    // Propagate reshape transforms through the graph, expecially the reference.
    scheduler_utils::propagateReshapeTransforms(fusion);

    // Reorder reference_tv after propagating the view operation. This will
    // reorder for better merging.
    reference_tv->reorder(
        scheduler_utils::domainReorderAsLogicalMap(reference_tv));
    // Reorder so that DeviceDims are in front
    reorderParallelizedToFront(reference_tv);

    // Break point is relative to logical domain, find the loop domain ID's in
    // the left/right side, we really need the values in domain, but easiest way
    // to do this is with Dependency check which will grab all intermediate
    // values too.
    auto lhs_all_vals = DependencyCheck::getAllValsBetween(
        {ref_orig_loop.begin() + num_device_dims,
         ref_orig_loop.begin() + device_aware_break_point},
        {reference_tv->getLoopDomain().begin() + num_device_dims,
         reference_tv->getLoopDomain().end()});

    std::unordered_set<Val*> lhs_all_vals_set(
        lhs_all_vals.begin(), lhs_all_vals.end());

    auto rhs_all_vals = DependencyCheck::getAllValsBetween(
        {ref_orig_loop.begin() + device_aware_break_point, ref_orig_loop.end()},
        {reference_tv->getLoopDomain().begin() + num_device_dims,
         reference_tv->getLoopDomain().end()});

    std::unordered_set<Val*> rhs_all_vals_set(
        rhs_all_vals.begin(), rhs_all_vals.end());

    // Make sure lhs and rhs groups are disjoint.
    for (auto lhs_val : lhs_all_vals) {
      if (rhs_all_vals_set.count(lhs_val) != 0) {
        std::ostringstream os;
        IrTransformPrinter printer(os);
        printer.printTransforms(reference_tv);
        NVF_THROW(
            "Error in pointwise scheduler. LHS and RHS of the 2D scheduler are "
            "not disjoint. ",
            lhs_val->toString(),
            " belongs to both. device_aware_break_point = ",
            device_aware_break_point,
            ". reference_tv = ",
            reference_tv->toString(),
            " and its transforms are:\n",
            os.str());
      }
    }
    NVF_ERROR(
        !rhs_all_vals.empty(),
        "Expecting at least one dimension in the RHS of the pointwise "
        "scheduler.");

    // Merge rhs, then lhs.
    IterDomain* rhs_id = nullptr;
    IterDomain* lhs_id = nullptr;
    for (int64_t pos = reference_tv->nDims() - 1; pos >= 0; pos--) {
      // Merge from right to left
      auto id = reference_tv->axis(pos);
      if (lhs_all_vals_set.count(id) > 0) {
        if (lhs_id == nullptr) {
          lhs_id = id;
          lhs_i = pos;
        } else {
          reference_tv->merge(pos, lhs_i);
          lhs_i = pos;
          if (rhs_i > lhs_i) {
            rhs_i--;
          }
        }
      } else if (rhs_all_vals_set.count(id) > 0) {
        if (rhs_id == nullptr) {
          rhs_id = id;
          rhs_i = pos;
        } else {
          reference_tv->merge(pos, rhs_i);
          rhs_i = pos;
          if (lhs_i > rhs_i) {
            lhs_i--;
          }
        }
      }
    }
    // Find the iter domains that should be in the lhs, and rhs.
  } else {
    // Don't need to worry about view transformations, just merge reference tv
    // as we normally would.

    std::unordered_map<int64_t, int64_t> loop_reorder_map =
        scheduler_utils::reorderLoopAsAllocationMap(reference_tv);
    if (!loop_reorder_map.empty()) {
      reference_tv->reorder(loop_reorder_map);
    }
    reorderParallelizedToFront(reference_tv);

    // Merge right side of break point
    for (int64_t i = reference_tv->nDims(); i > device_aware_break_point; i--) {
      auto axis_i = i - 1;
      if (rhs_i == -1) {
        rhs_i = axis_i;
      } else {
        reference_tv->merge(axis_i, rhs_i);
        rhs_i = axis_i;
      }
    }
    if (rhs_i >= 0) {
      // If there's an rhs
      reference_tv->reorder({{rhs_i, -1}});
    }

    // Merge left side of break point
    for (int64_t i = device_aware_break_point; i > num_device_dims; i--) {
      auto axis_i = i - 1;
      if (lhs_i == -1) {
        lhs_i = axis_i;
      } else {
        reference_tv->merge(axis_i, lhs_i);
        lhs_i = axis_i;
      }
    }
  }

  int64_t unswitch_pos = 0;
  IterDomain* vectorize_id = nullptr;
  if (pparams->break_point) {
    // 2D parallelization scheme
    NVF_ERROR(rhs_i >= 0 && lhs_i >= 0);

    // Right (inner merged) dimension is at inner most position, left (outer
    // merged) dimension is at lhs_i. Order as [lhs_i, rhs_i, unmerged...]
    reference_tv->reorder({{lhs_i, 0}, {-1, 1}});

    // vectorization without unroll
    if (pparams->unroll_factor_outer == 1 &&
        pparams->unroll_factor_inner == 1 &&
        pparams->vectorization_factor > 1) {
      reference_tv->split(1, pparams->vectorization_factor);
      reference_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDx));
      reference_tv->split(0, 1);
      // [outer, Unswitch | i-remainder, TIDx, Vectorization]
      // Here and in the following comments:
      // prefix [i] represent inner dimension
      // prefix [o] represent inner dimension
      // [|] separates the outer and inner dimensions
      reference_tv->axis(1)->parallelize(ParallelType::Unswitch);
      reference_tv->axis(3)->parallelize(ParallelType::TIDx);
      // Vectorization are propagated separately
      vectorize_id = reference_tv->axis(4);

      // [outer, Unswitch | i-remainder, TIDx, Vectorization]
      // To make consistent with unrolling:
      reference_tv->reorder({{1, 2}, {2, 1}, {3, 4}, {4, 3}});
      //[outer | i-remainder, Unswitch, Vectorization, TIDx]
    } else {
      // [outer | inner]
      if (pparams->vectorization_factor > 1) {
        reference_tv->split(1, pparams->vectorization_factor);
      }
      // [outer | i-remainder, Vect]
      reference_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDx));
      // [outer | i-remainder, TIDx, Vect]

      if (pparams->unroll_factor_inner > 1) {
        reference_tv->split(1, pparams->unroll_factor_inner);
      }
      // [outer| i-remainder, i-Unroll, TIDx, Vect]

      if (pparams->unroll_factor_outer > 1) {
        reference_tv->split(0, pparams->unroll_factor_outer);
      }
      // [o-remainder, o-Unroll, | i-remainder, i-Unroll, TIDx, Vect]

      reference_tv->split(0, 1);
      // [o-remainder, Unswitch, o-Unroll | i-remainder, i-Unroll, TIDx, Vect]

      int i_remainder_pos = pparams->unroll_factor_outer > 1 ? 3 : 2;
      reference_tv->reorder({{i_remainder_pos, 1}});
      // [o-remainder, i-remainder, Unswitch, o-Unroll, i-Unroll, TIDx, Vect]

      reference_tv->axis(2)->parallelize(ParallelType::Unswitch);
      // Here we do not set axis(3)->parallelize(Unroll) because we do not want
      // it to be propagated. We manually unroll by splitting the inline
      // propagation process into two steps:
      // step 1: inline at the unswitch position for cached inputs and outputs
      // step 2: inline at the inner most dim for the rest of the graph
      int tidx_pos = 3;
      if (pparams->unroll_factor_inner > 1) {
        tidx_pos++;
      }
      if (pparams->unroll_factor_outer > 1) {
        tidx_pos++;
      }
      reference_tv->axis(tidx_pos)->parallelize(ParallelType::TIDx);
      if (pparams->vectorization_factor > 1) {
        // can't use {-1}, there may be deviceId
        vectorize_id = reference_tv->axis(tidx_pos + 1);
      }
      // [o-remainder, i-remainder, Unswitch, o-Unroll, i-Unroll, TIDx, Vect]
    }

    // Move out of the way to furthest left point
    reference_tv->reorder({{1, 0}});
    // [i-remainder, o-remainder, Unswitch, o-Unroll, i-Unroll, TIDx, Vect]
    if (pparams->split_block) {
      reference_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));
      // [i-remainder, o-remainder, TIDy, Unswitch, o-Unroll, i-Unroll, TIDx,
      // Vect]
      if (pparams->flip_grid_binding) {
        // [BIDy | BIDx, TIDy | Unswitch, o-Unroll, i-Unroll, TIDx, Vect]
        reference_tv->axis(1)->parallelize(ParallelType::BIDx);
        reference_tv->axis(2)->parallelize(ParallelType::TIDy);
        if (pparams->split_grid_y_dim) {
          // [i-remainder, BIDy{65535} | BIDx, TIDy | Unswitch, o-Unroll,
          // i-Unroll, TIDx, Vect]
          reference_tv->split(0, 65535);
          reference_tv->axis(1)->parallelize(ParallelType::BIDy);
          unswitch_pos = 5;
        } else {
          reference_tv->axis(0)->parallelize(ParallelType::BIDy);
          unswitch_pos = 4;
        }
      } else {
        // [BIDx | BIDy TIDy | Unswitch, o-Unroll, i-Unroll, TIDx, Vect]
        reference_tv->axis(0)->parallelize(ParallelType::BIDx);
        reference_tv->axis(2)->parallelize(ParallelType::TIDy);
        if (pparams->split_grid_y_dim) {
          // [BIDx | i-remainder, BIDy{65535}, TIDy | Unswitch, o-Unroll,
          // i-Unroll, TIDx, Vect]
          reference_tv->split(1, 65535);
          reference_tv->axis(2)->parallelize(ParallelType::BIDy);
          unswitch_pos = 5;
        } else {
          reference_tv->axis(1)->parallelize(ParallelType::BIDy);
          unswitch_pos = 4;
        }
      }
    } else {
      // [BIDy | BIDx | Unswitch, Unroll, TIDx, Vect]
      if (pparams->flip_grid_binding) {
        // [BIDy | BIDx | Unswitch, Unroll, TIDx, Vect]
        reference_tv->axis(1)->parallelize(ParallelType::BIDx);
        if (pparams->split_grid_y_dim) {
          // [i-remainder, BIDy{65535} | BIDx | Unswitch, Unroll, TIDx]
          reference_tv->split(0, 65535);
          reference_tv->axis(1)->parallelize(ParallelType::BIDy);
          unswitch_pos = 4;
        } else {
          // [BIDy | BIDx | Unswitch, Unroll, TIDx, Vect]
          reference_tv->axis(0)->parallelize(ParallelType::BIDy);
          unswitch_pos = 3;
        }
      } else {
        // [BIDx | BIDy | Unswitch, Unroll, TIDx, Vect]
        reference_tv->axis(0)->parallelize(ParallelType::BIDx);
        if (pparams->split_grid_y_dim) {
          // [BIDx | i-remainder, BIDy{65535} | Unswitch, Unroll, TIDx, Vect]
          reference_tv->split(1, 65535);
          reference_tv->axis(2)->parallelize(ParallelType::BIDy);
          unswitch_pos = 4;
        } else {
          // [BIDx | BIDy | Unswitch, Unroll, TIDx, Vect]
          reference_tv->axis(1)->parallelize(ParallelType::BIDy);
          unswitch_pos = 3;
        }
      }
    }
  } else {
    // 1D Scheduler
    NVF_ERROR(rhs_i >= 0 && lhs_i == -1);

    // right hand side exists and is the only axis we care to schedule, move
    // it from the inner most position to left most. Order as [rhs_i,
    // unmerged...]
    reference_tv->reorder({{-1, 0}});

    if (pparams->unroll_factor_inner == 1 &&
        pparams->vectorization_factor > 1) {
      // Vectorize
      reference_tv->split(0, pparams->vectorization_factor);
      // Unswitch
      reference_tv->split(0, 1);
      // Threads
      reference_tv->split(0, kThreadX);

      reference_tv->axis(0)->parallelize(ParallelType::BIDx);
      reference_tv->axis(1)->parallelize(ParallelType::TIDx);
      reference_tv->axis(2)->parallelize(ParallelType::Unswitch);
      // Vectorization are propagated separately
      vectorize_id = reference_tv->axis(3);

      //[BIDx, TIDx, Unswitch, Vectorization]
      // To make consistent with unrolling:
      reference_tv->reorder({{1, 3}, {2, 1}, {3, 2}});
      //[BIDx, Unswitch, Vectorization, TIDx]
    } else {
      // Vectorize
      if (pparams->vectorization_factor > 1) {
        reference_tv->split(0, pparams->vectorization_factor);
      }
      // Threads
      reference_tv->split(0, kThreadX);
      // Unroll
      if (pparams->unroll_factor_inner > 1) {
        reference_tv->split(0, pparams->unroll_factor_inner);
      }
      // Unswitch
      reference_tv->split(0, 1);

      // [BIDx, Unswitch, Unroll, TIDx, Vect]
      reference_tv->axis(0)->parallelize(ParallelType::BIDx);
      reference_tv->axis(1)->parallelize(ParallelType::Unswitch);
      // Here we do not set axis(2)->parallelize(Unroll) because we do not want
      // it to be propagated. We manually unroll by splitting the inline
      // propagation process into two steps:
      // step 1: inline at the unswitch position for cached inputs and outputs
      // step 2: inline at the inner most dim for the rest of the graph
      int tidx_pos = pparams->unroll_factor_inner > 1 ? 3 : 2;
      reference_tv->axis(tidx_pos)->parallelize(ParallelType::TIDx);
      if (pparams->vectorization_factor > 1) {
        vectorize_id = reference_tv->axis(tidx_pos + 1);
      }
    }
    unswitch_pos = 2;
  }

  TransformPropagator propagator(reference_tv);
  MaxLogicalDomainInfoSpanningTree spanning_tree(reference_tv);
  spanning_tree.traverse(&propagator);
  scheduler_utils::parallelizeAllLike(reference_tv);

  if (pparams->vectorization_factor > 1) {
    // Grab all tensor views that should be vectorized
    auto inputs_outputs =
        scheduler_utils::getInputsOutputsWithInnerDim(reference_tv, true, true);
    std::vector<TensorView*> vectorized_tvs;
    bool should_vectorize_reference_tv = false;
    for (auto tv : inputs_outputs) {
      if (tv == reference_tv) {
        should_vectorize_reference_tv = true;
      }
      if (!tv->isFusionInput()) {
        vectorized_tvs.emplace_back(tv);
        continue;
      }
      // move inputs to consumers of inputs
      auto consumer_tvs = ir_utils::consumerTvsOf(tv);
      vectorized_tvs.insert(
          vectorized_tvs.end(), consumer_tvs.begin(), consumer_tvs.end());
    }
    // Vectorize all casts
    if (pparams->vectorize_casts) {
      for (auto tv : fusion->allTvs()) {
        if (auto uop = dynamic_cast<UnaryOp*>(tv->definition())) {
          if (uop->getUnaryOpType() == UnaryOpType::Cast &&
              (dataTypeSizeBit(tv->dtype()) < 8 ||
               dataTypeSizeBit(uop->in()->dtype()) < 8)) {
            vectorized_tvs.emplace_back(tv);
          }
        }
      }
    }
    if (!vectorized_tvs.empty()) {
      // Aggressively mark with vectorized and cleanup later. That way we
      // don't have to manually specify parallelization outside the reference.
      vectorize_id->parallelize(ParallelType::Vectorize);
      scheduler_utils::parallelizeAllLike(
          reference_tv, vectorized_tvs, {ParallelType::Vectorize});
      if (!should_vectorize_reference_tv) {
        vectorize_id->parallelize(ParallelType::Serial);
      }
    }
  }

  // Begin by inlining at the unswitch position for the entire DAG. The cached
  // inputs, and outputs will keep this inline position, but other tensors will
  // get a higher position in later inline propagation. We need this separate
  // step because we were not using ParallelType::Unroll, so we have to do
  // unrolling manually.
  inlineAllAt(reference_tv, unswitch_pos, true);

  auto all_tvs = fusion->allTvs();

  // Inline at the inner most position. The CA position of all tensors except
  // inputs, cached inputs and outputs will be updated.
  std::unordered_set<TensorView*> inner_most_tensors(
      all_tvs.begin(), all_tvs.end());
  for (const auto& [cached_input, input_idx] : cached_inputs) {
    inner_most_tensors.erase(cached_input);
  }
  for (const auto& [cached_output, output_idx] : cached_outputs) {
    auto output = fusion->outputs()[output_idx]->as<TensorView>();
    inner_most_tensors.erase(output);
  }
  // IndexSelectOp reads lookup tv without cache. Because pointwise scheduler
  // doesn't use ParallelType::Unroll, we need to exclude consumer of fusion
  // inputs to be inlineMost. This allows us to aggregate the allocation of
  // manual unroll ID and its inner ID.
  for (auto idx_sel : ir_utils::getOpsOfType<IndexSelectOp>(fusion)) {
    inner_most_tensors.erase(idx_sel->output(0)->as<TensorView>());
  }

  inlineMost(inner_most_tensors);

  scheduler_utils::promoteProducerMemoryTypes(fusion, cached_inputs);

  // TODO(#1401): We could let segmentation split a partially alias-producing
  // fusion into an alias-only segment and the rest. This way, the rest of the
  // fusion (which has fewer expressions) can potentially find a better
  // scheduler and we need to call markAliases only in NoOpScheduler.
  markAliases(fusion);
}

=======
>>>>>>> main
std::unique_ptr<HeuristicParams> PointWiseScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("PointWiseScheduler::computeHeuristics");
  bool use_tma = false;
  std::unique_ptr<HeuristicParams> pparams = nullptr;
  if (use_tma) {
    pparams = pointwise::tma::getPointwiseHeuristics(
        fusion, runtime_info, data_cache);
  } else {
    pparams = pointwise::non_tma::getPointwiseHeuristics(
        fusion, runtime_info, data_cache);
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
