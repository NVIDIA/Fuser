// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <debug.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <scheduler/cache_policy_refiner.h>
#include <scheduler/debug_utils.h>
#include <scheduler/mark_aliases.h>
#include <scheduler/pointwise_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/resize.h>
#include <scheduler/resize_heuristic.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/tools/loop_domain_scheduler.h>
#include <scheduler/tools/resize_utils.h>
#include <val_graph_visitor.h>

namespace nvfuser {

namespace {

// Just use the pointwise version for now
TensorView* getReferenceTensor(Fusion* fusion) {
  return pointwise_utils::getReferenceTensor(fusion);
}

// Returns the largest tensor with its number of elements
std::pair<TensorView*, int64_t> getLargestTensor(
    const std::vector<Val*>& vals,
    SchedulerRuntimeInfo& runtime_info) {
  int64_t max_num_elms = -1;
  TensorView* largest_tv = nullptr;
  for (auto tv : ir_utils::filterByType<TensorView>(vals)) {
    int64_t num_elms = 1;
    for (auto logical_id : tv->getLogicalDomain()) {
      auto inferred_val =
          runtime_info.expressionEvaluator().evaluate(logical_id->extent());
      NVF_ERROR(
          inferred_val.hasValue(),
          "Error inferring extent of: ",
          logical_id->toString());
      num_elms *= inferred_val.as<int64_t>();
    }
    if (num_elms > max_num_elms) {
      largest_tv = tv;
      max_num_elms = num_elms;
    }
  }
  return std::make_pair(largest_tv, max_num_elms);
}

} // namespace

bool ResizeScheduler::canScheduleCompileTime(Fusion* fusion) {
  if (!isOptionEnabled(EnableOption::ResizeScheduler)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Not enabled");
    return false;
  }

  if (!ir_utils::hasOpsOfType<SliceOp, PadOp>(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "No resize op to schedule");
    return false;
  }

  if (scheduler_utils::isResharding(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Fusion is resharding.");
    return false;
  }

  if (ir_utils::hasAnyReductionOps(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "No support for reduction ops");
    return false;
  }

  if (registry_utils::hasNonUniqueBcast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Broadcasting dimension might be broadcasting to multiple sizes.");
    return false;
  }

  // For now, the resize scheduler is only allowed for a limited set
  // of fusion patterns. The restrictions are planned to be
  // incrementally relaxed.

  IdModel id_model(fusion, /*build_graphs=*/false);
  const auto& broadcast_graph = id_model.buildBroadcastGraph();

  auto resize_tensor_ops = ir_utils::getOpsOfType<SliceOp, PadOp>(fusion);

  // Slicing of or to a broadcast ID is not allowed yet.
  for (auto resize_tensor_op : resize_tensor_ops) {
    TensorView* out_tv = resize_tensor_op->output(0)->as<TensorView>();
    for (auto logical_id : out_tv->getLogicalDomain()) {
      Resize* resize = dynamic_cast<Resize*>(logical_id->definition());
      if (resize == nullptr) {
        continue;
      }

      if (resize->out()->isBroadcast()) {
        scheduler_debug_utils::canScheduleRejectReason(
            schedulerType(), "Resize to a broadcast ID is not allowed.");
        return false;
      }

      // Need to check the broadcast group rather than just the input
      // ID only. For example,
      //
      // t0: [i0]
      // t1: [b1]
      // t2 = t0 + t1
      // t3 = slice(t2)
      //
      // Then, propagating the slice to its inputs would try to
      // propagate the resize op to b1 as well, which would fail due
      // to issue #3571
      const auto& input_group = broadcast_graph.toGroup(resize->in());
      if (std::any_of(
              input_group->begin(), input_group->end(), [](Val* inp_val) {
                return inp_val->as<IterDomain>()->isBroadcast();
              })) {
        scheduler_debug_utils::canScheduleRejectReason(
            schedulerType(), "Resize of a broadcast ID is not allowed.");
        return false;
      }
    }
  }

  auto ref_tv = getReferenceTensor(fusion);
  if (ref_tv == nullptr) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "No reference found");
    return false;
  }

  // Having different resizes between outputs is not allowed at this
  // moment. For example, consider a fusion like:
  //
  // t0 = [i0]
  // fusion.addInput(t0)
  // t1 = t0[:i0/2]
  // t2 = t0[i0/2:]
  // fusion.addOutput(t1)
  // fusion.addOutput(t2)
  //
  // For now, this is not going to be fused since t1 and t2 have
  // different resize ops, although in this case, since the extents of t1 and
  // t2 are the same, it should be relatively straightforward to fuse them
  // together.
  for (auto out_tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    if (out_tv == ref_tv) {
      continue;
    }
    auto exprs = ValGraphBFS::getExprGroupsBetween(
                     broadcast_graph,
                     broadcast_graph.toGroups(ref_tv->getLogicalDomain()),
                     broadcast_graph.toGroups(out_tv->getLogicalDomain()),
                     /*require_all_to_visited=*/false)
                     .first;
    for (const auto& [expr_g, dir] : exprs) {
      if (expr_g->front()->isA<Resize>()) {
        std::stringstream msg;
        msg << "Resize between reference and output not allowed.";
        msg << " Reference: " << ref_tv->toString()
            << ". Output: " << out_tv->toString()
            << ". Resize: " << expr_g->front()->toString();
        scheduler_debug_utils::canScheduleRejectReason(
            schedulerType(), msg.str());
        return false;
      }
    }
  }

  // Skip transpose-like patterns for now
  scheduler_tools::TransposeDomainMap domain_map(fusion);
  auto grouped_inputs_outputs = domain_map.groupInputsOutputsByInnerDim();
  if (grouped_inputs_outputs.size() >= 2) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Transpose-like patterns not supported.");
    return false;
  }

  return true;
}

std::unique_ptr<HeuristicParams> ResizeScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("ResizeScheduler::computeHeuristics");
  auto params = std::make_unique<ResizeParams>(SchedulerType::Resize);
  params->tag = "Resize heuristics";
  params->cparams.index_type = runtime_info.getIndexType();

  const int64_t bdimx = 128;

  const auto& [largest_output, max_num_elms] =
      getLargestTensor(fusion->outputs(), runtime_info);

  params->split_grid_x_dim =
      ceilDiv(max_num_elms, bdimx) > ResizeParams::max_gdimx;

  const auto largest_input =
      getLargestTensor(fusion->inputs(), runtime_info).first;
  if (largest_input != nullptr) {
    int64_t index_of_largest_input = std::distance(
        fusion->inputs().begin(),
        std::find(
            fusion->inputs().begin(), fusion->inputs().end(), largest_input));
    params->largest_input = index_of_largest_input;
  } else {
    params->largest_input = -1;
  }

  // Vectorization based on the largest input if there's any input
  // tv. This is because the current heuristics are designed to
  // optimize the read performance. The largest output is used if
  // there's no input.
  auto ref_tv_for_vectorization =
      largest_input != nullptr ? largest_input : largest_output;
  // Only consider the innermost dimension to vectorize for now.
  // TODO: Consider vectorizing merged IDs, not just the innermost
  params->vectorization_factor = vectorize_helper::getVectorizationFactor(
      runtime_info,
      ref_tv_for_vectorization,
      data_cache,
      (int64_t)ref_tv_for_vectorization->getLogicalDomain().size() - 1,
      {});

  return params;
}

void ResizeScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  FUSER_PERF_SCOPE("ResizeScheduler::schedule");

  FusionGuard fg(fusion);
  const auto resize_params = dynamic_cast<const ResizeParams*>(params);
  NVF_ERROR(resize_params != nullptr);

  scheduler_utils::clearMemorySpace(fusion);

  scheduler_utils::cacheInputs(fusion, true);
  scheduler_utils::cacheAndForkOutputs(fusion, true);

  auto resize_tensor_ops = ir_utils::getOpsOfType<SliceOp, PadOp>(fusion);

  IdModel id_model(fusion, /*build_graphs=*/false);
  const auto& exact_graph = id_model.buildExactGraph();

  // Replicate resize inputs if necessary to avoid conflicting
  // propagations
  const auto exclusivity_info_map = scheduler_tools::getNonExclusiveResizeInfo(
      resize_tensor_ops, exact_graph);
  for (auto resize_tensor_op : resize_tensor_ops) {
    auto out_tv = resize_tensor_op->output(0)->as<TensorView>();
    if (exclusivity_info_map.count(out_tv) == 0) {
      continue;
    }
    auto inp_tv = resize_tensor_op->input(0)->as<TensorView>();
    // Since cacheInput may skip caching if an input is used by
    // slice/pad, inp_tv may be a fusion input, in which case it is
    // not necessary to recompute the tensor.
    if (inp_tv->isFusionInput()) {
      continue;
    }
    auto inp_tv_copy = RecomputeTv::recompute(inp_tv);
    ir_utils::replaceValInExprInputs(resize_tensor_op, inp_tv, inp_tv_copy);
  }

  TensorView* largest_input = nullptr;
  if (resize_params->largest_input >= 0) {
    largest_input =
        fusion->inputs().at(resize_params->largest_input)->as<TensorView>();

    // The tensors are going to be reordered to align with the largest
    // input. To make it work, merge operations for reshape should be
    // cancelled.
    scheduler_tools::cancelReshapeInLoopDomains(largest_input);
  }

  for (auto expr : fusion->exprs()) {
    if (!expr->isOneOf<SliceOp, PadOp>()) {
      continue;
    }

    scheduler_tools::propagateResizeToInputs(expr);
  }

  auto ref_tv = getReferenceTensor(fusion);
  NVF_ERROR(ref_tv != nullptr);

  // Just simple scheduling for now.
  // TODO: Do something smarter. Can just use the pointwise scheduler?

  // Reorder tensors to align with the largest input. This is expected
  // to improve the memory read performance, while the write
  // performance could be lowered. This should generally be more
  // important to optimize the read performance, but more robust
  // decision would be needed.
  if (largest_input != nullptr) {
    std::vector<IterDomain*> ref_alloc;
    ref_alloc.reserve(largest_input->getMaybeAllocationDomain().size());
    std::copy_if(
        largest_input->getMaybeAllocationDomain().begin(),
        largest_input->getMaybeAllocationDomain().end(),
        std::back_inserter(ref_alloc),
        [](IterDomain* alloc_id) {
          return !alloc_id->isBroadcast() && !alloc_id->isReduction() &&
              !alloc_id->isDeviceDim();
        });

    // Reorder the reference as the allocation domain of the largest fusion
    // input
    scheduler_utils::reorderTensorLike(ref_tv, ref_alloc);
  }

  // Make sure the DID ID located at the outermost position
  const auto outermost_pos = scheduler_utils::reorderDevicesToOuter(ref_tv);

  const int64_t vec_factor = resize_params->vectorization_factor;

  const int64_t bdimx = 128;

  int64_t next_innermost_pos = -1;
  // [..., ...]
  //        ^
  //        +--- next_innermost_pos

  if (vec_factor > 1) {
    ref_tv->split(-1, vec_factor);
    --next_innermost_pos;
    // [..., vec_factor]
    //   ^
    //   +--- next_innermost_pos
  }

  ref_tv->flatten(outermost_pos, next_innermost_pos);
  // [..., I0, vec_factor]
  //       ^
  //       +--- next_innermost_pos

  ref_tv->split(next_innermost_pos, bdimx);
  ref_tv->axis(next_innermost_pos)->parallelize(ParallelType::TIDx);
  --next_innermost_pos;
  // [..., I0/bdimx, bdimx(TIDx), vec_factor]
  //         ^
  //         +--- next_innermost_pos

  if (resize_params->split_grid_x_dim) {
    ref_tv->split(next_innermost_pos, ResizeParams::max_gdimx);
    // [..., I0/bdimx/max_gdimx, max_gdimx, bdimx(TIDx), vec_factor]
  }
  ref_tv->axis(next_innermost_pos)->parallelize(ParallelType::BIDx);
  // [..., I0/bdimx/max_gdimx, max_gdimx(BIDx), bdimx(TIDx), vec_factor] or
  // [..., I0/bdimx(BIDx), bdimx(TIDx), vec_factor]

  // Propagate the reference to the other tensors. Note that the
  // update flag is enabled so to workaround the resize propagation
  // issue. This may not work if there's a tensor that is reshaped
  // from the reference tensor, but that should not be the case as the
  // reference is picked by the same routine used for the pointwise
  // scheduler.
  scheduler_tools::scheduleLoopDomainsLike(
      fusion->allTvs(),
      ref_tv->getLoopDomain(),
      /*update_loop_domain_only=*/true);

  if (vec_factor > 1) {
    const auto tvs_to_vectorize =
        scheduler_utils::getInputsOutputsWithInnerDim(ref_tv, true, true);
    for (auto tv_to_vectorize : tvs_to_vectorize) {
      if (tv_to_vectorize->isFusionInput()) {
        for (auto consumer_tv : ir_utils::consumerTvsOf(tv_to_vectorize)) {
          consumer_tv->axis(-1)->parallelize(ParallelType::Vectorize);
        }
      } else {
        tv_to_vectorize->axis(-1)->parallelize(ParallelType::Vectorize);
      }
    }
  }

  inlineMost();

  markAliases(fusion);
}

} // namespace nvfuser
