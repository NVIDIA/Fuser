// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <debug.h>
#include <instrumentation.h>
#include <ir/graphviz.h>
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

  auto resize_based_tensor_ops = ir_utils::getOpsOfType<SliceOp, PadOp>(fusion);

  // Slicing of or to a broadcast ID is not allowed yet.
  for (auto tensor_op : resize_based_tensor_ops) {
    TensorView* out_tv = tensor_op->output(0)->as<TensorView>();
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
        schedulerType(), "No referene found");
    return false;
  }

  // This doesn't work yet due to issue #3571
  if (getenv("BROADCAST_CHECK")) {
    if (std::any_of(
            ref_tv->getLogicalDomain().begin(),
            ref_tv->getLogicalDomain().end(),
            [](IterDomain* logical_id) { return logical_id->isBroadcast(); })) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(),
          "Reference with broadcast ID not supported yet: ",
          ref_tv->toString());
      return false;
    }
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

  // Disable the scheduler if there's a squeeze op. The loop option
  // may also need to be enabled in that case, but that option is not
  // turned on automatically yet.
#if 0
  if (ir_utils::hasOpsOfType<SqueezeOp>(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "SqueezeOp not supported.");
    return false;
  }
#endif

  if (hasAtLeastTwoValidGroups(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Transpose pattern not supported.");
    return false;
  }

  scheduler_tools::TransposeDomainMap domain_map(fusion);
  auto grouped_inputs_outputs = domain_map.groupInputsOutputsByInnerDim();
  if (grouped_inputs_outputs.size() >= 2) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Multiple transpose groups not supported.");
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

  int64_t max_num_elms = -1;
  for (auto out_tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    int64_t num_elms = 1;
    for (auto logical_id : out_tv->getLogicalDomain()) {
      auto inferred_val =
          runtime_info.expressionEvaluator().evaluate(logical_id->extent());
      NVF_ERROR(
          inferred_val.hasValue(),
          "Error inferring extent of: ",
          logical_id->toString());
      num_elms *= inferred_val.as<int64_t>();
    }
    std::cerr << "Output: " << out_tv->toString() << ", num elems: " << num_elms
              << "\n";
    max_num_elms = std::max(max_num_elms, num_elms);
  }

  std::cerr << "Max num elms: " << max_num_elms << "\n";

  if (ceilDiv(max_num_elms, bdimx) > ResizeParams::max_gdimx) {
    std::cerr << "Split gdimx\n";
    params->split_grid_x_dim = true;
  } else {
    params->split_grid_x_dim = false;
  }

  return params;
}

void ResizeScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  FUSER_PERF_SCOPE("ResizeScheduler::schedule");

  FusionGuard fg(fusion);
  const auto rparams = dynamic_cast<const ResizeParams*>(params);
  NVF_ERROR(rparams != nullptr);

  scheduler_utils::clearMemorySpace(fusion);

  {
    std::cout << std::endl;
    std::cout << "Resize scheduling\n";
    fusion->printMath();
    std::cout << std::endl;
  }

  {
    std::stringstream file_name;
    file_name << "pre_scheduling.dot";
    IrGraphGenerator::print(
        fusion,
        file_name.str().c_str(),
        IrGraphGenerator::DetailLevel::ComputeOnly);
  }

  auto cache_tvs = scheduler_utils::cacheInputs(fusion, true);
  for (const auto& cache_tv : cache_tvs) {
    std::cerr << "Inserted cache: " << cache_tv->definition()->toString();
  }

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
    std::cerr << "NonExclusive op: " << resize_tensor_op->toString();
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

  {
    std::cout << std::endl;
    std::cout << "After recomputation\n";
    fusion->printMath();
    std::cout << std::endl;

    std::stringstream file_name;
    file_name << "after_recomputation.dot";
    IrGraphGenerator::print(
        fusion,
        file_name.str().c_str(),
        IrGraphGenerator::DetailLevel::ComputeOnly);
  }

  for (auto expr : fusion->exprs()) {
    if (!expr->isOneOf<SliceOp, PadOp>()) {
      continue;
    }

    scheduler_tools::propagateResizeToInputs(expr);
  }

  // Mistral bwd has a reshape at the end of the fusion that merges
  // the two innermost dimensions. To make the below vectorization to
  // work, the vectorization needs to be applied to the innermost
  // dimension only, so the merge needs to be canceled.
  scheduler_tools::cancelReshapeTransforms(fusion);

  auto ref_tv = getReferenceTensor(fusion);

  // Just simple scheduling for now.
  // TODO: Do something smarter. Can just use the pointwise scheduler?

  // Make sure the DID ID located at the outermost position
  const auto outermost_pos = scheduler_utils::reorderDevicesToOuter(ref_tv);

  int64_t vec_factor = -1;
  if (getenv("VEC")) {
    vec_factor = atoi(getenv("VEC"));
  }

  // Skip vectorization of the first segment
  bool vectorize = vec_factor > 1 && (fusion->inputs().size() > 1);

  // Use this value if not -1
  int64_t gdimx = -1;
  if (getenv("GDIMX")) {
    gdimx = atoi(getenv("GDIMX"));
  }

  if (vectorize) {
    std::cerr << "Ref tensor: " << ref_tv->toString() << "\n";
    TensorView* allocation_ref_tv = nullptr;
    for (auto inp : ir_utils::filterByType<TensorView>(fusion->inputs())) {
      if (allocation_ref_tv == nullptr ||
          allocation_ref_tv->getLogicalDomain().size() <
              inp->getLogicalDomain().size()) {
        allocation_ref_tv = inp;
      }
    }
    std::cerr << "Allocation ref tensor: " << allocation_ref_tv->toString()
              << ", "
              << toDelimitedString(
                     allocation_ref_tv->getMaybeAllocationDomain())
              << "\n";

    auto ref_alloc = allocation_ref_tv->getMaybeAllocationDomain();
    std::vector<IterDomain*> ref_alloc_reordered;
    ref_alloc_reordered.reserve(ref_alloc.size());
    // Move broadcast outer
    for (const auto i : c10::irange(ref_alloc.size())) {
      if (ref_alloc.at(i)->isBroadcast()) {
        ref_alloc_reordered.push_back(ref_alloc.at(i));
      }
    }
    for (const auto i : c10::irange(ref_alloc.size())) {
      if (!ref_alloc.at(i)->isBroadcast()) {
        ref_alloc_reordered.push_back(ref_alloc.at(i));
      }
    }

    std::cerr << "Ref alloc before: "
              << toDelimitedString(
                     allocation_ref_tv->getMaybeAllocationDomain())
              << ", reordered: " << toDelimitedString(ref_alloc_reordered)
              << "\n";

    // Reorder the reference as the allocation domain of the fusion
    // inputs. For now, just use the first input
    // scheduler_utils::reorderTensorLike(ref_tv, ref_alloc_reordered);

    std::cerr << "Reordered ref: " << ref_tv->toString() << "\n";

    int64_t bdimx = 128;
    if (getenv("BDIMX")) {
      bdimx = atoi(getenv("BDIMX"));
    }

    ref_tv->split(-1, vec_factor);
    ref_tv->flatten(outermost_pos, -2);
    ref_tv->split(outermost_pos, bdimx);
    ref_tv->axis(-2)->parallelize(ParallelType::TIDx);
    if (gdimx > 0) {
      ref_tv->split(outermost_pos, gdimx);
    } else if (rparams->split_grid_x_dim) {
      ref_tv->split(outermost_pos, ResizeParams::max_gdimx);
    }
    ref_tv->axis(-3)->parallelize(ParallelType::BIDx);
  } else {
    // Schedule only the remaining IDs
    ref_tv->flatten(outermost_pos);
    ref_tv->split(outermost_pos, 128);
    ref_tv->axis(-1)->parallelize(ParallelType::TIDx);
    if (gdimx > 0) {
      ref_tv->split(outermost_pos, gdimx);
    } else if (rparams->split_grid_x_dim) {
      ref_tv->split(outermost_pos, ResizeParams::max_gdimx);
    }
    ref_tv->axis(-2)->parallelize(ParallelType::BIDx);
  }

  // Propagate the reference to the other tensors. Note that the
  // update flag is enabled so to workaround the resize propagation
  // issue. This may not work if there's a tensor that is reshaped
  // from the reference tensor, but that should not be the case as the
  // reference is picked by the same routine used for the pointwise
  // scheduler.
  scheduler_tools::scheduleLoopDomainsLike(
      fusion->allTvs(), ref_tv->getLoopDomain(), true);

  if (vectorize) {
    for (auto inp_tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
      if (inp_tv->name() == 2) {
        continue;
      }
      for (auto consumer_tv : ir_utils::consumerTvsOf(inp_tv)) {
        consumer_tv->axis(-1)->parallelize(ParallelType::Vectorize);
      }
    }
    // To avoid vectorizing the outputs of the first segment
    for (auto out_tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
      out_tv->axis(-1)->parallelize(ParallelType::Vectorize);
    }
  }

  inlineMost();

  markAliases(fusion);
}

} // namespace nvfuser
