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
#include <scheduler/tools/static_repeat.h>
#include <val_graph_visitor.h>

#include <memory>

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
  if (isOptionDisabled(DisableOption::ResizeScheduler)) {
    scheduler_debug_utils::canScheduleRejectReason(schedulerType(), "Disabled");
    return false;
  }

  for (auto tv : fusion->allTvs()) {
    if (tv->dtype() != DataType::Index &&
        dataTypeSizeBit(tv->dtype()) % 8 != 0) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "Does not support sub-byte data types.");
      return false;
    }
  }

  if (!scheduler_tools::hasResizeBasedOps(fusion)) {
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

  auto resize_tensor_ops = scheduler_tools::getResizeBasedOps(fusion);

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

  auto ref_tv_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReferenceTensors>(
          data_cache, [fusion]() {
            std::vector<TensorView*> data{getReferenceTensor(fusion)};
            return std::make_unique<std::vector<TensorView*>>(std::move(data));
          });
  TensorView* ref_tv = ref_tv_entry.get()[0];

  // Only consider the innermost dimension to vectorize for now.
  // TODO: Consider vectorizing merged IDs, not just the innermost
  params->vectorization_factor = vectorize_helper::getVectorizationFactor(
      runtime_info,
      ref_tv,
      data_cache,
      (int64_t)ref_tv->getLogicalDomain().size() - 1);

  return params;
}

namespace {

// Before propagating loop transformations of the ref tensor, all
// tensors need their loop domains to be reachable from the ref by
// just a backward traversal in the exact graph. This is to make
// tranformations to be consistently propagated through resize-induced
// cycles in the exact graph.
//
// There may be tensors that have further transformations that do not
// exist in the reference tensor, e.g., a reshape of the
// reference. Here, we transform those tensors so that they become
// ready to get the reference transformations propagated to.
void prepareForBackwardTransformPropagation(TensorView* ref_tv) {
  Fusion* fusion = ref_tv->fusion();

  // Tensors that appear before the ref should not need anything here
  // as they should already be ready to get propagated from the ref
  const auto all_dep_stmts = StmtSort::getStmtsTo({ref_tv});
  const std::unordered_set<Statement*> all_dep_stmt_set{
      all_dep_stmts.begin(), all_dep_stmts.end()};

  if (std::ranges::all_of(fusion->allTvs(), [&](TensorView* tv) {
        return all_dep_stmt_set.contains(tv);
      })) {
    return;
  }

  IdModel id_model(fusion);
  const auto& graph = id_model.buildBroadcastGraph();

  const auto ref_logical = graph.toGroups(ref_tv->getLogicalDomain());

  std::vector<TensorView*> tvs_with_extra_transforms;

  // Look for tensor whose logical domain is not rechable from ref by
  // just a backward traversal
  for (auto tv : fusion->allTvs()) {
    if (all_dep_stmt_set.contains(tv)) {
      continue;
    }

    const auto tv_logical =
        graph.toGroups(TensorDomain::noBroadcasts(tv->getLogicalDomain()));

    auto all_visited = ValGraphBFS::getExprGroupsBetween(
                           graph,
                           ref_logical,
                           tv_logical,
                           /*require_all_to_visited=*/false,
                           Direction::Backward)
                           .second;

    if (all_visited) {
      continue;
    }

    // This tensor is unreachable by just traversing backward
    tvs_with_extra_transforms.push_back(tv);
  }

  if (tvs_with_extra_transforms.empty()) {
    return;
  }

  // Make their loop domains look like the ref so that loop
  // transformations of the ref can be propagated
  scheduler_tools::scheduleLoopDomainsLike(
      tvs_with_extra_transforms, ref_tv->getLogicalDomain());
}

// Partition a given set of tensors to two disjoint sets based on a
// given iter domain and reachability from the iter domain. Returns two
// vectors of tensors, first of which contains all tensors that has an
// iter domain that is reachable from the given iter domain, whereas
// the rest of tensors are all grouped into the second
// list. Reachability is determined by using the permissive BFS
// traversal on a given graph.
std::pair<std::vector<TensorView*>, std::vector<TensorView*>> partitionTvsById(
    const std::vector<TensorView*> tvs,
    IterDomain* id,
    const ValGraph& graph) {
  ValGroups target_groups;
  for (auto tv : tvs) {
    target_groups.pushBack(graph.toGroups(tv->getLogicalDomain()));
  }

  const auto reachable_groups = getReachableValsFrom<ValGraphPermissiveBFS>(
      {graph.toGroup(id)},
      target_groups.vector(),
      /*allowed_direction=*/Direction::Undefined,
      graph);
  const std::unordered_set<ValGroup> reachable_group_set{
      reachable_groups.begin(), reachable_groups.end()};

  std::vector<TensorView*> reachable_tvs;
  std::vector<TensorView*> unreachable_tvs;

  for (auto tv : tvs) {
    if (std::ranges::any_of(
            tv->getLogicalDomain(), [&](IterDomain* logical_id) {
              return reachable_group_set.contains(graph.toGroup(logical_id));
            })) {
      reachable_tvs.push_back(tv);
    } else {
      unreachable_tvs.push_back(tv);
    }
  }

  return std::make_pair(reachable_tvs, unreachable_tvs);
}

} // namespace

void ResizeScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  FUSER_PERF_SCOPE("ResizeScheduler::schedule");

  FusionGuard fg(fusion);
  const auto resize_params = dynamic_cast<const ResizeParams*>(params);
  NVF_ERROR(resize_params != nullptr);

  scheduler_utils::clearMemorySpace(fusion);

  scheduler_utils::cacheInputs(fusion, true);
  scheduler_utils::cacheAndForkOutputs(fusion, true);

  auto ref_tv = getReferenceTensor(fusion);
  NVF_ERROR(ref_tv != nullptr);

  auto resize_tensor_ops = ir_utils::getOpsOfType<SliceOp, PadOp>(fusion);

  std::unique_ptr<IdModel> id_model =
      std::make_unique<IdModel>(fusion, /*build_graphs=*/false);
  id_model->buildExactGraph();

  // Replicate resize inputs if necessary to avoid conflicting
  // propagations
  const auto exclusivity_info_map = scheduler_tools::getNonExclusiveResizeInfo(
      resize_tensor_ops, id_model->idGraph(IdMappingMode::EXACT));
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
    scheduler_tools::cancelReshapeInLoopDomains(
        largest_input, /*skip_innermost_id=*/true);
  }

  // Propagate Resize ops to producer tensors. This is safe as this
  // scheduler only accepts pointwise ops with no reduction.
  for (auto expr : fusion->exprs()) {
    if (!scheduler_tools::isResizeBasedOp(expr)) {
      continue;
    }
    scheduler_tools::propagateResizeToInputs(expr);
  }

  // Update the IdModel
  id_model = std::make_unique<IdModel>(fusion, /*build_graphs=*/false);
  id_model->buildExactGraph();

  // Detect an ending repeat
  auto repeat_info = scheduler_tools::getMaybeStaticRepeatInfo(ref_tv);

  // Just simple scheduling for now.
  // TODO: Do something smarter. Can just use the pointwise scheduler?

  // Reorder tensors to align with the largest input. This is expected
  // to improve the memory read performance, while the write
  // performance could be lowered. This should generally be more
  // important to optimize the read performance, but more robust
  // decision would be needed.
  if (largest_input != nullptr && ref_tv->getLoopDomain().size() > 1) {
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
    // input. In order to avoid reordering the innermost ID, only
    // consider the rest of IDs. This matters when the innermost ID of
    // the input is not mapped with the reference innermost ID. see
    // ResizeTest.ReorderLikeInputShouldNotMoveInnermostID for a
    // concrete example.
    auto permutation = scheduler_utils::reorderDomainLike(
        {ref_tv->getLoopDomain().begin(),
         ref_tv->getLoopDomain().begin() + ref_tv->getLoopDomain().size() - 1},
        ref_alloc);
    // Keep the innermost ID as is
    permutation.push_back(ref_tv->getLoopDomain().size() - 1);
    ref_tv->reorder(permutation);
  }

  const int64_t bdimx = 128;

  // Make sure the DID ID located at the outermost position
  auto outermost_pos = scheduler_utils::reorderDevicesToOuter(ref_tv);

  // [DID, ..., ...]
  //        ^
  //        +--- outermost_pos

  // Move the static repeat ID to the outermost position if
  // detected. The repeat ID then just remains there with no
  // scheduling.
  bool repeat_id_moved_to_outermost = false;
  if (repeat_info.has_value()) {
    auto ref_factor_id_it = std::find_if(
        ref_tv->getLoopDomain().begin(),
        ref_tv->getLoopDomain().end(),
        [&](IterDomain* loop_id) {
          return id_model->idGraph(IdMappingMode::EXACT)
              .disjointValSets()
              .strictAreMapped(loop_id, repeat_info->factor_id);
        });
    // The factor ID should be found in the loop domain as the
    // reshape should be cancelled at this point. Gives up if not.
    if (ref_factor_id_it != ref_tv->getLoopDomain().end()) {
      auto factor_id_pos =
          std::distance(ref_tv->getLoopDomain().begin(), ref_factor_id_it);
      NVF_ERROR(
          factor_id_pos >= outermost_pos,
          "Unexpected to have DID-parallelized repeat factor axis: ",
          repeat_info->factor_id->toString());

      // [DID, ..., repeat_factor_id, ...]
      //        ^
      //        +--- outermost_pos
      ref_tv->reorder(std::unordered_map<int64_t, int64_t>{{factor_id_pos, 0}});
      ++outermost_pos;
      // [repeat_factor_id, DID, ...]
      //                         ^
      //                         +--- outermost_pos
      repeat_id_moved_to_outermost = true;
    }
  }

  const int64_t vec_factor = resize_params->vectorization_factor;

  int64_t next_innermost_pos = -1;
  // [..., ...]
  //        ^
  //        +--- next_innermost_pos

  if (vec_factor > 1) {
    NVF_ERROR(
        ref_tv->axis(-1) == ref_tv->getLogicalDomain().back(),
        "Vectorization is assumed to be done with the innermost logical ID at "
        "this moment, which is expected to remain at the innermost position in "
        "the loop domain. Logical domain: ",
        toDelimitedString(ref_tv->getLogicalDomain()),
        ". Current loop domain: ",
        toDelimitedString(ref_tv->getLoopDomain()));
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

  // Before propagating the reference transformations, make sure all
  // tensors are ready to get transformations from the ref
  prepareForBackwardTransformPropagation(ref_tv);

  // Propagate the reference to the other tensors. Note that the
  // update flag is enabled to workaround the resize propagation
  // issue.
  //
  // When an ending static repeat is detected and the repeat ID is
  // moved to the outermost position, propagation is done separately
  // between the tensors before the repeat and after the repeat. The
  // tensors are first grouped into the pre-repeat group and the
  // post-repeat group, where only the latter group has the repeat
  // IDs. When propagating the loop domain of the reference tensor,
  // which has the repeat ID, the full loop domain is propagated only
  // to the tensors that have IDs that are mapped with the repeat
  // ID. For the rest of the tensros, the repeat ID is dropped and
  // only the remaining loop domain is propagated.
  if (repeat_id_moved_to_outermost) {
    const auto& [tvs_with_repeat_id, tvs_without_repeat_id] = partitionTvsById(
        fusion->allTvs(),
        repeat_info->factor_id,
        id_model->maybeBuildGraph(IdMappingMode::BROADCAST));

    // The repeat ID should be located at the outermost position
    std::vector<IterDomain*> non_repeated_loop{
        ref_tv->getLoopDomain().begin() + 1, ref_tv->getLoopDomain().end()};

    scheduler_tools::scheduleLoopDomainsLike(
        tvs_without_repeat_id,
        non_repeated_loop,
        /*update_loop_domain_only=*/true);
    scheduler_tools::scheduleLoopDomainsLike(
        tvs_with_repeat_id,
        ref_tv->getLoopDomain(),
        /*update_loop_domain_only=*/true);
  } else {
    scheduler_tools::scheduleLoopDomainsLike(
        fusion->allTvs(),
        ref_tv->getLoopDomain(),
        /*update_loop_domain_only=*/true);
  }

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
