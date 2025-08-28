// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <debug.h>
#include <device_lower/analysis/fusion_info.h>
#include <device_lower/analysis/sync_information.h>
#include <device_lower/utils.h>
#include <exceptions.h>
#include <id_model/id_model.h>
#include <id_model/to_string.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <options.h>
#include <scheduler/debug_utils.h>
#include <scheduler/greedy.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/loop_domain_scheduler.h>
#include <scheduler/tools/maxinfo_propagator.h>
#include <scheduler/utils.h>
#include <transform_replay.h>
#include <val_graph_visitor.h>

#include <ATen/cuda/CUDAContext.h>

#include <ranges>
#include <vector>

namespace nvfuser {

namespace {

// These are the current supported constrained ops.
std::vector<Expr*> getAllConstrainedOps(Fusion* fusion) {
  return ir_utils::getOpsOfType<ArgsortOp, ScanOp, PadOp>(fusion);
}

// Given offsets of logical IDs, return corresponding loop ID offsets
std::vector<int64_t> getDependentLoopIds(
    TensorView* tv,
    const std::vector<int64_t>& logical_id_offsets) {
  std::vector<Val*> logical_ids;
  logical_ids.reserve(logical_id_offsets.size());
  std::ranges::transform(
      logical_id_offsets,
      std::back_inserter(logical_ids),
      [tv](int64_t logical_id_offset) {
        return tv->getLogicalDomain().at(logical_id_offset);
      });

  const auto logical_loop_all_ids = DependencyCheck::getAllValsBetween(
      {logical_ids.begin(), logical_ids.end()},
      {tv->getLoopDomain().begin(), tv->getLoopDomain().end()});
  const std::unordered_set<Val*> logical_loop_all_id_set{
      logical_loop_all_ids.begin(), logical_loop_all_ids.end()};

  std::vector<int64_t> loop_id_offsets;
  for (const auto [i, loop_id] : enumerate(tv->getLoopDomain())) {
    if (logical_loop_all_id_set.contains(loop_id)) {
      loop_id_offsets.push_back(i);
    }
  }

  return loop_id_offsets;
}

class CompileTimeChecker : private IterVisitor {
 public:
  static bool run(Fusion* fusion, const ValGraph& exact_graph) {
    CompileTimeChecker checker(fusion, exact_graph);
    if (!checker.can_schedule_ && !checker.reject_reason_.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          SchedulerType::Greedy, checker.reject_reason_);
    }
    return checker.can_schedule_;
  }

 private:
  CompileTimeChecker(Fusion* fusion, const ValGraph& exact_graph)
      : exact_graph_(exact_graph) {
    checkConflictingReshape();
    if (!can_schedule_) {
      return;
    }

    traverse(fusion);
    if (!can_schedule_) {
      return;
    }

    // If this fusion requires the exact block dimension, requires the
    // constrained IDs to be exactly mapped. This is not necessary but
    // sufficient.
    if (needs_exact_block_dim_ && mismatched_constrained_id_detected_) {
      can_schedule_ = false;
      setRejectReason(
          "Block dimension must be exact but non-matching constrained IDs "
          "found");
    }

    // Make sure constrained and unconstrained ids are
    // disjoint. Because of the requirement that all ID groups must be
    // used uniquely (no multiple distinctive use Expr groups), it is
    // suffient to look at reachable graph nodes from each of the
    // groups by a forward traversal and see if there's any common
    // nodes. Because there's no ID group that has multiple uses, it
    // is not necessary to traverse backward.
    if (unique_unconstrained_domain_.has_value()) {
      auto reachable_vals_from_unconstrained_domain =
          getReachableValsFrom<ValGraphPermissiveBFS>(
              unique_unconstrained_domain_.value().vector(),
              exact_graph_.disjointValSets().disjointSets(),
              Direction::Forward,
              exact_graph_);
      auto common_reachable_ids = getReachableValsFrom<ValGraphPermissiveBFS>(
          all_constrained_domain_.vector(),
          reachable_vals_from_unconstrained_domain,
          Direction::Forward,
          exact_graph_);
      if (!common_reachable_ids.empty()) {
        can_schedule_ = false;
        std::stringstream reason;
        reason << "Constrained and unconstrained IDs are merged at: "
               << nvfuser::toString(common_reachable_ids);
        setRejectReason(reason.str());
      }
    }
  }

  void dispatch(Expr* expr) override {
    // These are the ops that are currently allowed to exist in the
    // given fusion. Notably, BroadcastOp, ReductionOp and ReshapeOp
    // are still missing.
    can_schedule_ = can_schedule_ &&
        expr->isOneOf<
            LoadStoreOp,
            UnaryOp,
            BinaryOp,
            TernaryOp,
            FullOp,
            ReshapeOp,
            ArgsortOp,
            ScanOp,
            PadOp>();
    if (!can_schedule_) {
      return;
    }
    IterVisitor::dispatch(expr);
  }

  void handle(ArgsortOp* argsort) override {
    // Due to the current limitations of ArgsortOp codegen, all of the
    // TIDx threads participate, which means the TID parallelized iter
    // domain of this ArgsortOp must have an extent that is no less
    // than any other TID parallelized iter domains. Requiring the
    // exactness is not necessary but sufficient.
    needs_exact_block_dim_ = true;

    auto out_tv = ir_utils::getTvOutput(argsort);
    checkConstrainedTv(out_tv, {argsort->dim()});
    if (!can_schedule_) {
      return;
    }

    // Only static dim supported for now. See also
    // CudaKernelGenerator::handle(ArgsortOp*)
    auto sorted_id = out_tv->getLogicalDomain().at(argsort->dim());
    if (!sorted_id->extent()->isConstInt()) {
      can_schedule_ = false;
      std::stringstream reason;
      reason << "Symbolic dimension not supported yet: " << argsort->toString();
      setRejectReason(reason.str());
      return;
    }
  }

  void handle(PadOp* pad) override {
    checkConstrainedTv(ir_utils::getTvOutput(pad), pad->getPaddedAxes());
  }

  void handle(ScanOp* scan) override {
    auto out_tv = ir_utils::getTvOutput(scan);
    checkConstrainedTv(out_tv, {scan->dim()});

    // Only static dim supported for now. See also
    // CudaKernelGenerator::handle(ScanOp*)
    auto scan_id = out_tv->getLogicalDomain().at(scan->dim());
    if (!scan_id->extent()->isConstInt()) {
      can_schedule_ = false;
      std::stringstream reason;
      reason << "Symbolic dimension not supported yet: " << scan->toString();
      setRejectReason(reason.str());
      return;
    }
  }

  // Check if the logical IDs of the given constrained tv can be
  // acceptable.
  void checkConstrainedTv(
      TensorView* tv,
      const std::vector<int64_t>& constrained_logical_id_offsets) {
    const auto& logical_domain = tv->getLogicalDomain();
    const std::unordered_set<int64_t> constrained_logical_id_offset_set(
        constrained_logical_id_offsets.begin(),
        constrained_logical_id_offsets.end());

    ValGroups constrained_domain;
    ValGroups unconstrained_domain;
    for (const auto& [i, logical_id] : enumerate(logical_domain)) {
      if (constrained_logical_id_offset_set.contains(i)) {
        const auto& logical_id_group = exact_graph_.toGroup(logical_id);
        constrained_domain.pushBack(logical_id_group);
        // Keep track of all constrained IDs as well for reshape analysis
        all_constrained_domain_.pushBack(logical_id_group);
      } else {
        unconstrained_domain.pushBack(exact_graph_.toGroup(logical_id));
      }
    }

    // All the unconstrained iter domains would be flattened and
    // parallelized with BIDx. The BIDx parallelized iter
    // domain must be mapped across the fusion to avoid the grid
    // synchronization. For the mapping, the exact graph is used for
    // now since BroadcastOp is not yet allowed.
    if (unique_unconstrained_domain_.has_value()) {
      if (unique_unconstrained_domain_->set() != unconstrained_domain.set()) {
        can_schedule_ = false;
        std::stringstream reason;
        reason << "Mismatched unconstrained IDs detected with "
               << tv->toString() << ": "
               << nvfuser::toString(unconstrained_domain)
               << ". Ref: " << nvfuser::toString(*unique_unconstrained_domain_);
        setRejectReason(reason.str());
        unique_unconstrained_domain_.reset();
      }
    } else {
      unique_unconstrained_domain_ = unconstrained_domain;
    }

    // All the constrained iter domains would be flattened and parallelized
    // with TIDx. Check if the flattened constrained iter domain would
    // be unique. Nothing to do if already not found to be unique.
    if (!mismatched_constrained_id_detected_) {
      if (unique_constrained_domain_.has_value()) {
        if (unique_constrained_domain_->set() != constrained_domain.set()) {
          mismatched_constrained_id_detected_ = true;
          unique_constrained_domain_.reset();
        }
      } else {
        unique_constrained_domain_ = constrained_domain;
      }
    }
  }

  // In order to ensure no conflicting reshape exists, fusions are
  // only allowed to have one use ExprGroup for each ID group. This
  // condition is not strictly necessary, but it makes the
  // can-schedule analysis fairly simple as seen below.
  void checkConflictingReshape() {
    for (const ValGroup& val_group :
         exact_graph_.disjointValSets().disjointSets()) {
      const auto& use_groups = exact_graph_.getUses(val_group);
      // Root-to-logical exprs may include Resize ops too, but they
      // can be ignored for this analysis since transformations are
      // simply propagated along Resize ops
      int num_reshape_exprs = 0;
      for (const auto& use_group : use_groups) {
        if (use_group->front()->isA<Merge>() ||
            use_group->front()->isA<Split>()) {
          ++num_reshape_exprs;
        }
      }
      if (num_reshape_exprs > 1) {
        can_schedule_ = false;
        std::stringstream ss;
        ss << "Potentially conflicting reshape found for "
           << nvfuser::toString(val_group);
        setRejectReason(ss.str());
        return;
      }
    }
  }

  void setRejectReason(const std::string& reason) {
    // Only keeps the first reason
    if (reject_reason_.empty()) {
      reject_reason_ = reason;
    }
  }

 private:
  const ValGraph& exact_graph_;

  bool can_schedule_ = true;
  std::string reject_reason_;

  std::optional<ValGroups> unique_unconstrained_domain_;
  std::optional<ValGroups> unique_constrained_domain_;

  ValGroups all_constrained_domain_;

  // True if mismatched constrained ID was detected
  bool mismatched_constrained_id_detected_ = false;
  // True if the block dimension must be exactly determined
  bool needs_exact_block_dim_ = false;
};

class RunTimeChecker : private IterVisitor {
 public:
  static bool run(Fusion* fusion, SchedulerRuntimeInfo& runtime_info) {
    RunTimeChecker checker(fusion, runtime_info);
    if (!checker.can_schedule_ && !checker.reject_reason_.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          SchedulerType::Greedy, checker.reject_reason_);
    }
    return checker.can_schedule_;
  }

 private:
  RunTimeChecker(Fusion* fusion, SchedulerRuntimeInfo& runtime_info)
      : runtime_info_(runtime_info),
        max_threads_per_block_(
            at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock) {
    traverse(fusion);
  }

  void dispatch(Expr* expr) override {
    if (!can_schedule_) {
      return;
    }
    IterVisitor::dispatch(expr);
  }

  void handle(ArgsortOp* argsort) override {
    checkConstrainedTv(ir_utils::getTvOutput(argsort), {argsort->dim()});
  }

  void handle(PadOp* pad) override {
    checkConstrainedTv(ir_utils::getTvOutput(pad), pad->getPaddedAxes());
  }

  void handle(ScanOp* scan) override {
    checkConstrainedTv(ir_utils::getTvOutput(scan), {scan->dim()});
  }

  // Since all constrained IDs are flattened and parallelized with
  // TIDx, the extent of flattened ID must not exceed the maximum
  // number of threads per thread block.
  void checkConstrainedTv(
      TensorView* tv,
      const std::vector<int64_t>& constrained_logical_id_offsets) {
    int64_t size_of_constrained_ids = 1;
    for (const auto i : constrained_logical_id_offsets) {
      auto logical_id = tv->getLogicalDomain().at(i);
      auto extent_val =
          runtime_info_.expressionEvaluator().evaluate(logical_id->extent());
      NVF_ERROR(
          extent_val.hasValue(),
          "Cannot infer the extent of a constrained logical ID: ",
          logical_id->toString());
      size_of_constrained_ids *= extent_val.as<int64_t>();
    }

    if (size_of_constrained_ids > max_threads_per_block_) {
      std::stringstream reason;
      reason << "Extent of constrained logical IDs, " << size_of_constrained_ids
             << ", exceeds the maxinum number of threads per thread block, "
             << max_threads_per_block_;
      setRejectReason(reason.str());
      can_schedule_ = false;
    }
  }

  void setRejectReason(const std::string& reason) {
    // Only keeps the first reason
    if (!reject_reason_.empty()) {
      return;
    }

    reject_reason_ = reason;
  }

 private:
  SchedulerRuntimeInfo& runtime_info_;
  int64_t max_threads_per_block_ = 0;

  bool can_schedule_ = true;
  std::string reject_reason_;
};

// Propagate all reshape transformations throughout the fusion.
void propagateReshape(Fusion* fusion) {
  const auto reshape_ops = ir_utils::getOpsOfType<ReshapeOp>(fusion);
  const auto all_tvs = fusion->allTvs();

  // Visit al reshape ops in a topological order. Propagate the merge
  // and split ops to all tensors as long as they have matching input
  // IDs. Propagation should work consistently as all reshapes are
  // guaranteed to have no conflicting transformations. A single ID
  // group may get propagated multiple times if there are multiple
  // reshapes, but they are guaranteed to have the same
  // transformations.
  for (auto reshape : reshape_ops) {
    auto reshape_exprs = DependencyCheck::getAllExprsBetween(
        {reshape->out()->getRootDomain().begin(),
         reshape->out()->getRootDomain().end()},
        {reshape->out()->getLogicalDomain().begin(),
         reshape->out()->getLogicalDomain().end()});
    scheduler_tools::scheduleLoopDomainsBy(
        all_tvs, reshape_exprs, Direction::Forward);
  }
}

class ConstrainedOpScheduler : public OptOutDispatch {
 public:
  static void run(
      Fusion* fusion,
      const std::vector<TensorView*>& constrained_out_tvs,
      const ValGraph& exact_graph) {
    ConstrainedOpScheduler scheduler(fusion, constrained_out_tvs, exact_graph);
  }

 private:
  ConstrainedOpScheduler(
      Fusion* fusion,
      const std::vector<TensorView*>& constrained_out_tvs,
      const ValGraph& exact_graph)
      : exact_graph_(exact_graph) {
    for (auto constrained_tv : constrained_out_tvs) {
      dispatch(constrained_tv->definition());
    }
  }

  void handle(ArgsortOp* argsort) override {
    auto out_tv = ir_utils::getTvOutput(argsort);
    auto dim = argsort->dim();
    scheduleConstrainedTv(out_tv, {dim});
  }

  void handle(PadOp* pad) override {
    scheduleConstrainedTv(ir_utils::getTvOutput(pad), pad->getPaddedAxes());
  }

  void handle(ScanOp* scan) override {
    auto scan_dim = scan->dim();
    auto out_tv = ir_utils::getTvOutput(scan);
    scheduleConstrainedTv(out_tv, {scan_dim});
  }

  void scheduleConstrainedTv(
      TensorView* tv,
      const std::vector<int64_t>& constrained_logical_id_offsets) {
    NVF_ERROR(!constrained_logical_id_offsets.empty());

    const auto& constrained_loop_id_offsets =
        getDependentLoopIds(tv, constrained_logical_id_offsets);

    // Move the constrained_logical_ids innermost
    std::unordered_map<int64_t, int64_t> old2new;
    for (const auto [i, offset] : enumerate(constrained_loop_id_offsets)) {
      old2new.emplace(offset, i - std::ssize(constrained_loop_id_offsets));
    }
    tv->reorder(old2new);

    // Flatten the constrained ids
    if (constrained_loop_id_offsets.size() > 1) {
      tv->flatten(-std::ssize(constrained_loop_id_offsets), -1);
    }

    // Parallelize the flattened constrained id
    tv->axis(-1)->parallelize(ParallelType::TIDx);

    // All done if there's no unconstrained ID
    if (tv->getLoopDomain().size() == 1) {
      return;
    }

    // Scheduling of the unconstrained IDs with BIDx. Currently all
    // tensors are assumed to have exact-mapped IDs for BID in order to
    // avoid grid sync. Reordering is allowed, though TransposeOp is
    // not yet enabled.

    // Accommodate reordered unconstrained IDs
    if (ref_unconstrained_domain_.empty()) {
      ref_unconstrained_domain_ =
          exact_graph_.toGroups(std::vector<IterDomain*>{
              tv->getLoopDomain().begin(), tv->getLoopDomain().end() - 1});
    } else {
      std::vector<int64_t> permutation;
      permutation.reserve(ref_unconstrained_domain_.size());
      for (const auto i : arange(tv->getLoopDomain().size() - 1)) {
        auto id = tv->getLoopDomain().at(i);
        auto ref_it = std::ranges::find_if(
            ref_unconstrained_domain_,
            [&](const ValGroup& id_group) { return id_group->has(id); });
        NVF_ERROR(
            ref_it != ref_unconstrained_domain_.end(),
            "Failed find matching ID group: ",
            id->toString());
        permutation.push_back(
            std::distance(ref_unconstrained_domain_.begin(), ref_it));
      }
      tv->reorder(permutation);
    }

    tv->flatten(0, std::ssize(tv->getLoopDomain()) - 2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }

 private:
  const ValGraph& exact_graph_;
  ValGroups ref_unconstrained_domain_;
};

// Partition all tensors in a given fusion to disjoint sets using
// constrained tensors as references. Returns a map
// from each tensor to its assigned reference.
//
// The partitioning proceeds bottom-up, traversing from
// constrained tensors to all other tensors. When a tensor is
// reached that hasn't been grouped yet, it is assigned into the
// reference's subset. If the tensor is already part of a group, its
// original assignment remains unchanged.
//
// The traversal occurs both backward and forward directions, with a
// preference for backward. Currently, this doesn't make any
// difference since reshape is not allowed. However, backward schedule
// propagation can trivially work across reshapes, whereas forward
// propagation requires a reshape to be cancelled.
std::unordered_map<TensorView*, TensorView*> partitionFusion(
    Fusion* fusion,
    const std::unordered_set<TensorView*>& constrained_tvs) {
  FusionGuard fg(fusion);

  const auto all_exprs = fusion->exprs();
  const auto all_tvs = fusion->allTvs();

  std::unordered_map<TensorView*, TensorView*> tv_to_constrained_tv_map;

  // Register self mappings for constrained tensors
  for (auto tv : constrained_tvs) {
    tv_to_constrained_tv_map.emplace(tv, tv);
  }

  // Propagate source reference through a given expr. Returns true if
  // propagation is indeed done.
  auto propagateThroughExpr = [&](Expr* expr, Direction dir) -> bool {
    if (!ir_utils::isTvOp(expr)) {
      return false;
    }

    // Find a reference to propagate. If dir is Forward, the reference
    // of the first producer tensor with a reference is used as the
    // reference of this expr. Similarly, if dir is Backward, the reference
    // of the first consumer tensor with a reference is used.
    //
    // When multiple producers or consumers have different
    // references, the reference of the first producer or consumer is
    // propagated.
    TensorView* ref_to_propagate = nullptr;
    const auto& src_vals =
        dir == Direction::Forward ? expr->inputs() : expr->outputs();
    const auto& dst_vals =
        dir == Direction::Forward ? expr->outputs() : expr->inputs();

    auto src_with_ref_it = std::ranges::find_if(src_vals, [&](Val* src) {
      return src->isA<TensorView>() &&
          tv_to_constrained_tv_map.contains(src->as<TensorView>());
    });
    if (src_with_ref_it != src_vals.end()) {
      ref_to_propagate =
          tv_to_constrained_tv_map.at((*src_with_ref_it)->as<TensorView>());
    }

    // No reference to propagate is found
    if (ref_to_propagate == nullptr) {
      return false;
    }

    bool updated = false;

    for (auto dst_tv : ir_utils::filterByType<TensorView>(dst_vals)) {
      // If already set, don't overwrite. If not, propagate the output reference
      // if found.
      if (tv_to_constrained_tv_map.contains(dst_tv)) {
        continue;
      } else {
        NVF_ERROR(
            tv_to_constrained_tv_map.emplace(dst_tv, ref_to_propagate).second,
            "Trying to propagate reference multiple times to: ",
            dst_tv->toString());
        updated = true;
      }
    }

    return updated;
  };

  // Backward propagation across the fusion. Repeat until all
  // expressions are visited.
  auto propagate_backward = [&]() -> bool {
    bool updated = false;
    for (auto expr : all_exprs | std::views::reverse) {
      if (tv_to_constrained_tv_map.size() == all_tvs.size()) {
        return updated;
      }
      if (propagateThroughExpr(expr, Direction::Backward)) {
        updated = true;
      }
    }
    return updated;
  };

  // Forward propagation across the fusion. Unlike the backward prop,
  // immediately terminate once a propagation is done. This is for
  // prioritizing backward propagation.
  auto propagate_forward = [&]() -> bool {
    if (tv_to_constrained_tv_map.size() == all_tvs.size()) {
      return false;
    }
    for (auto expr : all_exprs) {
      if (propagateThroughExpr(expr, Direction::Forward)) {
        return true;
      }
    }
    return false;
  };

  while (tv_to_constrained_tv_map.size() != all_tvs.size()) {
    // Prioritize backprop
    if (propagate_backward()) {
      continue;
    }

    if (propagate_forward()) {
      continue;
    }

    // No progress made
    break;
  }

  if (tv_to_constrained_tv_map.size() != all_tvs.size()) {
    std::vector<TensorView*> ungrouped_tvs;
    std::ranges::copy_if(
        all_tvs, std::back_inserter(ungrouped_tvs), [&](auto tv) {
          return !tv_to_constrained_tv_map.contains(tv);
        });
    NVF_THROW(
        "Fail to group the following tensors: ",
        toDelimitedString(ungrouped_tvs));
  }

  return tv_to_constrained_tv_map;
}

SyncMap buildSyncMap(Fusion* fusion) {
  FusionInfo fusion_info;
  FusionInfoGuard info_guard(&fusion_info);
  fusion_info.set(std::make_unique<ConcretizedBroadcastDomains>(fusion));
  fusion_info.set(std::make_unique<PaddedParallelDimensions>(
      collectPaddedParallelDims(fusion)));
  fusion_info.set(std::make_unique<IdModel>(fusion, /*build_graphs=*/true));
  fusion_info.set(std::make_unique<ComputeAtMap>(fusion));
  fusion_info.set(std::make_unique<ParallelDimensionMap>(fusion));
  fusion_info.set(std::make_unique<ThreadPredicateMap>(fusion));
  return SyncMap(fusion, /*error_on_failure=*/false);
}

} // namespace

bool GreedyScheduler::canScheduleCompileTime(Fusion* fusion) {
  if (!isOptionEnabled(EnableOption::GreedyScheduler)) {
    scheduler_debug_utils::canScheduleRejectReason(
        SchedulerType::Greedy, "Not enabled");
    return false;
  }

  auto constrained_ops = getAllConstrainedOps(fusion);
  if (constrained_ops.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        SchedulerType::Greedy, "No constrained op found");
    return false;
  }

  IdModel id_model(fusion);
  const auto& exact_graph = id_model.buildExactGraph();

  return CompileTimeChecker::run(fusion, exact_graph);
}

bool GreedyScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  return RunTimeChecker::run(fusion, runtime_info);
}

std::unique_ptr<HeuristicParams> GreedyScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("GreedyScheduler::computeHeuristics");

  auto params = std::make_unique<HeuristicParams>(SchedulerType::Greedy);
  params->tag = "Greedy heuristics";
  params->cparams.index_type = runtime_info.getIndexType();

  return params;
}

void GreedyScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  FUSER_PERF_SCOPE("GreedyScheduler::schedule");

  scheduler_utils::clearMemorySpace(fusion);

  propagateReshape(fusion);

  auto constrained_exprs = getAllConstrainedOps(fusion);

  std::vector<TensorView*> constrained_tvs;
  constrained_tvs.reserve(constrained_exprs.size());
  std::ranges::transform(
      constrained_exprs,
      std::back_inserter(constrained_tvs),
      [](const Expr* expr) { return ir_utils::getTvOutput(expr); });

  IdModel id_model(fusion);
  const auto& exact_graph = id_model.buildExactGraph();

  // Schedule all constrained tensors
  ConstrainedOpScheduler::run(fusion, constrained_tvs, exact_graph);

  // Need to fetch constrained ops again as cacheAfter/Before may be used
  // TODO: Cleanup.
  constrained_exprs = getAllConstrainedOps(fusion);
  constrained_tvs.clear();
  std::ranges::transform(
      constrained_exprs,
      std::back_inserter(constrained_tvs),
      [](const Expr* expr) { return ir_utils::getTvOutput(expr); });

  // Partition the fusion
  auto tv_to_ref_map =
      partitionFusion(fusion, {constrained_tvs.begin(), constrained_tvs.end()});

  // Propagate the schedule of each constrained tv to its disjoint set
  for (auto constrained_tv : constrained_tvs) {
    std::unordered_set<TensorView*> tvs_to_transform;
    for (const auto& [tv, ref] : tv_to_ref_map) {
      if (ref == constrained_tv) {
        tvs_to_transform.insert(tv);
      }
    }

    SetSelector selector(tvs_to_transform);
    MaxLogicalDomainInfoSpanningTree tree(constrained_tv, &selector);
    TransformPropagator tp(constrained_tv);
    tree.traverse(&tp);

    scheduler_utils::parallelizeAllLike(
        constrained_tv, -1, {tvs_to_transform.begin(), tvs_to_transform.end()});
  }

  // Resolve conflicts. Find conflicting producer-consumer pairs
  // and insert memory promotion

  VectorOfUniqueEntries<TensorView*> tvs_to_upload_to_smem;
  const auto sync_map = buildSyncMap(fusion);

  for (const auto& [tv, pt_map] : sync_map.map()) {
    NVF_ERROR(
        !pt_map.hasBID(),
        "Grid sync not expected: ",
        tv->toString(),
        ", ",
        pt_map.toString());
    tvs_to_upload_to_smem.pushBack(tv);
  }

  for (const auto& tv : tvs_to_upload_to_smem) {
    // Create a copy of this tensor on shared memory
    auto no_reduction_logical_domain =
        TensorDomain::noReductions(tv->getLogicalDomain());
    std::vector<IterDomain*> new_logical_domain;
    new_logical_domain.reserve(no_reduction_logical_domain.size());
    for (const auto& dom : no_reduction_logical_domain) {
      new_logical_domain.push_back(dom->cloneWithoutRFactor());
    }
    auto copy = IrBuilder::create<TensorView>(
        IrBuilder::create<TensorDomain>(
            new_logical_domain,
            TensorDomain::getContiguityFilledWith(new_logical_domain, true)),
        tv->dtype());
    TransformReplay::selfReplay(tv->domain(), copy->domain());
    copy->setMemoryType(MemoryType::Shared);

    // Insert a copy op
    IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, copy, tv);

    // Replace use of this tv if it has a conflicting consumer
    const auto& tv_sync_map = sync_map.producerConsumerRawSync().at(tv);
    std::vector<Expr*> uses_to_update;
    std::ranges::copy_if(
        tv->uses(), std::back_inserter(uses_to_update), [&](Expr* use) {
          return std::ranges::any_of(use->outputs(), [&](Val* out) {
            TensorView* out_tv = dynamic_cast<TensorView*>(out);
            if (out_tv == nullptr) {
              return false;
            }
            auto it = tv_sync_map.find(out_tv);
            return it != tv_sync_map.end() && it->second.hasTID();
          });
        });
    NVF_ERROR(!uses_to_update.empty());

    for (auto tv_use : uses_to_update) {
      ir_utils::replaceValInExprInputs(tv_use, tv, copy);
    }
  }
}

} // namespace nvfuser
