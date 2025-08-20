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
#include <scheduler/tools/maxinfo_propagator.h>
#include <scheduler/utils.h>
#include <transform_replay.h>

#include <ATen/cuda/CUDAContext.h>

#include <ranges>
#include <vector>

namespace nvfuser {

namespace {

std::vector<Expr*> getAllConstrainedOps(Fusion* fusion) {
  return ir_utils::getOpsOfType<ArgsortOp, ScanOp, PadOp>(fusion);
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
    traverse(fusion);
  }

  void dispatch(Expr* expr) override {
    can_schedule_ = can_schedule_ &&
        expr->isOneOf<
            LoadStoreOp,
            UnaryOp,
            BinaryOp,
            TernaryOp,
            ScanOp,
            PadOp>();
    if (!can_schedule_) {
      return;
    }
    IterVisitor::dispatch(expr);
  }

  void handle(PadOp* pad) override {
    checkConstrainedTv(ir_utils::getTvOutput(pad), pad->getPaddedAxes());
  }

  void handle(ScanOp* scan) override {
    checkConstrainedTv(ir_utils::getTvOutput(scan), {scan->dim()});
  }

  void checkConstrainedTv(
      TensorView* tv,
      const std::vector<int64_t>& constrained_logical_id_offsets) {
    const auto& logical_domain = tv->getLogicalDomain();
    const std::unordered_set<int64_t> constrained_logical_id_offset_set(
        constrained_logical_id_offsets.begin(),
        constrained_logical_id_offsets.end());

    ValGroups non_constrained_ids;
    for (const auto& [i, logical_id] : enumerate(logical_domain)) {
      if (constrained_logical_id_offset_set.contains(i)) {
        continue;
      }

      non_constrained_ids.pushBack(exact_graph_.toGroup(logical_id));
    }

    if (ref_block_domain_.has_value()) {
      if (ref_block_domain_->set() != non_constrained_ids.set()) {
        can_schedule_ = false;
        std::stringstream reason;
        reason << "Mismatched unconstrained IDs detected with "
               << tv->toString() << ": "
               << nvfuser::toString(non_constrained_ids)
               << ". Ref: " << nvfuser::toString(*ref_block_domain_);
        setRejectReason(reason.str());
      }
    } else {
      ref_block_domain_ = non_constrained_ids;
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
  const ValGraph& exact_graph_;

  bool can_schedule_ = true;
  std::string reject_reason_;

  std::optional<ValGroups> ref_block_domain_;
};

class RunTimeChecker : private IterVisitor {
 public:
  static bool run(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache) {
    RunTimeChecker checker(fusion, runtime_info, data_cache);
    if (!checker.can_schedule_ && !checker.reject_reason_.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          SchedulerType::Greedy, checker.reject_reason_);
    }
    return checker.can_schedule_;
  }

 private:
  RunTimeChecker(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache)
      : runtime_info_(runtime_info),
        data_cache_(data_cache),
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

  void handle(PadOp* pad) override {
    checkConstrainedTv(ir_utils::getTvOutput(pad), pad->getPaddedAxes());
  }

  void handle(ScanOp* scan) override {
    checkConstrainedTv(ir_utils::getTvOutput(scan), {scan->dim()});
  }

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

    std::cerr << "Size of constrained IDs for " << tv->toString() << ": "
              << size_of_constrained_ids << "\n";

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
  HeuristicDataCache* data_cache_ = nullptr;
  int64_t max_threads_per_block_ = 0;

  bool can_schedule_ = true;
  std::string reject_reason_;

  std::optional<ValGroups> ref_block_domain_;
};

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

  void handle(PadOp* pad) override {
    std::cerr << "Scheduling " << pad->toString();

    auto out_tv = ir_utils::getTvOutput(pad);

    scheduleConstrainedTv(out_tv, pad->getPaddedAxes());
  }

  void handle(ScanOp* scan) override {
    std::cerr << "Scheduling " << scan->toString();

    auto scan_dim = scan->dim();

    auto in_tv = ir_utils::getTvInput(scan);
    auto out_tv = ir_utils::getTvOutput(scan);

    // Currently, scan input must be a register tensor
    if (in_tv->getMemoryType() != MemoryType::Local) {
      in_tv->cacheAfter();
    }

    // Currently, scan output must be a register tensor
    if (out_tv->getMemoryType() != MemoryType::Local) {
      out_tv = out_tv->cacheBefore();
    }

    scheduleConstrainedTv(out_tv, {scan_dim});
  }

  void scheduleConstrainedTv(
      TensorView* tv,
      const std::vector<int64_t>& constrained_logical_id_offsets) {
    std::cerr << "Scheduling: " << tv->toString() << "\n";
    NVF_ERROR_EQ(
        tv->getLogicalDomain(),
        tv->getLoopDomain(),
        "Logical and loop domains are assumed to be the same: ",
        tv->toString());

    NVF_ERROR(!constrained_logical_id_offsets.empty());

    // Move the constrained_logical_ids innermost
    std::unordered_map<int64_t, int64_t> old2new;
    for (const auto [i, offset] : enumerate(constrained_logical_id_offsets)) {
      old2new.emplace(offset, i - std::ssize(constrained_logical_id_offsets));
    }
    tv->reorder(old2new);
    std::cerr << "Reordered: " << tv->toString() << "\n";

    // Flatten the constrained ids
    if (constrained_logical_id_offsets.size() > 1) {
      tv->flatten(-std::ssize(constrained_logical_id_offsets), -1);
    }
    std::cerr << "Flattened: " << tv->toString() << "\n";

    // Parallelize the flattened constrained id
    tv->axis(-1)->parallelize(ParallelType::TIDx);

    if (tv->getLoopDomain().size() == 1) {
      return;
    }

    // Scheduling of the unconstrained IDs with BIDx. Currently all
    // tensors are assumed to have exact-mapped IDs for BID in order to
    // avoid grid sync.
    if (ref_block_ids_.empty()) {
      ref_block_ids_ = exact_graph_.toGroups(std::vector<IterDomain*>{
          tv->getLoopDomain().begin(), tv->getLoopDomain().end() - 1});
    } else {
      std::vector<int64_t> permutation;
      permutation.reserve(ref_block_ids_.size());
      for (const auto i : arange(tv->getLoopDomain().size() - 1)) {
        auto id = tv->getLoopDomain().at(i);
        auto ref_it = std::ranges::find_if(
            ref_block_ids_,
            [&](const ValGroup& id_group) { return id_group->has(id); });
        NVF_ERROR(
            ref_it != ref_block_ids_.end(),
            "Failed find matching ID group: ",
            id->toString());
        permutation.push_back(std::distance(ref_block_ids_.begin(), ref_it));
      }

      tv->reorder(permutation);
      std::cerr << "Reordered: " << tv->toString() << "\n";
    }

    // Merge the remaining IDs
    tv->flatten(0, std::ssize(tv->getLoopDomain()) - 2);
    tv->axis(0)->parallelize(ParallelType::BIDx);

    std::cerr << "Scheduled: " << tv->toString() << "\n";
  }

 private:
  const ValGraph& exact_graph_;
  ValGroups ref_block_ids_;
};

std::unordered_map<TensorView*, TensorView*> partitionFusion(
    Fusion* fusion,
    const std::unordered_set<TensorView*>& constrained_tvs) {
  FusionGuard fg(fusion);

  const auto all_exprs = fusion->exprs();
  const auto all_tvs = fusion->allTvs();

  std::unordered_map<TensorView*, TensorView*> tv_to_constrained_tv_map;

  for (auto tv : constrained_tvs) {
    tv_to_constrained_tv_map.emplace(tv, tv);
  }

  auto propagateThroughExpr = [&](Expr* expr, Direction dir) -> bool {
    if (!ir_utils::isTvOp(expr)) {
      return false;
    }

    std::cerr << "Prop (" << dir << "): " << expr->toString();

    // The reference of each output may not be the same. Use the first
    // output that has a reference. Conflicting outputs will be
    // resolved later.
    TensorView* src_ref = nullptr;
    const auto& src_vals =
        dir == Direction::Forward ? expr->inputs() : expr->outputs();
    const auto& dst_vals =
        dir == Direction::Forward ? expr->outputs() : expr->inputs();

    auto src_with_ref_it = std::ranges::find_if(src_vals, [&](Val* src) {
      return src->isA<TensorView>() &&
          tv_to_constrained_tv_map.contains(src->as<TensorView>());
    });
    if (src_with_ref_it != src_vals.end()) {
      src_ref =
          tv_to_constrained_tv_map.at((*src_with_ref_it)->as<TensorView>());
    }

    bool updated = false;

    for (auto dst_tv : ir_utils::filterByType<TensorView>(dst_vals)) {
      // If already set, don't overwrite. If not, propagate the output reference
      // if found.
      if (tv_to_constrained_tv_map.contains(dst_tv)) {
        continue;
      } else if (src_ref != nullptr) {
        NVF_ERROR(tv_to_constrained_tv_map.emplace(dst_tv, src_ref).second);
        updated = true;
      }
    }

    return updated;
  };

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

  // Propagates constrained tv grouping from outputs to inputs
  propagate_backward();

  // Initial back prop done
  {
    std::cerr << "Initial backprop done\n";
    for (const auto& [tv, ref] : tv_to_constrained_tv_map) {
      std::cerr << tv->toString() << " -> " << ref->toString() << "\n";
    }
  }

  while (tv_to_constrained_tv_map.size() != all_tvs.size()) {
    std::cerr << "Num tvs total: " << all_tvs.size()
              << ", num entries: " << tv_to_constrained_tv_map.size() << "\n";

    std::vector<TensorView*> ungrouped_tvs;
    std::ranges::copy_if(
        all_tvs, std::back_inserter(ungrouped_tvs), [&](auto tv) {
          return !tv_to_constrained_tv_map.contains(tv);
        });
    std::cerr << "Still ungrouped: " << toDelimitedString(ungrouped_tvs)
              << "\n";

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

  std::cerr << "Final grouping\n";
  for (auto tv : all_tvs) {
    if (tv_to_constrained_tv_map.contains(tv)) {
      std::cerr << tv->toString() << " -> "
                << tv_to_constrained_tv_map.at(tv)->toString() << "\n";
    }
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

} // namespace

bool GreedyScheduler::canScheduleCompileTime(Fusion* fusion) {
  if (!isOptionEnabled(EnableOption::GreedyScheduler)) {
    scheduler_debug_utils::canScheduleRejectReason(
        SchedulerType::Greedy, "Not enabled");
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
  // TODO: Consider the shared memory capacity for resolving conflicts
  return RunTimeChecker::run(fusion, runtime_info, data_cache);
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

  auto constrained_exprs = getAllConstrainedOps(fusion);

  std::cerr << "Constrained ops: " << toDelimitedString(constrained_exprs);

  std::vector<TensorView*> constrained_out_tvs;
  constrained_out_tvs.reserve(constrained_exprs.size());
  std::ranges::transform(
      constrained_exprs,
      std::back_inserter(constrained_out_tvs),
      [](const Expr* expr) { return ir_utils::getTvOutput(expr); });

  std::cerr << "Constrained tvs: " << toDelimitedString(constrained_out_tvs)
            << "\n";

  IdModel id_model(fusion);
  const auto& exact_graph = id_model.buildExactGraph();

  ConstrainedOpScheduler::run(fusion, constrained_out_tvs, exact_graph);

  auto tv_to_ref_map = partitionFusion(
      fusion, {constrained_out_tvs.begin(), constrained_out_tvs.end()});

  fusion->print();

  for (auto constrained_tv : constrained_out_tvs) {
    std::unordered_set<TensorView*> tvs_to_transform;
    for (const auto& [tv, ref] : tv_to_ref_map) {
      if (ref == constrained_tv) {
        tvs_to_transform.insert(tv);
      }
    }

    std::cerr << "Scheduling " << toDelimitedString(tvs_to_transform) << "\n";

    // constrained_tv must be already scheduled at this point

    SetSelector selector(tvs_to_transform);
    MaxLogicalDomainInfoSpanningTree tree(constrained_tv, &selector);
    TransformPropagator tp(constrained_tv);
    tree.traverse(&tp);

    scheduler_utils::parallelizeAllLike(
        constrained_tv, -1, {tvs_to_transform.begin(), tvs_to_transform.end()});
  }

  fusion->print();

  // Resolve conflicts. Find conflicting producer-consumer pairs
  // and insert memory promotion

  FusionInfo fusion_info;
  FusionInfoGuard info_guard(&fusion_info);
  fusion_info.set(std::make_unique<ConcretizedBroadcastDomains>(fusion));
  fusion_info.set(std::make_unique<PaddedParallelDimensions>(
      collectPaddedParallelDims(fusion)));
  fusion_info.set(std::make_unique<IdModel>(fusion, /*build_graphs=*/true));
  fusion_info.set(std::make_unique<ComputeAtMap>(fusion));
  fusion_info.set(std::make_unique<ParallelDimensionMap>(fusion));
  fusion_info.set(std::make_unique<ThreadPredicateMap>(fusion));

  VectorOfUniqueEntries<TensorView*> tvs_to_upload_to_smem;
  SyncMap sync_map(fusion, /*error_on_failure=*/false);
  for (const auto& [tv, pt_map] : sync_map.map()) {
    std::cerr << "Needs raw sync: " << tv->toString() << ", "
              << pt_map.toString() << "\n";
    NVF_ERROR(
        !pt_map.hasBID(),
        "Grid sync not expected: ",
        tv->toString(),
        ", ",
        pt_map.toString());
    tvs_to_upload_to_smem.pushBack(tv);
  }

  for (const auto& tv : tvs_to_upload_to_smem) {
    const auto& tv_sync_map = sync_map.producerConsumerRawSync().at(tv);
    std::vector<Expr*> uses_to_update;
    std::ranges::copy_if(
        tv->uses(), std::back_inserter(uses_to_update), [&](Expr* use) {
          auto it = tv_sync_map.find(ir_utils::getTvOutput(use));
          return it != tv_sync_map.end() && it->second.hasTID();
        });
    NVF_ERROR(!uses_to_update.empty());

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

    IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, copy, tv);

    copy->setMemoryType(MemoryType::Shared);

    TransformReplay::replayCasP(copy, tv, -1);
    std::cerr << "Copy on shared memory: " << copy->toString() << "\n";

    for (auto tv_use : uses_to_update) {
      auto expr = ir_utils::replaceValInExprInputs(tv_use, tv, copy);
      std::cerr << "Replaced expr: " << expr->toString();
    }
  }

  std::cout << std::endl;
  std::cout << "Final:" << std::endl;
  fusion->print();
  std::cout << std::endl;
}

} // namespace nvfuser
