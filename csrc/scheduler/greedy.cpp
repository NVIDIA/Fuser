// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <debug.h>
#include <device_lower/analysis/sync_information.h>
#include <device_lower/utils.h>
#include <exceptions.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <options.h>
#include <scheduler/greedy.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/maxinfo_propagator.h>
#include <scheduler/utils.h>
#include <transform_replay.h>

#include <ranges>
#include <vector>

namespace nvfuser {

namespace {

class StaticChecker : private IterVisitor {
 public:
  static bool run(Fusion* fusion) {
    StaticChecker checker(fusion);
    return checker.can_schedule_;
  }

 private:
  StaticChecker(Fusion* fusion) {
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
  }

 private:
  bool can_schedule_ = true;
};

std::vector<Expr*> getAllConstrainedOps(Fusion* fusion) {
  return ir_utils::getOpsOfType<ArgsortOp, ScanOp, PadOp>(fusion);
}

class ConstrainedOpScheduler : private IterVisitor {
 public:
  static void run(Fusion* fusion) {
    ConstrainedOpScheduler scheduler(fusion);
  }

 private:
  ConstrainedOpScheduler(Fusion* fusion) {
    traverse(fusion);
  }

  void handle(PadOp* pad) override {
    std::cerr << "Scheduling " << pad->toString();

    auto out_tv = ir_utils::getTvOutput(pad);

    scheduleConstrainedTv(out_tv, pad->getPaddedAxes());
  }

  void handle(ScanOp* scan) override {
    std::cerr << "Scheduling " << scan->toString();

    auto out_tv = ir_utils::getTvOutput(scan);

    scheduleConstrainedTv(out_tv, {scan->dim()});
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

    // Merge the remaining IDs
    if (std::ssize(tv->getLoopDomain()) > 2) {
      tv->flatten(0, std::ssize(tv->getLoopDomain()) - 2);
    }

    NVF_ERROR_LE(std::ssize(tv->getLoopDomain()), 2);
    NVF_ERROR_EQ(
        tv->axis(-1)->getParallelType(),
        ParallelType::TIDx,
        "Unexpected loop domain: ",
        tv->toString());

    if (std::ssize(tv->getLoopDomain()) == 2) {
      tv->axis(0)->parallelize(ParallelType::BIDx);
    }

    std::cerr << "Scheduled: " << tv->toString() << "\n";
  }
};

std::unordered_map<TensorView*, TensorView*> partitionFusion(
    Fusion* fusion,
    const std::unordered_set<TensorView*>& constrained_tvs) {
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
    return false;
  }

  return StaticChecker::run(fusion);
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

  auto tv_to_ref_map = partitionFusion(
      fusion, {constrained_out_tvs.begin(), constrained_out_tvs.end()});

  ConstrainedOpScheduler::run(fusion);

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

  // TODO: Resolve conflicts. Find conflicting producer-consumer pairs
  // and insert memory promotion

  SyncMap sync_map(fusion, /*error_on_failure=*/false);
  for (const auto& [tv, pt_map] : sync_map.map()) {
    std::cerr << "Needs raw sync: " << tv->toString() << ", "
              << pt_map.toString() << "\n";
  }
}

} // namespace nvfuser
