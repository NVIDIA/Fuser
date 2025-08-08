// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <debug.h>
#include <exceptions.h>
#include <instrumentation.h>
#include <iter_visitor.h>
#include <options.h>
#include <scheduler/greedy.h>
#include <scheduler/runtime_info.h>

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
        expr->isOneOf<LoadStoreOp, UnaryOp, BinaryOp, TernaryOp, ScanOp>();
  }

 private:
  bool can_schedule_ = true;
};

std::vector<Expr*> getAllConstrainedOps(Fusion* fusion) {
  return ir_utils::getOpsOfType<ArgsortOp, ScanOp>(fusion);
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

  void handle(ScanOp* scan) override {
    std::cerr << "Scheduling " << scan->toString();

    auto out_tv = ir_utils::getTvOutput(scan);

    NVF_ERROR_EQ(
        out_tv->getLogicalDomain(),
        out_tv->getLoopDomain(),
        "For now, logical and loop domains are assumed to be the same: ",
        out_tv->toString());

    auto scan_dim = out_tv->domain()->noReductions().at(scan->dim());
    scan_dim->parallelize(ParallelType::TIDx);

    // Move the scan_dim innermost
    out_tv->reorder({{scan->dim(), -1}, {-1, scan->dim()}});

    std::cerr << out_tv->toString() << "\n";

    // Merge the remaining IDs
    if (std::ssize(out_tv->getLoopDomain()) > 2) {
      out_tv->flatten(0, std::ssize(out_tv->getLoopDomain()) - 1);
    }

    NVF_ERROR_LE(std::ssize(out_tv->getLoopDomain()), 2);
    NVF_ERROR_EQ(out_tv->axis(-1)->getParallelType(), ParallelType::TIDx);

    if (std::ssize(out_tv->getLoopDomain()) == 2) {
      out_tv->axis(0)->parallelize(ParallelType::BIDx);
    }
  }
};

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

  ConstrainedOpScheduler::run(fusion);

  fusion->print();
}

} // namespace nvfuser
