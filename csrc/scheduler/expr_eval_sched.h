// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <scheduler/heuristic.h>
#include <scheduler/registry.h>

namespace nvfuser {

class Fusion;
class SchedulerRuntimeInfo;
class HeuristicSummary;

//! ExprEval scheduler represents the case where we allocate outputs directly using EE. No code is generated.
class ExprEvalScheduler : public SchedulerEntry {
 public:
  explicit ExprEvalScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  //! This scheduler only accepts matmul and linear nodes
  static bool canScheduleCompileTime(Fusion* fusion);

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  constexpr static ScheduleHeuristic heuristicType() {
    return ScheduleHeuristic::ExprEval;
  }

  void schedule(Fusion* fusion) override;

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);
};

//! Provides a dummy heuristic type to ensure
//!  unified interface on ExprEval scheduler.
class ExprEvalHeuristic : public HeuristicParams {
 public:
  using HeuristicParams::HeuristicParams;

  size_t hash() const override {
    return 0;
  }
  std::shared_ptr<HeuristicParams> clone() const override {
    return std::make_shared<ExprEvalHeuristic>();
  }
  bool sameAs(const std::shared_ptr<HeuristicParams>& other) const override {
    auto other_casted = std::dynamic_pointer_cast<ReductionParams>(other);
    return other_casted != nullptr && other_casted->cparams == cparams;
  };
};

} // namespace nvfuser