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

//! NoOp scheduler represents the case where scheduler will
//!  not do any scheduling operations and forward the un-scheduled
//!  fusion directly to code generation and kernel compilation.
//!
//! Typical use case of this scheduler is to handle edge cases
//!  such as where all tensors are size-1 or size-0.

class NoOpScheduler : public SchedulerEntry {
 public:
  explicit NoOpScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  //! Check if the no-op heuristics apply in given fusion
  static bool canScheduleCompileTime(Fusion* fusion);

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  void schedule(Fusion* fusion) override;

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);
};

//! Provides a dummy heuristic type to ensure
//!  unified interface on NoOp scheduler.
class NoOpHeuristic : public HeuristicParams {
 public:
  using HeuristicParams::HeuristicParams;

  size_t hash() const override {
    return 0;
  }
  std::shared_ptr<HeuristicParams> clone() const override {
    return std::make_shared<NoOpHeuristic>();
  }
  bool sameAs(const std::shared_ptr<HeuristicParams>& other) const override {
    auto other_casted = std::dynamic_pointer_cast<ReductionParams>(other);
    return other_casted != nullptr && other_casted->cparams == cparams;
  };
};

} // namespace nvfuser
