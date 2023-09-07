// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <scheduler/registry.h>
#include <scheduler/utils.h>

namespace nvfuser {

class HeuristicSummary;

//! Reduction types based on the given fusion.
//! If there are no reduction tvs, None.
//! If there are only inner reduction tvs, Inner.
//! If there are only outer reduction tvs, Outer.
//! If there are both inner and outer reduction tvs, InnerOuter.
enum class ReductionType { Inner, Outer, InnerOuter, None, NotInitiliazed };

class PersistentKernelScheduler : public SchedulerEntry {
 public:
  explicit PersistentKernelScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  virtual ~PersistentKernelScheduler() = default;

  // schedule using the appropriate scheudler and heuristic
  virtual void schedule(Fusion* fusion) = 0;

  // Dispatch check to appropriate sub class
  // scheduler based on reduction type.
  static bool canScheduleCompileTime(Fusion* fusion);
  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

 protected:
  // common methods shared by sub classes

 private:
  ReductionType reduction_type_;

  // get the appropriate heuristic
  virtual void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) = 0;


  // utility funcitons used only by other member functions in this class.
  static ReductionType getReductionType(Fusion* fusion);
  static bool leadingCommonCompileTimeCheck(
      Fusion* fusion,
      ScheduleHeuristic heuristic);
  static bool tailingCommonCompileTimeCheck(
      Fusion* fusion,
      const std::vector<TensorView*>& reduction_tvs,
      ScheduleHeuristic heuristic);
};

} // namespace nvfuser
