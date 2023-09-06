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
enum class ReductionType { Inner, Outer, InnerOuter, None };

class PersistentKernelScheduler : public SchedulerEntry {
 public:
  explicit PersistentKernelScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  virtual ~PersistentKernelScheduler() = default;

  // schedule using the appropriate scheudler and heuristic
  virtual void schedule(Fusion* fusion) = 0;

  // check if can schedule
  static bool canScheduleCompileTime(Fusion* fusion) = 0;

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) = 0;

 protected:


 private:
  // get the appropriate heuristic
  virtual void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) = 0;

  // reduction type, this instance is corresponding to.
  ReductionType reduction_type_;
};



} // namespace nvfuser
