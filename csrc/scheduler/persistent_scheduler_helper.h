// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <scheduler/all_schedulers.h>
#include <scheduler/utils.h>

namespace nvfuser {

//! Reduction types based on the given fusion.
//! If there are no reduction tvs, None.
//! If there are only inner reduction tvs, Inner.
//! If there are only outer reduction tvs, Outer.
//! If there are both inner and outer reduction tvs, InnerOuter.
enum class ReductionType { Inner, Outer, InnerOuter, None, NotInitiliazed };

class PersistentSchedulerHelper {
 protected:
  static bool compileTimeCheckReductionAxis(
      Fusion* fusion,
      const std::vector<TensorView*>& reduction_tvs,
      ScheduleHeuristic heuristic);

  static bool leadingCommonCompileTimeCheck(
      Fusion* fusion,
      ScheduleHeuristic heuristic);

  static bool tailingCommonCompileTimeCheck(
      Fusion* fusion,
      const std::vector<TensorView*>& reduction_tvs,
      ScheduleHeuristic heuristic);

  static bool checkReductionType(
      const std::vector<TensorView*>& reduction_tvs,
      ScheduleHeuristic heuristic);

  static bool commonCompileTimeCheck(
      Fusion* fusion,
      ScheduleHeuristic heuristic);

  static bool runTimeCheckIterSize(
      const scheduler_utils::ReductionTvProperties& properties,
      ScheduleHeuristic heuristic);
};

} // namespace nvfuser
