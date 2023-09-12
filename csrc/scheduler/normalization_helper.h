// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <scheduler/heuristic_types.h>
#include <scheduler/utils.h>

namespace nvfuser {

class PersistentSchedulerHelper {
 protected:
  //! helper functions used by compileTime and runTime checks
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

  //! helper functions used by getHeuristics
  static std::
      tuple<TensorView*, scheduler_utils::ReductionTvProperties, int64_t>
      getCommonHeuristicParams(
          Fusion* fusion,
          SchedulerRuntimeInfo& runtime_info,
          HeuristicSummary* data_cache,
          const std::vector<TensorView*>& reduction_tvs);

  static std::pair<bool, int64_t> checkAndSetPersistentBufferHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache,
      const std::vector<TensorView*>& reduction_tvs = {},
      const bool is_inner_outer = false);

  static std::pair<int64_t, int64_t> getTensorInputNumAndMaxTypeSize(
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache,
      TensorView* reduced_tv);

  static int64_t getOutReductionDataTypeSize(
      const std::vector<TensorView*>& reduction_tvs);

  //! helper functions used by scheduleKernel
  static void beforeSchedule(
      Fusion* fusion,
      const ReductionParams& rparams,
      std::vector<TensorView*>& dummy_outputs,
      std::vector<TensorView*>& cached_inputs,
      std::vector<TensorView*>& reduction_tvs,
      std::vector<std::pair<TensorView*, TensorView*>>& cached_outputs);

  // If called from schedulePersistentKernel, reduction_tvs are either inner
  // reductions or outer reductions. If called from
  // schedulePersistentKernelInnerOuter, reduction_tvs are inner reductions,
  // outer reductions are handled by scheduleCombinedOuter.
  static TensorView* scheduleReductionGeneral(
      Fusion* fusion,
      const ReductionParams& rparams,
      std::vector<TensorView*>& reduction_tvs);
};

} // namespace nvfuser
