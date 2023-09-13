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

// convenience function to get persistent kernel heuristics using
// runtime_inputs. used in cpp tests.
TORCH_CUDA_CU_API std::shared_ptr<ReductionParams> getPersistentHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache = nullptr);

// defines utility functions used by persistent kernel schedulers
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
      TensorView* reference_tv,
      ScheduleHeuristic heuristic);

  static bool checkReductionType(
      const std::vector<TensorView*>& reduction_tvs,
      ScheduleHeuristic heuristic);

  static bool innerOrOuterCompileTimeCheck(
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
          const std::vector<TensorView*>& reduction_tvs,
          TensorView* reference_tv);

  static std::tuple<bool, int64_t, scheduler_utils::PersistentBufferSizeReturn>
  checkAndSetPersistentBufferHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache);

  static std::pair<int64_t, int64_t> getTensorInputNumAndMaxTypeSize(
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache,
      TensorView* reduced_tv);

  //! helper functions used by scheduleKernel
  static void beforeSchedule(
      Fusion* fusion,
      const ReductionParams& rparams,
      std::vector<TensorView*>& dummy_outputs,
      std::vector<TensorView*>& cached_inputs,
      std::vector<TensorView*>& reduction_tvs,
      std::vector<std::pair<TensorView*, TensorView*>>& cached_outputs);

  // schedule inner or outer reduction tv
  static TensorView* scheduleReductionGeneral(
      Fusion* fusion,
      const ReductionParams& rparams,
      std::vector<TensorView*>& reduction_tvs);
};

} // namespace nvfuser
