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
namespace PersistentSchedulerHelper {
//! helper functions used by compileTime and runTime checks
bool compileTimeCheckReductionAxis(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs,
    ScheduleHeuristic heuristic);

bool leadingCommonCompileTimeCheck(Fusion* fusion, ScheduleHeuristic heuristic);

bool tailingCommonCompileTimeCheck(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs,
    TensorView* reference_tv,
    ScheduleHeuristic heuristic);

bool checkReductionType(
    const std::vector<TensorView*>& reduction_tvs,
    ScheduleHeuristic heuristic);

bool innerOrOuterCompileTimeCheck(Fusion* fusion, ScheduleHeuristic heuristic);

bool runTimeCheckIterSize(
    const scheduler_utils::ReductionTvProperties& properties,
    ScheduleHeuristic heuristic);

// argument passed to 
sturct HeuristicArgs {
  scheduler_utils::ReductionTvProperties properties;
  int64_t max_persistent_buffer_size;
  bool project_persistent_buffers;
  int64_t n_tensor_inputs;
  int64_t max_input_dtype_size;
  int64_t vectorize_factor;
};
//! helper functions used by getHeuristics
std::tuple<TensorView*, scheduler_utils::ReductionTvProperties, int64_t>
getCommonHeuristicParams(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache,
    const std::vector<TensorView*>& reduction_tvs,
    TensorView* reference_tv);

std::tuple<bool, scheduler_utils::PersistentBufferSizeReturn>
checkAndSetPersistentBufferHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache);

std::pair<int64_t, int64_t> getTensorInputNumAndMaxTypeSize(
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache,
    TensorView* reduced_tv);

//! helper functions used by scheduleKernel
void beforeSchedule(
    Fusion* fusion,
    const ReductionParams& rparams,
    std::vector<TensorView*>& dummy_outputs,
    std::vector<TensorView*>& cached_inputs,
    std::vector<TensorView*>& reduction_tvs,
    std::vector<std::pair<TensorView*, TensorView*>>& cached_outputs);

// schedule inner or outer reduction tv
TensorView* scheduleReductionGeneral(
    Fusion* fusion,
    const ReductionParams& rparams,
    std::vector<TensorView*>& reduction_tvs);
}; // namespace PersistentSchedulerHelper

} // namespace nvfuser
