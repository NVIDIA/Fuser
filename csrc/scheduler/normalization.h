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
// runtime_inputs. Used in cpp tests.
TORCH_CUDA_CU_API std::shared_ptr<ReductionParams> getPersistentHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache = nullptr);

// Defines utility functions used by persistent kernel schedulers
namespace persistent_scheduler {

//! Check ops and inputs of the given fusion.
//! Used by all persistent kernels in compile time check.
//! This is the first part of the compile time check.
bool checkOpsAndInputs(Fusion* fusion, ScheduleHeuristic heuristic);

//! Check reduction types, inner, outer, or innerOuter
bool checkReductionType(
    const std::vector<TensorView*>& reduction_tvs,
    ScheduleHeuristic heuristic);

//! Check reduction ops have the same axes.
bool checkReductionAxis(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs,
    ScheduleHeuristic heuristic);

//! Check view ops, reduction root size, persistent buffer, and fusion topology.
bool checkViewRootPersistentTopology(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs,
    TensorView* reference_tv,
    ScheduleHeuristic heuristic);

//! compile time check for inner or outer persistent kernel.
//! constructed using checkOpsAndInputs, checkReductionType, checkReductionAxis,
//! and checkViewRootPersistentTopology
bool innerOrOuterCompileTimeCheck(Fusion* fusion, ScheduleHeuristic heuristic);

//! Don't go persistent if we can't use a small fraction of the
//! available SMs yet have a large reduction size.
//! used by inner persistent kernel and innerOuter persistent kernel for run
//! time check.
bool runTimeCheckIterSize(
    const scheduler_utils::ReductionTvProperties& properties,
    ScheduleHeuristic heuristic);

//! helper functions used by getPersistentHeuristic
//! returns reduced tensor, reduction properties, and vectorize factor
std::tuple<TensorView*, scheduler_utils::ReductionTvProperties, int64_t>
getReductionPropertiesVectFactor(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache,
    const std::vector<TensorView*>& reduction_tvs,
    TensorView* reference_tv);

//! returns whether buffer projection is allowed and buffer size.
std::tuple<bool, scheduler_utils::PersistentBufferSizeReturn> getBufferSizeInfo(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache);

//! returns number of tensor inputs and max type size of tensor inputs.
std::pair<int64_t, int64_t> getTensorInputNumAndMaxTypeSize(
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache,
    TensorView* reduced_tv);

//! helper functions used by schedulePersistentKernel
//! Grab the reduction, input, and output tensor views.
//! dummy_outputs are helper tensors for persistent buffer projection.
void beforeSchedule(
    Fusion* fusion,
    const ReductionParams& rparams,
    std::vector<TensorView*>& dummy_outputs,
    std::vector<TensorView*>& cached_inputs,
    std::vector<TensorView*>& reduction_tvs,
    std::vector<std::pair<TensorView*, TensorView*>>& cached_outputs);

//! schedule inner or outer reduction tv
TensorView* scheduleReductionGeneral(
    Fusion* fusion,
    const ReductionParams& rparams,
    std::vector<TensorView*>& reduction_tvs);

// get argument passed to innerPersistentHeuristic and outerPersistentHeuristic
struct PersistentHeuristicArgs {
  scheduler_utils::ReductionTvProperties properties;
  int64_t max_persistent_buffer_size;
  bool project_persistent_buffers;
  int64_t n_tensor_inputs;
  int64_t max_input_dtype_size;
  int64_t vectorize_factor;
};
PersistentHeuristicArgs getInnerOrOuterPersistentHeuristicArgs(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache,
    ScheduleHeuristic heuristic);

//! schedule inner or outer persistent kernel
void scheduleInnerOrOuterPersistentKernel(
    Fusion* fusion,
    const ReductionParams& rparams,
    ScheduleHeuristic heuristic);

} // namespace persistent_scheduler
} // namespace nvfuser
