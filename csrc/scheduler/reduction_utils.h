// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <ir/all_nodes.h>
#include <scheduler/reduction_heuristic.h>

namespace nvfuser {

namespace reduction_scheduler_utils {

// Consistent parallelization based on provided reduction parameters. Provided
// tensor is expected to be reduced by canonicalDimReduction before sending
// here. reduction_tv should be provided as the tensorview to reduce.
// RFactor of reduction_tv will be returned if applicable otherwise reduction_tv
// is returned
TensorView* scheduleReductionTV(
    const ReductionParams& rparams,
    TensorView* reduction_tv,
    bool has_iter_axis);

// Inlining function intended for single or multi reduction fusions.
TORCH_CUDA_CU_API void multiReductionInliner(
    Fusion* fusion,
    TensorView* reduction_tv,
    TensorView* reference_tv,
    const bool unroll,
    const bool vectorize,
    const bool is_outer_grid_persistence,
    std::vector<TensorView*> reduction_tvs,
    std::vector<TensorView*> cached_inputs,
    std::vector<std::pair<TensorView*, TensorView*>> cached_outputs,
    std::vector<TensorView*> dummy_outputs = {});

// Propagate transformations with internal cutoff boundary at boundaryNodesSet
// in P2C forward propagate, disable propagation to TensorView in
// boundaryNodesSet in C2P backward propagate, disable propagation from
// TensorView in boundaryNodesSet
TORCH_CUDA_CU_API void propagateTransformation(
    TensorView* reference_tv,
    const std::unordered_set<TensorView*>& boundaryNodesSet =
        std::unordered_set<TensorView*>());

// Propagate RFactor from first reduction TensorView to others
TORCH_CUDA_CU_API void propagateRFactor(
    TensorView* reference_tv,
    TensorView* reduction_tv,
    const std::vector<TensorView*>& reduction_tvs);

// Propagate Parallelization from reference TensorView to other TensorViews
// Parallel types Unroll, Vectorize, and MisalignedVectorize are explicitly
// handled for tensorviews in cached_inputs and cached_outputs.
// If reduction_tv and reference_tv shouldn't be unrolled, clear that parallel
// type. unroll and vectorize are members of ReductionParams
TORCH_CUDA_CU_API void propagateParallelization(
    Fusion* fusion,
    TensorView* reduction_tv,
    TensorView* reference_tv,
    const bool unroll,
    const bool vectorize,
    const bool is_outer_grid_persistence,
    const std::vector<TensorView*>& reduction_tvs,
    const std::vector<TensorView*>& cached_inputs,
    const std::vector<std::pair<TensorView*, TensorView*>>& cached_outputs,
    const std::vector<TensorView*>& selected_tvs = {});

// Sort and rfactor the reference tv in a consistent way for reduction inliner.
// Order of the sort is:
//
// [i-block dims, i-thread dims, i-non-constant sized, i-constant sized,
//  r-block dims, r-thread dims, r-non-constant sized, r-constant sized,
//  i/r-unswitched, i/r-unroll/vectorized, broadcasted dims]
//
// Rfactored axes are reductions bound to grid or blocks. If no axes are bound
// to a grid or block dimension it will rfactor the r-unswitch dimension.
// Reduction inliner expects an rfactored domain.
TORCH_CUDA_CU_API TensorView* sortAndRFactor(TensorView* reference_tv);

// If project_to_inputs is true, take all projectable persistent buffers,
// and move them to the inputs. Otherwise, try to project to their immediate
// producers if these producers are persistent buffers.
// This function create dummy outputs which should be used in later stages of
// the scheduling.
TORCH_CUDA_CU_API std::vector<TensorView*> projectPersistentBuffers(
    Fusion* fusion,
    const bool project_to_inputs);

} // namespace reduction_scheduler_utils
} // namespace nvfuser
