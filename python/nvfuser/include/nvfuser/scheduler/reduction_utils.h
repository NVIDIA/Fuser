// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <scheduler/reduction_heuristic.h>
#include <scheduler/utils.h>
#include <visibility.h>

namespace nvfuser {

namespace reduction_scheduler_utils {

// Consistent parallelization based on provided reduction parameters. Provided
// tensor is expected to be reduced by canonicalDimReduction before sending
// here. reduction_tv should be provided as the tensorview to reduce.
// RFactor of reduction_tv will be returned if applicable otherwise reduction_tv
// is returned
TensorView* scheduleReductionTV(
    const ReductionParams* rparams,
    TensorView* reduction_tv,
    bool has_iter_axis);

// Propagate transformations with internal cutoff boundary at boundaryNodesSet
// in P2C forward propagate, disable propagation to TensorView in
// boundaryNodesSet in C2P backward propagate, disable propagation from
// TensorView in boundaryNodesSet
NVF_API void propagateTransformation(
    TensorView* reference_tv,
    const std::unordered_set<TensorView*>& boundaryNodesSet =
        std::unordered_set<TensorView*>());

// Propagate RFactor from first reduction TensorView to others
void propagateRFactor(
    TensorView* reference_tv,
    TensorView* reduction_tv,
    const std::vector<TensorView*>& reduction_tvs);

// Get all cached input/output and shared memory TensorViews that are
// vectorizable and unrollable.
//
// Parameters:
//   reference_tv: TensorView created during RFactor, used to find vectorizable
//                 TensorViews.
//   is_vectorize: Indicates if vectorization is applied in the scheduler.
//   cached_inputs: Inputs cached in registers or shared memory.
//   cached_outputs: Outputs cached in registers.
NVF_API std::unordered_set<TensorView*> getCachedTvsToUnrollOrVectorize(
    TensorView* reference_tv,
    bool is_vectorize,
    const std::vector<TensorView*>& cached_inputs,
    const std::vector<std::pair<TensorView*, TensorView*>>& cached_outputs);

// Propagate parallelization from the reference TensorView to other TensorViews.
// Unroll and Vectorize types are explicitly handled for
// TensorViews in unroll_vectorizable_cached_tvs. Clears unroll parallelization
// for reduction_tv and reference_tv if they shouldn't be unrolled.
//
// Parameters:
//   reduction_tv: The reduction TensorView being scheduled and parallelized.
//                 Needs to clear its vectorization or convert to grouped
//                 reduction.
//
//   reference_tv: The reference TensorView being scheduled and parallelized,
//                 propagates parallelization to other selected TensorViews.
//
//   is_unroll_or_vectorization: Indicates if unroll or vectorization is used in
//                               the scheduler.
//
//   reduction_tvs: All reduction TensorViews in the fusion. May add grouped
//                  parallelization.
//
//   unroll_vectorizable_cached_tvs: Cached TensorViews that are unrollable
//                                   or vectorizable.
//
//   selected_tvs: TensorViews selected for parallelization, default is all Tvs.
NVF_API void propagateParallelization(
    TensorView* reduction_tv,
    TensorView* reference_tv,
    const bool is_unroll_or_vectorization,
    const bool use_grouped_reduction,
    const std::vector<TensorView*>& reduction_tvs,
    const std::unordered_set<TensorView*>& unroll_vectorizable_cached_tvs,
    const std::vector<TensorView*>& selected_tvs = {});

// Sort and rfactor the reference tv in a consistent way for reduction inliner.
// Order of the sort is:
//
// [i-device dims, i-block dims, i-thread dims, i-constant sized, i-non-constant
// sized
//  r-block dims, r-thread dims, r-non-constant sized, r-constant sized,
//  i/r-unswitched, i/r-unroll/vectorized, broadcasted dims]
//
// Rfactored axes are reductions bound to grid or blocks. If no axes are bound
// to a grid or block dimension it will rfactor the r-unswitch dimension.
// Reduction inliner expects an rfactored domain.
NVF_API TensorView* sortAndRFactor(
    TensorView* reference_tv,
    const bool is_non_persistent_outer_reduction = false);

// If project_to_inputs is true, take all projectable persistent buffers,
// and move them to the inputs. Otherwise, try to project to their immediate
// producers if these producers are persistent buffers.
// This function create dummy outputs which should be used in later stages of
// the scheduling.
NVF_API std::vector<TensorView*> projectPersistentBuffers(
    Fusion* fusion,
    const scheduler_utils::PersistentBufferInfo& persistent_info,
    const bool project_to_inputs);

//! Get reduction types based on the given fusion or reduction tvs.
//! If there are no reduction tvs, return None.
//! If there are only inner reduction tvs, return Inner.
//! If there are only outer reduction tvs, return Outer.
//! If there are both inner and outer reduction tvs, return InnerOuter.
enum class ReductionType { Inner, Outer, InnerOuter, None };
std::ostream& operator<<(std::ostream& os, ReductionType reduction_type);
std::string toString(ReductionType reduction_type);
ReductionType getReductionType(Fusion* fusion);
ReductionType getReductionType(const std::vector<TensorView*>& reduction_tvs);

/**
 * @brief Vectorize shared memory consumers
 *
 * Applies vectorization to shared memory consumers.
 * If extent of the last dim multiples vectorization factor exceeds hardware
 * limitations, additional split is added.
 *
 * @param smem_consumers Vector of TensorView pointers representing shared
 * memory consumers
 * @param io_vectorization_factor Vectorization factor set for fusion inputs and
 * outputs
 * @note TODO: Optimize writing to shared memory and address bank conflicts for
 * float32 with innermost extent of 8
 */
void sharedMemoryConsumerVectorization(
    std::vector<TensorView*>& smem_consumers,
    const int64_t io_vectorization_factor);

} // namespace reduction_scheduler_utils
} // namespace nvfuser
