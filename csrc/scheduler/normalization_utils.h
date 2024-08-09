// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <executor_params.h>
#include <ir/all_nodes.h>
#include <scheduler/heuristic_types.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <cmath>
#include <optional>
#include <ostream>
#include <vector>

namespace nvfuser {
class SchedulerRuntimeInfo;
class HeuristicSummary;

namespace normalization_scheduler_utils {

//! Utility class to iterate candidates of launch configurations in a
//! preferred order. The iteration order is defined as:
//!
//!   for bdimx in all valid bdimx in an decreasing order
//!     for gdimy in valid gdimy values in an increasing order
//!
//! Each of bdimx and gdimy determines bdimy and gdimx, respecitively,
//! such that the number of threads per block is always 256 and the
//! number of blocks is always equal to the number of SMs.
class PreferredLaunchConfig {
 public:
  //! Minimum blockDim.x.
  static constexpr int kMinBdimx = 8;
  //! Maximum blockDim.x.
  static constexpr int kMaxBdimx = 16;

  PreferredLaunchConfig();

  int bdimx() const {
    return bdimx_;
  }

  int bdimy() const {
    return bdimy_;
  }

  int gdimx() const {
    return gdimxAt(grid_dims_pos_);
  }

  int gdimy() const {
    return gdimyAt(grid_dims_pos_);
  }

  //! Peek the next gdimx. -1 is returned if no further gdimx is available.
  int peekNextGdimx() const;

  //! Peek the next gdimy. -1 is returned if no further gdimy is available.
  int peekNextGdimy() const;

  //! Move to the next launch configuration. Will be marked as invalid
  //! if no valid configuration exists. Return true if successfully moved.
  bool moveToNextConfig();

  //! Try setting blockDim to the next valid config if
  //! available. Return false if no valid config exists. gridDim is
  //! reset.
  bool moveToNextBdim();

  //! Query if the next configuration will cause blockDim.x to become
  //! smaller.
  bool isNextSmallerBdimx() const;

  //! Query if blockDim.x can be further lowered
  bool canLowerBdimx() const;

  //! Query if no valid configuration is found
  bool isInvalid() const {
    return !valid_;
  }

 private:
  //! Populate the list of valid gridDim configurations
  void initValidGdims();

  int gdimxAt(int pos) const {
    return valid_grid_dims_.at(pos).first;
  }

  int gdimyAt(int pos) const {
    return valid_grid_dims_.at(pos).second;
  }

  //! Set blockDim.x and in turn blockDim.y. Return true if the
  //! specified blockDim.x is successfully set. If dry_run is true,
  //! just check if the given config is valid but do not modify the
  //! current config.
  bool setBdimx(int bdimx, bool dry_run = false);

  void resetGdim() {
    grid_dims_pos_ = 0;
  }

  void resetBdim() {
    // Start with the maximum bdimx and lower it until satisfactory
    // config is found
    setBdimx(kMaxBdimx);
  }

  //! Try setting gridDim to the next valid config if
  //! available. Return false if no valid config exists
  bool moveToNextGdim();

  int getNextGdimsPos() const;

  void invalidate() {
    valid_ = false;
  }

  friend std::ostream& operator<<(std::ostream& os, PreferredLaunchConfig cfg) {
    os << "{gdimx: " << cfg.gdimx() << ", gdimy: " << cfg.gdimy()
       << ", bdimx: " << cfg.bdimx() << ", bdimy: " << cfg.bdimy() << "}";
    return os;
  }

 private:
  //! Remember if it is still a valid configuration
  bool valid_ = false;

  //! List of valid gridDims ordered by the dimension of
  //! gridDim.x. Larger gridDim.x is preferred as it would promote
  //! larger independent parallelism
  std::vector<std::pair<int, int>> valid_grid_dims_;
  //! The offset of the Current gridDim in valid_grid_dims_
  int grid_dims_pos_ = 0;

  //! Current blockDim.x
  int bdimx_ = 0;
  //! Current blockDim.y
  int bdimy_ = 0;
};

//! Scheduling parameters for grid outer normalization
struct GridOuterNormalizationParams {
  LaunchParams launch_params;
  int64_t persistent_buffer_factor = -1;
  int64_t unswitch_factor = -1;
};

std::optional<GridOuterNormalizationParams> getGridOuterNormalizationParams(
    int64_t total_reduction_numel,
    int64_t total_iteration_numel,
    int64_t vectorize_factor,
    int64_t persistent_buffer_size);

//! check iter type of each domain in inner and outer reduction tvs
//! inner reduction must be [I,I,...R,R]
//! outer reduction must be [R,R,...I,I]
bool checkIfReductionsAreInnerOuter(
    const std::vector<TensorView*>& inner_reduction_tvs,
    const std::vector<TensorView*>& outer_reduction_tvs);

//! check if the inner reduction has shared input with outer reduction
bool hasSharedInput(
    const std::vector<TensorView*>& inner_reduction_tvs,
    const std::vector<TensorView*>& outer_reduction_tvs);

//! The first part of outer reduction is computed with inner reduction and the
//! second part is scheduled separately. So, (1) the outer reduction tvs can
//! only be connected with inner reduction tvs through their producers. (2)
//! Outer reduction tvs are also scheduled separately and they can only be
//! connected through their producers.
bool isConnectedOnlyThroughReductionProducer(
    const std::vector<TensorView*>& inner_reduction_tvs,
    const std::vector<TensorView*>& outer_reduction_tvs);

// Returns true if every iteration domain in inner reduction tv is a reduction
// domain in outer reduction tv.
bool isReductionIterationAxisMatched(
    const std::vector<TensorView*>& inner_reduction_tvs,
    const std::vector<TensorView*>& outer_reduction_tvs);

//! in combined_inner_outer_reduction, the partial results of outer reductions
//! must be persistent, calculate the size of these buffers when estimate
//! register usage
int64_t partialReductionBufferSize(
    const std::vector<TensorView*>& outer_reduction_tvs,
    SchedulerRuntimeInfo& runtime_info);

//! Calculate the persistent buffer batches and threads per block.
//! Start from a large value of inner_dim_numel / (inner_vect * warpSize/4),
//! gradually reduce to small values but not smaller than a threshold determined
//! by inner_dim_numel and outer_dim_numel. If the persistent buffer batch is
//! smaller than the maximum allowed batch which is determined by the avilable
//! registers, this function will return that batch value. Otherwise, it will
//! return nullopt except when ignore_register_size_limit is true where it will
//! return whatever the batch value is.
// This exception is needed because the register usage in canScheduleRuntime is
// based on std::min(project_buffer, not_project_buffer). However, in
// getPersistentHeuristics() we enforce project_buffer to input if dtype=float
// and feature size <=14K. It leads to register spills but still faster than
// unprojected version due to the reuse of a input para in this grid persistent
// kernel. This is a tmp solution before we have a new persistent heuristics,
// where the projection should not soley based on size of buffers.
std::pair<std::optional<int64_t>, int64_t>
getOptionalInnerOuterPersistentBufferBatches(
    const int64_t inner_dim_numel,
    const int64_t outer_dim_numel,
    const int64_t persistent_buffer_size,
    const int64_t vectorize_factor,
    const int64_t warp_size,
    const bool ignore_register_size_limit);

// Return a scheduleHeuristic based on reduction types.
using ReductionType = reduction_scheduler_utils::ReductionType;
ScheduleHeuristic getPersistentHeuristicFor(ReductionType reduction_type);

// get argument passed to innerPersistentHeuristic and outerPersistentHeuristic
struct PersistentKernelProperties {
  int64_t inner_most_dimension_numel;
  int64_t total_reduction_numel;
  int64_t total_iteration_numel;
  int64_t max_persistent_buffer_size;
  int64_t n_tensor_inputs;
  int64_t max_dtype_size;
  int64_t vectorize_factor;
  bool project_persistent_buffers;
  PrimDataType index_type;
  bool has_exp_op;
  std::vector<TensorView*> persistent_buffers;
  std::string toString() const {
    std::stringstream ss;
    ss << "===== Persistent Kernel Properties ========\n"
       << "inner_most_dimension_numel: " << inner_most_dimension_numel << "\n"
       << "total_reduction_numel: " << total_reduction_numel << "\n"
       << "total_iteration_numel: " << total_iteration_numel << "\n"
       << "max_persistent_buffer_size: " << max_persistent_buffer_size << "\n"
       << "n_tensor_inputs: " << n_tensor_inputs << "\n"
       << "max_input_dtype_size: " << max_dtype_size << "\n"
       << "max allowed vectorize_factor: " << vectorize_factor << "\n"
       << "project_persistent_buffers: " << project_persistent_buffers << "\n";
    return ss.str();
  }
};
PersistentKernelProperties getPersistentKernelProperties(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache,
    ScheduleHeuristic heuristic);

// Verify the presence of a reduction TensorView connected to a Fusion input
void checkReductionTvForScheduling(Fusion* fusion, TensorView* ref_red_tv);

// Check the operations and input tensors of the fusion. This
// verification is a common step shared by all persistent kernel implementations
// during compile-time checks.
bool checkOpsAndInputs(Fusion* fusion, ScheduleHeuristic heuristic);

// Returns true if the reduction pattern is consistent. For the
// InnerPersistentKernelScheduler and OuterPersistentKernelScheduler, a single
// vector of TensorViews is provided, while for the
// InnerOuterPersistentKernelScheduler, two vectors of TensorViews are provided.
bool checkReductionPattern(
    Fusion* fusion,
    ScheduleHeuristic schedule_heuristic,
    const std::vector<TensorView*>& reduction_tvs1,
    const std::vector<TensorView*>& reduction_tvs2 = {});

// The compile-time checks for both the InnerPersistentKernelScheduler and
// OuterPersistentKernelScheduler are identical. These checks are constructed
// using checkOpsAndInputs, checkReductionPattern, and checkViewBufferTopology.
bool compileTimeCheck(Fusion* fusion, ScheduleHeuristic schedule_heuristic);

// Common preparations before the actual schedule, used by all persistent
// schedulers. Write to dummy_outputs, cached_inputs, reduction_tvs, and
// cached_outputs.
void beforeSchedule(
    Fusion* fusion,
    const ReductionParams& rparams,
    std::vector<TensorView*>& dummy_outputs,
    std::vector<TensorView*>& cached_inputs,
    std::vector<TensorView*>& reduction_tvs,
    std::vector<std::pair<TensorView*, TensorView*>>& cached_outputs);

// schedule a reduction tv, used by all persistent schedulers.
// will group reduction ops for OuterPersistentKernelScheduler with multiple
// reduction tvs.
TensorView* scheduleReductionGeneral(
    Fusion* fusion,
    const ReductionParams& rparams,
    std::vector<TensorView*>& reduction_tvs,
    ScheduleHeuristic schedule_heuristic);

// Used by InnerPersistentKernelScheduler and  OuterPersistentKernelScheduler
void schedulePersistentKernel(
    Fusion* fusion,
    const ReductionParams& rparams,
    ScheduleHeuristic schedule_heuristic);

// Get max register or shared memory size for persistent buffer
int64_t getMaxRegOrSharedMemorySizeForPersistentBuffer(
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<TensorView*>& persistent_buffers);

// Returns true if persistent buffers are projected to inputs, meaning the
// inputs are cached instead of the persistent buffers. The decision of
// projection is primarily based on the required sizes of the two cases --
// projection is done if projecting to the inputs results in a smaller size.

// This function is used by inner persistent and InnerOuter persistent
// schedulers.
// TODO: Outer persistent scheduler should also use this function.
// If the scheduler is innerOuter with outer broadcast, projection is allowed
// even it leads to a larger buffer size becuase the scheduled kernel allows the
// reuse of the outer broadcast Tv when iterating over the outer reduction
// dimension and leads to higher performance ( TODO: needs re-evaluate, may not
// true if the buffer size is increased a lot when projecting to inputs). See
// https://github.com/NVIDIA/Fuser/issues/402

// However, we experimentally found that certain relatively expensive operations
// should not be projected even when that would require a larger buffer size.
// Specifically,
// - rng: should never be projected no matter how much larger the buffer would
// consume
// - exp in inner normalization: only allowed to get projected if the buffer is
// smaller than a certain size Otherwise, as long as the projected inputs are
// smaller than the original persistent buffers, this function returns true.
bool isProjectBufferToInputs(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const scheduler_utils::PersistentBufferInfo& persistent_buffer_info,
    const scheduler_utils::PersistentBufferSizeReturn&
        persistent_buffer_size_info,
    const ScheduleHeuristic sh,
    const bool check_projected_buffer_size = true);

// move persistent buffer marked in rparams->smem_persistent_buffers from
// register to smem
void movePersistentBufferToSmem(
    Fusion* fusion,
    const ReductionParams& rparams,
    const std::vector<TensorView*>& cached_inputs);

} // namespace normalization_scheduler_utils
} // namespace nvfuser
