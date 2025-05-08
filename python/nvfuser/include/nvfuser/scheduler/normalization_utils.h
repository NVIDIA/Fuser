// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <id_model/id_model.h>
#include <ir/all_nodes.h>
#include <runtime/executor_params.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/scheduler_types.h>
#include <scheduler/utils.h>
#include <val_graph.h>

#include <cmath>
#include <optional>
#include <ostream>
#include <vector>

namespace nvfuser {
class SchedulerRuntimeInfo;
class HeuristicDataCache;

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

// Return a scheduleHeuristic based on reduction types.
using ReductionType = reduction_scheduler_utils::ReductionType;
SchedulerType getPersistentHeuristicFor(ReductionType reduction_type);

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
  bool has_rng_op;
  bool disable_project_to_avoid_recompute;
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
       << "disable_project_to_avoid_recompute: "
       << disable_project_to_avoid_recompute << "\n"
       << "project_persistent_buffers: " << project_persistent_buffers << "\n"
       << "persistent_buffers: " << toDelimitedString(persistent_buffers)
       << "\n";
    return ss.str();
  }
};
PersistentKernelProperties getPersistentKernelProperties(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    SchedulerType heuristic);

// Verify the presence of a reduction TensorView connected to a Fusion input
void checkReductionTvForScheduling(Fusion* fusion, TensorView* ref_red_tv);

// Check the operations and input tensors of the fusion. This
// verification is a common step shared by all persistent kernel implementations
// during compile-time checks.
bool checkOpsAndInputs(Fusion* fusion, SchedulerType scheduler_type);

// Returns true if the reduction pattern is consistent. For the
// InnerPersistentKernelScheduler and OuterPersistentKernelScheduler, a single
// vector of TensorViews is provided, while for the
// InnerOuterPersistentKernelScheduler, two vectors of TensorViews are provided.
bool checkReductionPattern(
    Fusion* fusion,
    SchedulerType scheduler_type,
    const std::vector<TensorView*>& reduction_tvs1,
    const std::vector<TensorView*>& reduction_tvs2 = {});

// The compile-time checks for both the InnerPersistentKernelScheduler and
// OuterPersistentKernelScheduler are identical. These checks are constructed
// using checkOpsAndInputs, checkReductionPattern, and checkViewBufferTopology.
bool compileTimeCheck(Fusion* fusion, SchedulerType scheduler_type);

// Common preparations before the actual schedule, used by all persistent
// schedulers. Write to dummy_outputs, cached_inputs, reduction_tvs, and
// cached_outputs.
void beforeSchedule(
    Fusion* fusion,
    const ReductionParams* rparams,
    std::vector<TensorView*>& dummy_outputs,
    std::vector<TensorView*>& cached_inputs,
    std::vector<TensorView*>& reduction_tvs,
    std::vector<TensorView*>& smem_consumers,
    std::vector<std::pair<TensorView*, TensorView*>>& cached_outputs);

// schedule a reduction tv, used by all persistent schedulers.
// will group reduction ops for OuterPersistentKernelScheduler with multiple
// reduction tvs.
TensorView* scheduleReductionGeneral(
    Fusion* fusion,
    const ReductionParams* rparams,
    std::vector<TensorView*>& reduction_tvs,
    SchedulerType scheduler_type);

// Used by InnerPersistentKernelScheduler and  OuterPersistentKernelScheduler
void schedulePersistentKernel(
    Fusion* fusion,
    const ReductionParams* rparams,
    SchedulerType scheduler_type);

// Get max register or shared memory size for persistent buffer
int64_t getMaxRegOrSharedMemorySizeForPersistentBuffer(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<TensorView*>& reduction_tvs,
    const scheduler_utils::PersistentBufferInfo& persistent_buffer_info,
    const bool can_use_smem_persistent,
    const bool project_to_inputs);

enum class BufferProjectionStrategy {
  // Recompute persistent buffers from inputs, only need to cache inputs in
  // registers or shared memories, usually used when size of required cached
  // inputs is smaller than the size of persistent buffers.
  ProjectToInputs,
  // Don't project to inputs, to avoid recompute from inputs. This saves
  // computation cost but uses more registers or shared memories. Usually used
  // when the required buffer size  is small and hardware has high bandwidth to
  // flops ratio.
  NoProjectToAvoidRecompute,
  // Project to inputs is disabled due to other reasons, e.g. can't reduce
  // buffer size, recompute requires very expensive rng ops, not supported due
  // to view ops.
  NoProjectOtherReasons
};

// Returns BufferProjectionStrategy based on buffer size, hardware, and fusion
// ops.

// This function is used by inner persistent and InnerOuter persistent
// schedulers.
// Using shared memory to store persistent buffers is not supported yet for
// inner persistent scheduler with 3D reduction type.
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
BufferProjectionStrategy isProjectBufferToInputs(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<TensorView*>& reduction_tvs,
    const scheduler_utils::PersistentBufferInfo& persistent_buffer_info,
    const scheduler_utils::PersistentBufferSizeReturn&
        persistent_buffer_size_info,
    const SchedulerType sh,
    const bool can_use_smem_persistent,
    const bool check_projected_buffer_size = true);

// Set memory type of persistent buffer marked in
// rparams->smem_persistent_buffers as shared memory. Return a vector of the
// consumers of the shared memory tensors, they are cached after the smem
// tensors and will be vectorized by the scheduler if possible to avoid shared
// memory bank conflicts.
std::vector<TensorView*> movePersistentBufferToSmem(
    Fusion* fusion,
    const ReductionParams* rparams,
    const std::vector<TensorView*>& cached_inputs);

// Find the resolution points of a persistent buffer. See also
// the comments of PersistentBufferResolution in utils.cpp. Unlike
// PersistentBufferResolution, this analysis traverses a given fusion
// both forward and backward, which is necessary in some cases. For
// example:
//
// t0 = makeSymbolicTensor(2)
// t1 = makeSymbolicTensor(2)
// t2 = set(t0)
// t3 = sum(t2, 1)
// t4 = broadcast(t3, {false, true})
// t5 = add(t1, t2)
// t6 = add(t4, t1)
// fusion.addOutput(t5)
// fusion.addOutput(t6)
//
// The path from t2 to t3, t4 and t6 is a normalization path. While t1 itself
// does not depend on t2, since it is used with t2, inlining of t2
// also means t1 must be inlined, which in turn means t6 must be
// inlined. However, t6 depends on the reduction, inlining of t2 is
// not possible. For normalization fusions like this pattern,
// PersistentBufferResolution is not able to detect the resolution
// point. getResolutionPointsOf addresses the problem by traversing
// both forward and backward directions. See
// PersistentBufferTest.GetResolutionIssue1123 for a concrete example
std::vector<TensorView*> getResolutionPointsOf(
    TensorView* persistent_buffer,
    IdModel& id_model);

// Return empirical maximum persistent batch size for inner persistent scheduler
int64_t getInnerPersistentMaxBatchSize(bool is_high_bandwidth_flops_ratio);

// Check if an unmappable tensor can be persistent. The primary reason
// of being unable to be persistent is broadcast inlining, which may
// cause inconsistent parallelization
bool isCacheableUnmappableTv(
    TensorView* unmappable_tv,
    const std::vector<TensorView*>& reduction_tvs,
    const ValGraph& almost_exact_graph);

} // namespace normalization_scheduler_utils
} // namespace nvfuser
