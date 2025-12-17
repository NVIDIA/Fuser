// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <compute_at_map.h>
#include <exceptions.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <scheduler/tools/domain_map.h>
#include <scheduler/utils.h>

#include <optional>
#include <vector>

namespace nvfuser {

class SchedulerRuntimeInfo;
class HeuristicDataCache;

namespace pointwise_utils {

// Return reference tensor view.
TensorView* getReferenceTensor(Fusion* fusion);

// Get the largest output tensor using HeuristicDataCache
TensorView* getLargestOutputTensor(
    Fusion* fusion,
    HeuristicDataCache* data_cache);

// Get reference loop domain with proper reordering
std::vector<IterDomain*> getReferenceLoop(
    TensorView* largest_out,
    bool has_reshapes,
    HeuristicDataCache* data_cache);

// Get element counts for each dimension in the reference loop
std::pair<std::vector<int64_t>, int64_t> getElementCounts(
    const std::vector<IterDomain*>& ref_loop,
    SchedulerRuntimeInfo& runtime_info);

// Get logical reorder map for the largest output tensor
std::unordered_map<int64_t, int64_t> getLogicalReorderMap(
    TensorView* largest_out,
    bool has_reshapes,
    HeuristicDataCache* data_cache);

// Structure to hold common runtime properties for pointwise schedulers
struct FusionRuntimeProperties {
  TensorView* largest_out = nullptr;
  int64_t device_multiprocessor_count = 0;
  bool has_reshapes = false;
  std::vector<IterDomain*> ref_loop;
  std::vector<int64_t> elem_counts;
  int64_t n_elems = 0;
  std::vector<TensorView*> vectorizable_inputs_outputs;
  int64_t max_dtype_size_bit_for_vectorization = 0;
  int64_t min_dtype_size_bit_for_vectorization = 0;
  PrimDataType index_type = PrimDataType::Int;
};

// Get common runtime properties for pointwise schedulers
// Returns std::nullopt for zero-dimensional or zero-size tensors
std::optional<FusionRuntimeProperties> getFusionRuntimeProperties(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache);

// Structure to hold break point calculation results (only break point info)
struct BreakPointInfo {
  int break_point = 0;
  bool flip_grid_binding = false;
  int64_t right_elem_count = 0;
  bool is_outer_broadcast_dominated = false;
};

// Calculate optimal break point for 2D scheduling
// Returns just the break point information without thread/grid dimensions
BreakPointInfo getBreakPoint(
    Fusion* fusion,
    const FusionRuntimeProperties& prop,
    HeuristicDataCache* data_cache,
    bool is_tma,
    int64_t max_vect_factor = 1,
    int64_t kThreadX = 128);

// Structure to hold complete block and grid configuration
struct BlockGridConfig {
  int break_point = 0;
  bool flip_grid_binding = false;
  int64_t right_elem_count = 0;
  int64_t bdimx = 0;
  int64_t bdimy = 1;
  int64_t gdim_left = 1;
  int64_t gdim_right = 1;
  bool is_outer_broadcast_dominated = false;
};

// Calculate block and grid dimensions based on break point information
// Returns complete block/grid configuration for pointwise schedulers
BlockGridConfig getBlockGridConfig(
    const FusionRuntimeProperties& prop,
    const BreakPointInfo& bp_info,
    int64_t max_vect_factor,
    int64_t kThreadX);

// Structure to hold results from common pointwise scheduling setup
struct CommonScheduleInfo {
  std::vector<std::pair<TensorView*, int64_t>> cached_inputs;
  std::vector<std::pair<TensorView*, int64_t>> cached_outputs;
  std::vector<TensorView*> input_tvs;
  std::vector<TensorView*> output_tvs;
  TensorView* reference_tv = nullptr;
  int64_t lhs_i = -1; // Position of lhs merged dimension (-1 for 1D scheduler)
  int64_t rhs_i = -1; // Position of rhs merged dimension
};

// Common scheduling setup for pointwise schedulers
// Performs: memory space clearing, input/output caching, cache policy
// refinement, reference tensor setup, and broadcast ordering Returns
// std::nullopt for zero-dimensional tensors
std::optional<CommonScheduleInfo> commonPointwiseSchedule(
    Fusion* fusion,
    int64_t break_point);

} // namespace pointwise_utils
} // namespace nvfuser
