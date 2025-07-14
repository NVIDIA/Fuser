// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/all_nodes.h>
#include <scheduler/utils.h>

namespace nvfuser {
class SchedulerRuntimeInfo;
class HeuristicDataCache;

namespace inner_outer_utils {
// The roundup is due to the fact that the shared memory buffer is allocated
// as: ceilDiv(dim_size / vectorize_factor, threads_per_block).
// Let after_vect = dim_size / vectorize_factor;
// n_batch = ceilDiv(after_vect, threads_per_block);
// Then the shared memory buffer size is n_batch * vectorize_factor *
// threads_per_block * data_type_size. This function returns the maximum
// possible shared memory buffer size considering all possible block sizes.
int64_t roundUpSharedMemory(
    int64_t tv_buffer_size,
    int64_t data_type_size,
    int64_t vectorize_factor,
    int64_t threads_per_block_min,
    int64_t threads_per_block_max,
    int64_t threads_per_block_step);

// Size of buffers storing intermediate outer reduction results
// TODO: check if we can directly start with [buffer_size = 1]
int64_t partialOuterReductionBufferSizeBit(
    const std::vector<TensorView*>& reduction_tvs,
    SchedulerRuntimeInfo& runtime_info);

// Decide where to store persistent buffers.
// By default, they reside in registers.
// If register space runs low but there's ample shared memory,
// move one or more buffers to shared memory until the register space is
// sufficient.
struct PersistentBufferStorageParams {
  // representing buffers that are stored in shared memory, other buffers are
  // stored in registers.
  std::vector<TensorView*> smem_persistent_buffers;

  // Total number of bytes occupied by all persistent buffers stored in shared
  // memory.
  int64_t smem_buffer_size_bit = 0;

  // Shared memory buffers that are circular buffered.
  int64_t circular_buffered_smem_size_bit = 0;

  // Shared memory buffers that are not circular buffered, e.g. outer bcast tv
  // which doesn't have iter dim to circulate. It is shared for different
  // circular loops and smem usage is not related to circular buffer stages.
  int64_t non_circular_buffered_smem_size_bit = 0;

  // Total number of bits occupied by all persistent buffers stored in
  // registers.
  int64_t regs_buffer_size_bit = 0;

  // Additional shared memory usage per block that is not associated with
  // persistent buffers. This includes memory for driver overhead and workspace
  // for reductions.
  int64_t smem_overhead_bit = 0;

  // Flag indicating whether there are sufficient registers and shared memory
  // available to accommodate all persistent buffers as required for efficient
  // execution.
  bool has_enough_regs_and_smem = false;

  // Flag indicating whether the persistent buffers are recomputed using inputs.
  bool project_to_input = false;
};
PersistentBufferStorageParams getPersistentBufferStorageParams(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const std::vector<TensorView*>& reduction_tvs,
    const int64_t vectorize_factor,
    const int64_t threads_per_block_min,
    const int64_t threads_per_block_max,
    const bool is_warp_specialized);

// Prioritize keeping buffers used by outer broadcast tensors to shared memory
// because:
// (1) They are reused in every iteration of the outer loop, has lower IO.
// (2) Load occurs before the outer loop. Temporary register usage won't
//     increase register pressure since the loop is the high-pressure region.
std::vector<TensorView*> sortProjectableBufferInputs(
    const std::vector<TensorView*>& projectable_buffer_inputs,
    const std::vector<TensorView*>& outer_broadcast_tvs);

// Returns tvs that are persistent due to grouped reductions,
// such as the RMS tv in RMSNorm backward.

// Normally, an inner broadcast tv can be inlined with ops before and after
// inner reductions. But with grouped reductions, the loop over grouped domain
// is separated into two disjoint loops by the runtime function. To be
// accessible in both, certain tvs must be hoisted outside.

// The algorithm traces all paths from `cached_input` to fusion outputs:
// (1) tvs on paths that include a reduction go into `p_of_reductions`.
// (2) tvs on paths without reductions go into `c_of_reductions`.
// (3) tvs in both sets are considered persistent.
std::vector<TensorView*> getGroupedReductionPersistentTvs(
    Fusion* fusion,
    TensorView* cached_input,
    const std::vector<TensorView*>& reduction_tvs);

} // namespace inner_outer_utils
} // namespace nvfuser
