// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/ivalue.h>
#include <exceptions.h>
#include <fusion.h>
#include <scheduler/reduction_heuristic.h>
#include <scheduler/registry.h>
#include <scheduler/utils.h>

// TODO: If caching inputs would require persistence we are sending it to the
// persistent kerenl scheduler. This isn't necessary if the only persistent
// buffers are inputs as we could re-read them from global memory. Need to
// consider if this is worth implementing.

namespace nvfuser {

class SchedulerRuntimeInfo;
class HeuristicSummary;

class InnerOuterPersistentKernelScheduler : public SchedulerEntry {
 public:
  //! register file size allowed for persistent buffers in
  //! innerOuterPersistentHeuristic. May cause register spills but still
  //! improves the overall performance. Here 54 = (1-overhead/255) * 64, reduce
  //! to 48 if not vectorized.
  constexpr static int64_t register_file_size_combined = (int64_t)4 * 54 * 1024;
  constexpr static int64_t register_file_size_combined_nonvectorized =
      (int64_t)4 * 48 * 1024;

  // max threads per block for innerOuterPersistentHeuristic.
  // innerOuterPersistentHeuristic creates additional
  // persistent tensors to store intermediate outer reduction results. It also
  // have both inner and outer reductions, the register pressure is very high.
  // Limit the max threads per block to 256, allows each thread to use 255
  // registers. If not vectorized, more gmem access ops is required, increase to
  // 512 for higher occupancy to hide gmem access latency.
  constexpr static int64_t max_threads_per_block_combined = 256l;
  constexpr static int64_t max_threads_per_block_combined_nonvectorized = 512l;

  explicit InnerOuterPersistentKernelScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  void schedule(Fusion* fusion) override;

  static bool canScheduleCompileTime(Fusion* fusion);

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  constexpr static ScheduleHeuristic heuristicType() {
    return ScheduleHeuristic::InnerOuterPersistent;
  }

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);
};

std::shared_ptr<ReductionParams> getInnerOuterPersistentHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache = nullptr);

std::shared_ptr<ReductionParams> getInnerOuterPersistentHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache = nullptr);

void scheduleInnerOuterPersistentKernel(
    Fusion* fusion,
    const ReductionParams& rparams);

} // namespace nvfuser
