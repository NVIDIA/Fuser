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
class HeuristicDataCache;

class InnerOuterPersistentKernelScheduler : public SchedulerEntry {
 public:
  // This scheduler has very high register pressure due to extra registers to
  // store intermediate outer reduction results. So prefer to allow 255
  // registers per thread and then the max threads per block is 256.
  constexpr static int64_t threads_per_block_min = 128l;
  constexpr static int64_t threads_per_block_max = 256l;

  void schedule(Fusion* fusion, const HeuristicParams* params) override;

  bool canScheduleCompileTime(Fusion* fusion) override;

  bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache = nullptr) override;

  constexpr static SchedulerType schedulerType() {
    return SchedulerType::InnerOuterPersistent;
  }

  std::unique_ptr<HeuristicParams> computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache) override;
};

} // namespace nvfuser
