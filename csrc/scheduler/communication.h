// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <scheduler/heuristic.h>
#include <scheduler/registry.h>

namespace nvfuser {

class Fusion;
class SchedulerRuntimeInfo;
class HeuristicDataCache;

//! Communication scheduler represents the case where scheduler will
//!  not do any scheduling operations and forward the un-scheduled
//!  fusion directly to code generation and kernel compilation.
//!
//! Typical use case of this scheduler is to handle edge cases
//!  such as where all tensors are size-1 or size-0.

class CommunicationScheduler : public SchedulerEntry {
 public:
  //! Check if the no-op heuristics apply in given fusion
  bool canScheduleCompileTime(Fusion* fusion) override;

  bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache = nullptr) override;

  std::unique_ptr<HeuristicParams> computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache) override;

  void schedule(Fusion* fusion, const HeuristicParams* params) override;

  constexpr static SchedulerType schedulerType() {
    return SchedulerType::Communication;
  }
};

} // namespace nvfuser
