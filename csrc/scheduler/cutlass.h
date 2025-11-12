// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <mma_type.h>
#include <scheduler/heuristic.h>
#include <scheduler/registry.h>
#include <scheduler/scheduler_types.h>
#include <visibility.h>

namespace nvfuser {

// Heuristic parameters for CUTLASS scheduling
class NVF_API CutlassParams : public HeuristicParams {
 public:
  // If mma_tile.m is 256, we will do 2SM
  GemmTile mma_tile = {256, 256, 256};

  // This is the tile of output elements handled in the epilogue for each SM,
  // but it is not the "epilogue tile" used for pipelining stores. If this is
  // smaller than mma_tile (i.e. mma_tile.m==256 and per_sm_tile.m==128) then we
  // are doing 2sm mma.
  GemmTile per_sm_tile = {128, 256, 256};

  // Shape of the cluster in CTAs, order is M, N, 1
  GemmTile cluster_shape = {4, 4, 1};

  CutlassParams() : HeuristicParams(SchedulerType::Cutlass) {
    // This is not used for code generation, but FusionKernelRuntime expects it
    // to be set.
    cparams.index_type = DataType::Int;
  }

  std::string toString() const override;

  size_t hash() const override;

  bool sameAs(const HeuristicParams* other) const override;

  std::unique_ptr<HeuristicParams> clone() const override;
};

class CutlassScheduler : public SchedulerEntry {
 public:
  bool canScheduleCompileTime(Fusion* fusion) override;

  bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache = nullptr) override;

  //! Compute heuristics for the fusion using nvMatmulHeuristics, if available.
  //! Otherwise a static config is used.
  std::unique_ptr<HeuristicParams> computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache = nullptr) override;

  //! For the Cutlass scheduler, schedule() actually does nothing
  void schedule(Fusion* fusion, const HeuristicParams* params) override;

  constexpr static SchedulerType schedulerType() {
    return SchedulerType::Cutlass;
  }
};

} // namespace nvfuser
