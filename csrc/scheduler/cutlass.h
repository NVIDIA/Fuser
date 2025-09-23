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
  GemmTile mma_tile = {256, 256, 256};
  // If this is smaller than mma_tile, then we are doing 2sm mma
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

// CUTLASS scheduler entry point
class CutlassScheduler : public SchedulerEntry {
 public:
  // Check if the fusion can be scheduled at compile time
  bool canScheduleCompileTime(Fusion* fusion) override;

  // Check if the fusion can be scheduled at runtime
  bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache = nullptr) override;

  // Compute heuristics for the fusion
  std::unique_ptr<HeuristicParams> computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache = nullptr) override;

  // Schedule the fusion
  void schedule(Fusion* fusion, const HeuristicParams* params) override;

  constexpr static SchedulerType schedulerType() {
    return SchedulerType::Cutlass;
  }

 private:
  // Check if the fusion contains supported matmul patterns
  bool hasSupportedMatmulPattern(Fusion* fusion);

  // Check if the epilogue is supported by CUTLASS
  bool hasSupportedEpilogue(Fusion* fusion);

  // Find the matmul operation in the fusion
  TensorView* findMatmulOutput(Fusion* fusion);
};

} // namespace nvfuser
