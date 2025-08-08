// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <scheduler/heuristic.h>
#include <scheduler/registry.h>
#include <scheduler/scheduler_types.h>
#include <visibility.h>

namespace nvfuser {

// Heuristic parameters for CUTLASS scheduling
class CutlassParams : public HeuristicParams {
 public:
  // CUTLASS kernel configuration parameters
  int tile_m = 128;  // M dimension tile size
  int tile_n = 128;  // N dimension tile size
  int tile_k = 32;   // K dimension tile size
  
  // Warp shape configuration
  int warp_m = 64;
  int warp_n = 64;
  int warp_k = 16;
  
  // Thread block shape
  int num_warps_m = 2;
  int num_warps_n = 2;
  int num_warps_k = 1;
  
  // Epilogue configuration
  bool has_epilogue_fusion = false;
  
  // Precision configuration
  bool use_fp8 = false;
  bool use_nvfp4 = false;
  
  CutlassParams() : HeuristicParams(SchedulerType::Cutlass) {}
  
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