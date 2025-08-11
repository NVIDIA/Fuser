// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <exceptions.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <scheduler/cutlass.h>
#include <scheduler/runtime_info.h>
#include <scheduler/utils.h>

namespace nvfuser {

// CutlassParams implementation

std::string CutlassParams::toString() const {
  std::stringstream ss;
  ss << "CutlassParams (" << scheduler_type << ")\n";
  ss << "  CTA Tile: " << cta_tile.m << "x" << cta_tile.n << "x" << cta_tile.k
     << "\n";
  ss << "  Warp Tile: " << warp_tile.m << "x" << warp_tile.n << "x"
     << warp_tile.k << "\n";
  return ss.str();
}

size_t CutlassParams::hash() const {
  size_t h = 0;
  // TODO: better hash function
  return h;
}

bool CutlassParams::sameAs(const HeuristicParams* other) const {
  if (!other->isStrictlyA<CutlassParams>()) {
    return false;
  }
  const auto* other_cutlass = other->as<CutlassParams>();
  return cta_tile == other_cutlass->cta_tile &&
      warp_tile == other_cutlass->warp_tile &&
      cluster_dims == other_cutlass->cluster_dims &&
      HeuristicParams::sameAs(other);
}

std::unique_ptr<HeuristicParams> CutlassParams::clone() const {
  return std::make_unique<CutlassParams>(*this);
}

// CutlassScheduler implementation

bool CutlassScheduler::canScheduleCompileTime(Fusion* fusion) {
  FUSER_PERF_SCOPE("CutlassScheduler::canScheduleCompileTime");

  // Check if fusion has a supported matmul pattern
  if (!hasSupportedMatmulPattern(fusion)) {
    return false;
  }

  // Check if epilogue is supported
  if (!hasSupportedEpilogue(fusion)) {
    return false;
  }

  return true;
}

bool CutlassScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("CutlassScheduler::canScheduleRunTime");

  // For now, all runtime checks are deferred to compile time checks
  // In the future, we may want to check tensor sizes, alignment, etc.

  return true;
}

std::unique_ptr<HeuristicParams> CutlassScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("CutlassScheduler::computeHeuristics");

  auto params = std::make_unique<CutlassParams>();

  // For now, use default parameters
  // TODO: Implement actual heuristics based on problem size, GPU arch, etc.
  // Once libheuristics is available via pycutlass wheel, integrate it here

  // Set launch parameters
  // TODO: set dimensions based on other heuristic params
  int64_t block_dim_x = 384;
  params->lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL, // gdimx
      LaunchParams::UNINITIALIZED_VAL, // gdimy
      LaunchParams::UNINITIALIZED_VAL, // gdimz
      block_dim_x, // bdimx
      1, // bdimy
      1); // bdimz

  return params;
}

void CutlassScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  FUSER_PERF_SCOPE("CutlassScheduler::schedule");

  NVF_CHECK(
      params->isA<CutlassParams>(), "CutlassScheduler expects CutlassParams");

  // CUTLASS scheduling doesn't involve traditional scheduling operations
  // like split, reorder, etc. The scheduler type is already determined
  // by the time this method is called.

  // TODO: We may want to add metadata to the fusion or specific ops
  // to guide CUTLASS code generation
}

bool CutlassScheduler::hasSupportedMatmulPattern(Fusion* fusion) {
  // Only accept ScaledMmaOp for JIT CUTLASS kernels
  for (auto expr : fusion->exprs()) {
    if (expr->isA<ScaledMmaOp>()) {
      return true;
    }
  }
  return false;
}

bool CutlassScheduler::hasSupportedEpilogue(Fusion* fusion) {
  // For now, we support all epilogues that don't involve complex reductions
  // or unsupported operations

  auto matmul_output = findMatmulOutput(fusion);
  if (!matmul_output) {
    return false;
  }

  // Check all uses of the matmul output
  for (auto use : matmul_output->uses()) {
    if (use->isA<ReductionOp>()) {
      // Complex reductions not supported yet
      return false;
    }
    // TODO: Add more checks for unsupported operations
  }

  return true;
}

TensorView* CutlassScheduler::findMatmulOutput(Fusion* fusion) {
  for (auto expr : fusion->exprs()) {
    if (expr->isA<ScaledMmaOp>()) {
      return expr->output(0)->as<TensorView>();
    }
  }
  return nullptr;
}

} // namespace nvfuser
