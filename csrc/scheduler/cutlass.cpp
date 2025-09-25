// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/utils.h>
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
  ss << "  MMA Tile: " << mma_tile.toVector() << "\n";
  ss << "  Per-SM MMA Tile: " << per_sm_tile.toVector() << "\n";
  ss << "  Cluster shape: " << cluster_shape.toVector() << "\n";
  return ss.str();
}

size_t CutlassParams::hash() const {
  size_t h = 0;
#define HASHTILE(t)            \
  hashCombine(h, (size_t)t.m); \
  hashCombine(h, (size_t)t.n); \
  hashCombine(h, (size_t)t.k);
  HASHTILE(mma_tile);
  HASHTILE(per_sm_tile);
  HASHTILE(cluster_shape);
#undef HASHTILE
  return h;
}

bool CutlassParams::sameAs(const HeuristicParams* other) const {
  if (!other->isStrictlyA<CutlassParams>()) {
    return false;
  }
  const auto* other_cutlass = other->as<CutlassParams>();
  return cparams == other->cparams && mma_tile == other_cutlass->mma_tile &&
      per_sm_tile == other_cutlass->per_sm_tile &&
      cluster_shape == other_cutlass->cluster_shape;
}

std::unique_ptr<HeuristicParams> CutlassParams::clone() const {
  return std::make_unique<CutlassParams>(*this);
}

// CutlassScheduler implementation

bool CutlassScheduler::canScheduleCompileTime(Fusion* fusion) {
  FUSER_PERF_SCOPE("CutlassScheduler::canScheduleCompileTime");

  // TODO: Enable this scheduler by default once we are confident in the pattern
  // matching and heuristic
  if (!isOptionEnabled(EnableOption::CutlassScheduler)) {
    return false;
  }

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

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << params->toString() << std::endl;
  }

  return params;
}

void CutlassScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  FUSER_PERF_SCOPE("CutlassScheduler::schedule");

  NVF_CHECK(
      params->isA<CutlassParams>(), "CutlassScheduler expects CutlassParams");

  // CUTLASS scheduling doesn't involve traditional scheduling operations
  // like split, reorder, etc. The scheduler type is already determined
  // by the time this method is called.

  // We may want to add metadata to the fusion or specific ops to guide CUTLASS
  // code generation
}

bool CutlassScheduler::hasSupportedMatmulPattern(Fusion* fusion) {
  // Only accept ScaledMmaOp for JIT CUTLASS kernels
  bool has_non_scaled_mma = false;
  int64_t num_scaled_mmas = 0;
  for (auto expr : fusion->exprs()) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }
    if (expr->isA<ScaledMmaOp>()) {
      num_scaled_mmas++;
    } else {
      has_non_scaled_mma = true;
    }
  }
  // TODO: accept fusions with epilogue
  return num_scaled_mmas == 1 && !has_non_scaled_mma;
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
