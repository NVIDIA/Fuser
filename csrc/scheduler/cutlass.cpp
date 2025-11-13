// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cutlass/gemm.h>
#include <device_lower/utils.h>
#include <exceptions.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <scheduler/cutlass.h>
#include <scheduler/debug_utils.h>
#include <scheduler/nvmmh.h>
#include <scheduler/runtime_info.h>
#include <scheduler/utils.h>

#include <ATen/cuda/CUDAContextLight.h>

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

  const cudaDeviceProp* device_prop = at::cuda::getCurrentDeviceProperties();
  if (device_prop->major != 10 ||
      !(device_prop->minor == 0 || device_prop->minor == 3)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Cutlass scheduler only supports Blackwell (cc 10.0) and above but "
        "current device is cc ",
        device_prop->major,
        ".",
        device_prop->minor);
    return false;
  }

  const std::string reject_reason =
      cutlass_codegen::getGemmRejectReason(fusion);
  if (reject_reason.empty()) {
    return true;
  }
  scheduler_debug_utils::canScheduleRejectReason(
      schedulerType(), reject_reason);
  return false;
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

  // If nvMatmulHeuristics is not available, this call will leave params
  // default-initialized
  fillNvMatmulHeuristicsParams(params.get(), fusion, runtime_info);

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
  // like split, reorder, etc.
}

} // namespace nvfuser
