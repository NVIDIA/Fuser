// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <scheduler/normalization_inner_tma.h>

#include <instrumentation.h>
#include <scheduler/debug_utils.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>
#include <scheduler/utils.h>

#include <ATen/cuda/CUDAContext.h>

#include <memory>

namespace nvfuser {
namespace inner_persistent {
namespace tma {

using PersistentKernelProperties =
    normalization_scheduler_utils::PersistentKernelProperties;

namespace {

// TODO: Implement TMA-specific heuristics for inner persistent scheduler
// This is a placeholder implementation that can be enhanced with TMA
// optimizations

} // namespace

std::unique_ptr<ReductionParams> getInnerPersistentHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);

  // TODO: Implement TMA-specific heuristics
  // For now, return nullptr to indicate TMA is not yet supported
  // This will cause the main scheduler to fall back to non-TMA implementation
  return nullptr;
}

void scheduleInnerPersistent(Fusion* fusion, const ReductionParams* rparams) {
  FusionGuard fg(fusion);
  NVF_ERROR(
      rparams != nullptr &&
          rparams->scheduler_type ==
              InnerPersistentKernelScheduler::schedulerType(),
      "Incorrect parameters sent to inner persistent TMA scheduler");

  // TODO: Implement TMA-specific scheduling
  // For now, this is a placeholder
  normalization_scheduler_utils::schedulePersistentKernel(
      fusion, rparams, rparams->scheduler_type);
}

} // namespace tma
} // namespace inner_persistent
} // namespace nvfuser
