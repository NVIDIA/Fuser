// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/normalization.h>
#include <scheduler/registry_utils.h>

#include <c10/util/irange.h>
#include <disjoint_set.h>
#include <executor_utils.h>
#include <expr_evaluator.h>
#include <grouped_reduction.h>
#include <instrumentation.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <root_domain_map.h>
#include <scheduler/debug_utils.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <tensor_metadata.h>
#include <limits>

#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {

using ReductionType = reduction_scheduler_utils::ReductionType;

std::shared_ptr<ReductionParams> getPersistentHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("getPersistentHeuristicsFromIValue");
  SchedulerRuntimeInfo runtime_info(fusion, runtime_inputs);
  auto reduction_type = reduction_scheduler_utils::getReductionType(fusion);
  switch (reduction_type) {
    case ReductionType::Inner:
      return InnerPersistentKernelScheduler::getPersistentHeuristic(
          fusion, runtime_info, data_cache);
    case ReductionType::Outer:
      return OuterPersistentKernelScheduler::getPersistentHeuristic(
          fusion, runtime_info, data_cache);
    case ReductionType::InnerOuter:
      return InnerOuterPersistentKernelScheduler::getPersistentHeuristic(
          fusion, runtime_info, data_cache);
    case ReductionType::None:
      NVF_ERROR(false, "No reduction detected.");
      return nullptr;
    default:
      NVF_ERROR(false, "Reduction type not defined!");
      return nullptr;
  }
}

} // namespace nvfuser
