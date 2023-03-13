// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/ivalue.h>

#include <fusion.h>
#include <scheduler/reduction_heuristic.h>

namespace nvfuser {

class SchedulerRuntimeInfo;
class HeuristicSummary;

TORCH_CUDA_CU_API std::shared_ptr<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache = nullptr);

TORCH_CUDA_CU_API std::shared_ptr<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache = nullptr);

TORCH_CUDA_CU_API void scheduleReduction(
    Fusion* fusion,
    const ReductionParams& rparams);
} // namespace nvfuser
