// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>

namespace nvfuser {

class SchedulerRuntimeInfo;
class HeuristicSummary;
class MatmulParams;

//! An implementation of functionality that will prepare heuristics for fusion
//!  that represents matmul. May return empty object if any of conditions are
//!  not met.
TORCH_CUDA_CU_API std::shared_ptr<MatmulParams> getMatmulHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache = nullptr);

//! An implementation of compile time checks. Returns messasge if given fusion
//!  does not represent matmul, otherwise an empty string is returned.
TORCH_CUDA_CU_API std::string getMatmulCompileTimeRejectReason(Fusion* fusion);

//! An implementation of runtime time checks. Returns messasge if given fusion
//!  does not represent matmul, otherwise an empty string is returned.
TORCH_CUDA_CU_API std::string getMatmulRunTimeRejectReason(
    Fusion* fusion,
    HeuristicSummary* data_cache,
    SchedulerRuntimeInfo& runtime_info);

} // namespace nvfuser
