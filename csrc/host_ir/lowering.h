// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <vector>

#include "fusion_segmenter.h"
#include "host_ir/container.h"
#include "runtime/executor_abstract.h"

namespace nvfuser {

// Consider moving this to namespace hir.
std::unique_ptr<hir::HostIrContainer> lowerSegmentedFusionToHostIr(
    const SegmentedFusion& segmented_fusion,
    // TODO(#4927): Launch parameters should be passed in at runtime, not
    // compile time.  They can change according to input sizes.
    const std::vector<LaunchParams>& launch_params_per_segment,
    std::vector<std::unique_ptr<ExecutorAbstract>>& executors);

} // namespace nvfuser
