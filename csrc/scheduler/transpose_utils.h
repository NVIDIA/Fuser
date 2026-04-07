// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <memory>
#include <vector>

#include "scheduler/transpose_heuristic.h"

namespace nvfuser {
namespace transpose {
namespace utils {

bool hasSmallTransposeDimensions(
    const std::unique_ptr<TransposeParams>& params);

// See note [Supporting small transpose dimensions] in transpose_utils.cpp
void maybeBuildVirtualInnerDims(
    TransposeParams* tparams,
    int64_t device_multiprocessor_count,
    int64_t n_elems,
    const std::vector<int64_t>& shape_in_ref1,
    int64_t inner_most1,
    int64_t inner_most2);

} // namespace utils
} // namespace transpose
} // namespace nvfuser
