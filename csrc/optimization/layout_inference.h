// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <unordered_map>

#include <fusion.h>

namespace nvfuser {

using MemoryFormat = std::vector<int64_t>;

// Propagate memory format from input to the entire fusion. It does NOT modify any fusion IR, but instead stores the propagated memory format as an unordered_map from TensorView to permutation.
//
// See details in Note [ Memory Format Propagation ]
std::unordered_map<const TensorView*, MemoryFormat> inferenceMemoryFormat(
    Fusion* fusion);

} // namespace nvfuser
