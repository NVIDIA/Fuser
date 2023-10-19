// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <unordered_map>

#include <fusion.h>
#include <ir/interface_nodes.h>

namespace nvfuser::optimization {

// Maps aliases (e.g. the output of a View) to their direct sources (e.g. the
// input of the same View). Consider path compression, a common optimization
// used in disjoint-set data structure, so it's easy to figure out the root of
// an alias.
using AliasAnalysisResult =
    std::unordered_map<const TensorView*, const TensorView*>;

// Finds aliases of the fusion inputs.
AliasAnalysisResult findAliases(const Fusion& fusion);

} // namespace nvfuser::optimization
