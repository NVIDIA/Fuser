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

// Maps aliases (e.g. fusion outputs) to their sources (e.g. fusion inputs).
using AliasAnalysisResult =
    std::unordered_map<const TensorView*, const TensorView*>;

// Finds aliases of the fusion inputs.
AliasAnalysisResult findAliases(const Fusion& fusion);

} // namespace nvfuser::optimization
