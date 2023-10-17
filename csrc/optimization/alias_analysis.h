// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <unordered_map>

namespace nvfuser::optimization {

using AliasAnalysisResult =
    std::unordered_map<const TensorView*, const TensorView*>;

AliasAnalysisResult findAliases(const Fusion& fusion);

} // namespace nvfuser::optimization
