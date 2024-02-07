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

std::unordered_map<const TensorView*, MemoryFormat> inferenceMemoryFormat(Fusion* fusion);

} // namespace nvfuser
