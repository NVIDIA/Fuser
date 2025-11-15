// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <unordered_set>

#include <ir/interface_nodes.h>
#include <scheduler/utils.h>

namespace nvfuser {

// Propagates the given device/stream ids from ref to target.
void shardLoopLike(
    const TensorView* ref,
    TensorView* tv,
    const std::unordered_set<ParallelType>& selected_parallel_types,
    PropagateDirection direction);

} // namespace nvfuser
