// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <unordered_set>
#include <vector>

#include <ir/interface_nodes.h>
#include <scheduler/utils.h>

namespace nvfuser {

// Propagates the given device/stream ids from ref to target.
void shardLoopLike(
    const TensorView* ref,
    TensorView* tv,
    const std::unordered_set<ParallelType>& selected_parallel_types,
    PropagateDirection direction);

// Shards all tensors in tvs like reference.
// Accepts a set of parallel types to shard on.
// If empty, all DID parallel types are used.
void shardAllLike(
    TensorView* ref,
    const std::vector<TensorView*>& tvs,
    const std::unordered_set<ParallelType>& parallel_types);

// Shards all TVs between from and to AND between TVs created inside a fusion
// and to. This is required for (1) expressions like rng_uniform that create a
// TV inside a fusion that is not between a path from user visible TVs. (2)
// multi-output expressions may have output tensors that are not along a path to
// the fusion output which would not be reachable otherwise. (2) sharding
// propagation checks all TVs in the fusion are assigned a device mesh
// regardless if they are reachable. To keep the checks simple, we require all
// TVs are assigned a mesh if they exist in the fusion.
void shardBetween(
    const std::vector<TensorView*>& from,
    const std::vector<TensorView*>& to,
    TensorView* ref);

// Same as above but using the outputs of the from and to expressions
// to form the from and to TVs.
void shardBetween(
    const std::vector<Expr*>& from,
    const std::vector<Expr*>& to,
    TensorView* ref);

} // namespace nvfuser
