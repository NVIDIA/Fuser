// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <unordered_set>

#include "ir/interface_nodes.h"
#include "type.h"

namespace nvfuser {

// Return true if the TensorView is contiguous. This function is more
// permissive than torch.Tensor.is_contiguous because it allows expanded
// broadcasts.
bool isTvContiguous(const TensorView* tv);

// Find the producing logical id of the given allocation id traversing
// through device splits. For unsharded allocation_id, logical_id is the same as
// allocation_id.
IterDomain* projectShardedAllocationToLogical(
    TensorView* tv,
    IterDomain* allocation_id);

// Finds the allocated id corresponding to the given logical id
// traversing through device splits. For e.g.: `i0` -> `DIDx(d), i0/d` will
// return `i0/d`. For unsharded logical_id, allocation_id is the same as
// logical_id.
IterDomain* projectLogicalToShardedAllocation(
    TensorView* tv,
    IterDomain* logical_id);

// Propagate sharding for the given parallel types from loop domain to
// allocation domain, refining contiguity as needed so allocation aliases the
// original storage layout.
void shardAllocationAsLoop(
    TensorView* tv,
    const std::unordered_set<ParallelType>& parallel_types);

} // namespace nvfuser
