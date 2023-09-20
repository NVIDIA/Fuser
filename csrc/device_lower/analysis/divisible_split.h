// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>

#include <compute_at_map.h>
#include <fusion.h>
#include <ir/all_nodes.h>

namespace nvfuser {

// Looks through all transformations assocaited with view, or enforced divisible
// vectorization splits and gathers all splits that provably don't have a
// remainder, therefore the extents of the associated IterDomains do not require
// a ceilDiv expressions.
std::unordered_set<Split*> getAllDivisibleSplits(Fusion* fusion);

// Same as above but will use provided ComputeAtMap instead of building its own.
std::unordered_set<Split*> getAllDivisibleSplits(
    Fusion* fusion,
    const ComputeAtMap* ca_map);

} // namespace nvfuser
