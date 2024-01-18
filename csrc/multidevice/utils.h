// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/interface_nodes.h>

namespace nvfuser {

// Returns whether a TensorView has its first non-reduction axis parallelized
// on Didx
// Checks that the other non-reduction axis are not parallelized on Didx
bool isSharded(TensorView*);

// Returns the subset of tvs which elements have the same multi-device sharding
// as ref
std::unordered_set<TensorView*> haveDifferentSharding(
    TensorView* ref,
    std::unordered_set<TensorView*> tvs);

// Returns whether an Expr embbeds multi-device resharding
bool isResharding(Expr* expr);

} // namespace nvfuser
