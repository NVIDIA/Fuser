// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <unordered_set>

#include "ir/interface_nodes.h"
#include "multidevice/utils.h"
#include "visibility.h"

namespace nvfuser {

// Returns whether an Expr embeds multi-device resharding
NVF_API bool isResharding(const Expr* expr);

// Returns whether two tensors have different shardings. Expect a
// producer/consumer relationship between the arguments.
bool haveDifferentShardings(
    const TensorView* producer,
    DomainType producer_domain_type,
    const TensorView* consumer,
    DomainType consumer_domain_type,
    const std::unordered_set<ParallelType>& parallel_types);

// Same as the above but checks loop domains for both producer and consumer.
bool haveDifferentShardings(
    const TensorView* producer,
    const TensorView* consumer,
    const std::unordered_set<ParallelType>& parallel_types);

} // namespace nvfuser
