// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <id_model/id_model.h>

namespace nvfuser {

// Get the domains to predicate for a given tensor used as a consumer
// of a given expr.
std::vector<IterDomain*> getPredicateDomains(
    TensorView* consumer_tv,
    const Expr* expr);

// Get a replace map for predicate indexing of a given tensor appearing
// in a given loop-nest.
//
// The unswitched_loop parameter is an optional ForLoop that is used
// when this predicate is for an unswitched, unrolled or vectorized
// loop.
std::unordered_map<Val*, Val*> getPredicateIndexReplacementMap(
    TensorView* tv,
    const std::vector<ForLoop*>& for_loops,
    const std::unordered_map<ValGroup, Val*>& index_map,
    const ValGraph& traversal_graph,
    const IdModel& id_model,
    bool is_start_predicate,
    ForLoop* unswitched_loop = nullptr);

} // namespace nvfuser
