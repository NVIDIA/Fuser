// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <val_graph.h>

namespace nvfuser {

// Choose an IterDomain from the ValGroup that is amenable to transforms.
// Specifically we prefer, in descending order:
//   1. Iteration domains
//   2. Broadcast domains
//   3. Reduction domains
IterDomain* representativeId(const ValGroup& vg);

// Given a ValGraph and two ValGroups g0 and g1 in this graph, if there is
// already a merge of g0 with g1 in graph, return the output ValGroup of that
// merge. Otherwise create an new ValGroup that is a merge of g0 and g1 in
// graph, and a new ExprGroup that is the definition of the new ValGroup.
// After the merge, g0 and g1 will remain valid pointers.
ValGroup merge(ValGraph* graph, const ValGroup& g0, const ValGroup& g1);

// Given a ValGraph and a ValGroup g in this graph, if there is already a split
// of g in graph with the same factor, then return the output ValGroups of that
// split. Otherwise create two new ValGroups that are a split of g in
// graph, and a new ExprGroup that is the definition of the new ValGroups.
// After the split, g will remain valid pointers.
std::pair<ValGroup, ValGroup> split(
    ValGraph* graph,
    const ValGroup& g,
    Val* factor,
    bool inner_split = true);
std::pair<ValGroup, ValGroup> split(
    ValGraph* graph,
    const ValGroup& g,
    int64_t factor,
    bool inner_split = true);

// Given a ValGraph and two ValGroups g0 and g1 in this graph, if there is
// already a swizzle of g0 with g1 in graph, return the output ValGroups of that
// swizzle. Otherwise create two new ValGroups that are a swizzle of g0 and g1
// in graph, and a new ExprGroup that is the definition of the new ValGroups.
std::pair<ValGroup, ValGroup> swizzle(
    ValGraph* graph,
    SwizzleType swizzle_type,
    const ValGroup& g0,
    const ValGroup& g1);

} // namespace nvfuser
