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

// Given a ValGraph and two ValGroups g0 and g1 in this graph, if there is
// already a merge of g0 with g1 in graph, return the output ValGroup of that
// merge. Otherwise create an new ValGroup that is a merge of g0 and g1 in
// graph, and a new ExprGroup that is the definition of the new ValGroup.
ValGroup merge(ValGraph* graph, const ValGroup& g0, const ValGroup& g1);

} // namespace nvfuser
