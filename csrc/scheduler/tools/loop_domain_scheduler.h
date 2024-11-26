// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <vector>

namespace nvfuser {

class Expr;
class TensorView;
class IterDomain;

namespace scheduler_tools {

// Create the loop domain of given tensors as specified by the
// reference. The new loop domain is connected to the existing IDs of
// each tensor by replaying exprs found in the Exact ValGraph.
void scheduleLoopDomainsLike(
    const std::vector<TensorView*>& tvs,
    const std::vector<IterDomain*>& ref_loop_dom);

// Replay a transform expr on the loop domain of each of the given
// tensors. If the input of the transform matches with the loop
// domain, the transform is replayed as a forward op. If the output
// matches with the loop domain, it's replayed as a backward op.
void scheduleLoopDomainsBy(
    const std::vector<TensorView*>& tvs,
    Expr* transform);

} // namespace scheduler_tools
} // namespace nvfuser
