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
class Fusion;
class TensorView;
class IterDomain;
class ViewOp;

namespace scheduler_tools {

// Create the loop domain of given tensors as specified by the
// reference. The new loop domain is connected to the existing IDs of
// each tensor by replaying exprs found in the Exact ValGraph.
//
// If update_loop_domain_only is true, uses the current loop domain as
// the starting domain and updates it to make it look like the given
// reference loop domain.
void scheduleLoopDomainsLike(
    const std::vector<TensorView*>& tvs,
    const std::vector<IterDomain*>& ref_loop_dom,
    bool update_loop_domain_only = false);

// Replay a transform expr on the loop domain of each of the given
// tensors. If the input of the transform is exact mapped with the loop
// domain, the transform is replayed as a forward op. If the output
// is exact mapped with the loop domain, it's replayed as a backward
// op. The loop domain of each tensor is updated with the replayed
// transform expr. If it's replayed as a forward op, the outputs
// replace the inputs in the loop domain. If it's replayed as a
// backward op, the inputs replace the outputs in the loop domain. The
// new IDs are inserted at the outermost position of the input IDs.
//
// For example, suppose a fusion has:
//
// t0 = makeSymbolicTensor(1);
// t1 = sin(t0);
// t2 = t1[1:]); // slice
//
// In this case, t2 has a resize op with a left expansion factor of
// -1. This function can be used to propagate this resize onto t1 as
// follows:
//
// scheduleLoopDomainsBy({t1}, t2->axis(0)->definition());
//
// Then the t1 domain should look like:
//
// Logical: i0
//  resize i0 by {-1, 0} -> i1
// Loop: i1
//
// Now that the loop domain of t1 and t2 are exact mapped, it's also
// possible to inline t1 into the innermost position of t2. See
// LoopDomainSchedulingTest.ScheduleLoopDomainsBy1 for more examples.
void scheduleLoopDomainsBy(
    const std::vector<TensorView*>& tvs,
    Expr* transform);

void cancelReshapeTransforms(Fusion* fusion);

} // namespace scheduler_tools
} // namespace nvfuser
