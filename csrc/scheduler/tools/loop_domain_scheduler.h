// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <bfs.h>

#include <vector>

namespace nvfuser {

class Expr;
class Fusion;
class TensorView;
class IterDomain;
class ReshapeOp;

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
// tensors. If the replay direction is specified, the expr is replayed
// as specified. Otherwise, if the input of the transform is exact mapped with
// the loop domain, the transform is replayed as a forward op. If the output is
// exact mapped with the loop domain, it's replayed as a backward op. The loop
// domain of each tensor is updated with the replayed transform expr. If it's
// replayed as a forward op, the outputs replace the inputs in the loop domain.
// If it's replayed as a backward op, the inputs replace the outputs in the loop
// domain. The new IDs are inserted at the outermost position of the input IDs.
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
    Expr* transform,
    Direction replay_dir = Direction::Undefined);

// For each of immediate and indirect consumer tensors of from_tv,
// schedule its loop domain such that reshape transforms appearing
// between the tensor and from_tv are cancelled. For example, suppose
// a fusion has:
//
// t0 = makeSymbolicTensor(3); // [i0, i1, i2]
// t1 = permute(t0, {1, 0, 2}); // [i1, i0, i2]
// t2 = reshape(t1, {i1, i0*i2}); // [i1, i0*i2]
// t3 = sin(t2) // [i1, i0*i2]
//
// In this case, cancelReshapeInLoopDomains(t0) would affect t2 and t3
// as follows:
//
// t2:
//  root: [i1, i0*i2] (unchanged)
//  logical: [i1, i0*i2] (unchanged)
//  loop: [i1, i0, i2]
//
// t3:
//  logical: [i1, i0*i2] (unchanged)
//  loop: [i1, i0, i2]
//
// t1 would not be changed at all as there's no reshape between t0 and
// t1.
//
// This scheduling could help optimize memory accesses to
// fusion inputs. In the above case, we could then reorder the loop
// domains of t1, t2 and t3 as [i0, i1, i2], i.e., the same ordering
// as t0, which could minimize strided accesses.
//
// This scheduling is not always feasible. Specifically, if a reshape
// output iter domain is resized, the loop domain needs to keep using
// the reshape output iter domain. Similarly, if a rehape output iter
// domain is reduced, the reshape is currently not cancelled. This is
// because if a reshape has a split and only one of the split output
// iter domain is reduced, the split needs to remain. If a reshape
// only consists of merge transforms, cancellation should be possible,
// but that is not currently supported.
//
// When the skip_innermost_id flag is true, any reshape that involves
// innermost logical ID is not canceled even when it's technically
// possible. This is a WAR for the resize scheduler.
void cancelReshapeInLoopDomains(
    TensorView* from_tv,
    bool skip_innermost_id = false);

} // namespace scheduler_tools
} // namespace nvfuser
