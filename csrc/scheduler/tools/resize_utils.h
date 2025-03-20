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

class Expr;
class TensorView;

namespace scheduler_tools {

bool isResizeBasedOp(Expr* expr);

bool hasResizeBasedOps(Fusion* fusion);

std::vector<Expr*> getResizeBasedOps(Fusion* fusion);

// For a given resize-based tensor op such as SliceOp and PadOp, make the loop
// domain of each dependent producer tensor exact-mapped by propagating
// the iter-domain ops of the output tensor of the given op. Note that
// fusion inputs are skipped as their loop domains don't matter.
void propagateResizeToInputs(Expr* resize_op);

// Given a topologically ordered list of resize-based tensor ops such
// as slice and pad, check if they can be propagated to fusion inputs
// exclusively without causing any visible side effect. For example,
// if a tensor is sliced and also is used to produce an output without
// the slicing, the slice is considered non exclusive as the slice
// input has the other visible consumer. Propagating the resize of the
// slice to the slice input is invalid since the output computed from
// the slice input depends on the full iteration space.
//
// For example, consider the following case:
//
// t0 = makeSymbolicTensor(1)
// fusion.addInput(t0)
// t1 = t0 + 1
// t2 = t1[1:10]
// t3 = t1 + 1
// fusion.addOutput(t2)
// fusion.addOutput(t3)
//
// In this case, propating the resize op of the slice would alter t1,
// which would in turn affect t3, which is a fusion output. Since the
// change would be visible due to the change of t3, this resize op is
// considered non-exclusive.
//
// Consider a slightly different case as shown below:
//
// t0 = makeSymbolicTensor(1)
// fusion.addInput(t0)
// t1 = t0[1:10]
// t2 = t0 + 1
// fusion.addOutput(t1)
// fusion.addOutput(t2)
//
// Note that the slice is directly done with the fusion input. Since
// we do not propagate resize ops to fusion inputs, this can be
// considered exclusive. However, this is also considered
// non-exclusive since the actual scheduling inserts a cache after t0,
// which can cause a visible side effect if the resize is propagated.
//
// Another non-exclusivness comes from dependent fusion outputs. For
// example, if a slice input depends on a fusion output, propagation
// would alter the fusion output. Consider a case like:
//
// t0 = makeSymbolicTensor(1)
// fusion.addInput(t0)
// t1 = t0 + 1
// t2 = t1[1:10] // slice
// fusion.addOutput(t1)
// fusion.addOutput(t2)
//
// If the resize op for the slice is propagated to t1, only the
// section of [1:10] would be computed. Since that would change a
// fusion output, the resize op is considered non-exclusive.
//
// When there's a chain of resize-based ops, for example:
//
// t0 = makeSymbolicTensor(1)
// fusion.addInput(t0)
// t1 = t0 + 1
// t2 = t1[1:10]
// t3 = t2[2:5]
// t4 = t1 + 1
// fusion.addOutput(t3)
// fusion.addOutput(t4)
//
// We do not consider the second slice as non-exclusive as
// long as the first slice is considered non-exclusive. This will be
// important when resolving the non-exclusiveness by replication.
//
// The function returns a map from tensors that are outputs to
// non-exclusive ops to ResizeExclusivityInfo. This map will be
// used to resolve the non-exclusiveness by replication.
struct ResizeExclusivityInfo {
  // Dependent tensors that should not be resized
  std::vector<TensorView*> non_exclusive_dep_tvs;
  // ID groups of resize input IDs
  ValGroups resized_ids;

  bool operator==(const ResizeExclusivityInfo& other) const {
    return non_exclusive_dep_tvs == other.non_exclusive_dep_tvs &&
        resized_ids == other.resized_ids;
  }

  bool operator!=(const ResizeExclusivityInfo& other) const {
    return !(*this == other);
  }
};

std::unordered_map<TensorView*, ResizeExclusivityInfo> getNonExclusiveResizeInfo(
    const std::vector<Expr*>& ordered_resize_tensor_ops,
    const ValGraph& exact_graph);

} // namespace scheduler_tools
} // namespace nvfuser
