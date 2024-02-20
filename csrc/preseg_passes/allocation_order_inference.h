// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <unordered_map>

#include <fusion.h>
#include <preseg_passes/optimization_pass.h>

namespace nvfuser {

// allocation order is the permutation to apply on a tensor view's rfactor
// domain to its allocation domain.
//
// i.e. For a channels last 4d tensor, we mark it as (0, 2, 3, 1). This is
// trying to present it more consistently with how we construct it with c++ API.
//     std::vector<IterDomain*> tv0_nhwc = {
//         tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
//     tv0->setAllocationDomain(tv0_nhwc, true);
using AllocationOrder = std::vector<int64_t>;

// Propagate allocation order from input to the entire fusion. It does NOT
// modify any fusion IR, but instead stores the propagated allocation order as
// an unordered_map from TensorView to permutation.
//
// See details in Note [ Allocation Order Propagation ]
std::unordered_map<const TensorView*, AllocationOrder> inferenceAllocationOrder(
    Fusion* fusion);

// Realize allocation order propagation on fusion inputs to optimize allocation domain of output tensor. This optimization pass currently only applies to fusion outputs, but not intermediate tensors.
class AllocationDomainPass : public OptimizationPass<AllocationDomainPass> {
  friend class OptimizationPass<AllocationDomainPass>;

 protected:
  static void runPass(Fusion* fusion);
};

} // namespace nvfuser
