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

namespace nvfuser::preseg_passes {

// Propagate allocation domain from srcs to dsts.
// The pass update allocation domain on dsts tensor views.
//
// See details in Note [ Allocation Order Propagation ]
void inferenceAllocationOrder(
    Fusion* fusion,
    const std::vector<TensorView*>& srcs,
    const std::vector<TensorView*>& dsts);

// Realize allocation order propagation on fusion inputs to optimize allocation
// domain of output tensor. This optimization pass currently only applies to
// fusion outputs, but not intermediate tensors.
class AllocationDomainPass : public OptimizationPass<AllocationDomainPass> {
  friend class OptimizationPass<AllocationDomainPass>;

 protected:
  static void runPass(Fusion* fusion);
};

} // namespace nvfuser::preseg_passes
