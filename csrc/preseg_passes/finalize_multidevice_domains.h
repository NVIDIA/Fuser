// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <preseg_passes/optimization_pass.h>

#include <string>

#include <fusion.h>

namespace nvfuser::preseg_passes {

// This pass:
// 1. Validates that all TensorViews have a device mesh or none.
// 2. Resharding expressions are mapped to collective libraries which expect
// contiguous tensors and output contiguous buffers. This pass checks that
// inputs are contiguous.
// 3. Sets the allocation domain of all fusion tvs if they have a device mesh.
// The allocation domain is obtained by transforming the `maybeAllocationDomain`
// using the transforms to loop domain. This ensures that the allocation domain
// has DID loop splits. All iterdomains derived from a given logical iterdomain
// are placed together. See `setLoopAndAllocationDomain` for more details.
// Eventually, this pass should run after `markAliasesPrepare` and
// `AllocationDomainPass` after they are fixed.
class FinalizeMultideviceDomainsPass
    : public OptimizationPass<FinalizeMultideviceDomainsPass> {
  friend class OptimizationPass<FinalizeMultideviceDomainsPass>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "FinalizeMultideviceDomainsPass";
  }
};

} // namespace nvfuser::preseg_passes
