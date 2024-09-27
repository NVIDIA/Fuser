// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/make_resharding_contiguous.h>

#include <fusion.h>
#include <ir/interface_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <multidevice/utils.h>

namespace nvfuser::preseg_passes {

namespace {
void setShardedAllocationDomain(TensorView* tv) {
  if (!tv->hasAllocation()) {
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }
}
} // namespace

void MakeReshardingContiguousPass::runPass(Fusion* fusion) {
  for (Expr* expr : fusion->exprs()) {
    if (!isResharding(expr)) {
      continue;
    }
    for (auto* tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
      for (auto c : tv->getContiguity()) {
        if (c.has_value()) {
          NVF_CHECK(
              c.value(),
              "Resharding expression input must be contiguous: ",
              expr);
        }
      }
      setShardedAllocationDomain(tv);
    }
    for (auto tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
      setShardedAllocationDomain(tv);
    }
  }
}

} // namespace nvfuser::preseg_passes
