// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/pass/hoist_to_host.h>

namespace nvfuser {

void hoistScalarComputationToHost(kir::Kernel* kernel) {
  if (!kernel->hasManaged("hoist_to_host")) {
    return;
  }
  for (auto v : kernel->getManaged<std::vector<Val*>>("hoist_to_host")) {
    TORCH_INTERNAL_ASSERT(
        !v->isA<TensorView>(),
        "Hoisting tensor computation to host is not supported yet");
    kernel->addKernelInput(v);
  }
}

std::vector<Expr*> removeExprsHoistedToHost(
    kir::Kernel* kernel,
    const std::vector<Expr*>& exprs) {
  std::unordered_set<Val*> hoisted_vals(
      kernel->getKernelInputs().begin(), kernel->getKernelInputs().end());
  std::vector<Expr*> new_exprs;
  for (auto expr : exprs) {
    bool all_outputs_hoisted = true;
    bool any_outputs_hoisted = false;
    for (auto out : expr->outputs()) {
      if (hoisted_vals.count(out)) {
        any_outputs_hoisted = true;
      } else {
        all_outputs_hoisted = false;
      }
    }
    TORCH_INTERNAL_ASSERT(
        all_outputs_hoisted == any_outputs_hoisted,
        "Expression cannot have both hoisted and non-hoisted outputs");
    if (!all_outputs_hoisted) {
      new_exprs.push_back(expr);
    }
  }
  // TODO: this will leave some dead code in the kernel, but it is not a big
  // deal for now. In a followup PR, we should write a dead code elimination
  // pass to remove the dead code.
  return new_exprs;
}

} // namespace nvfuser
