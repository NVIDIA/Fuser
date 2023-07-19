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

} // namespace nvfuser
