// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "host_ir/passes.h"

#include "host_ir/allocate_and_deallocate.h"

namespace nvfuser::hir {

void runPasses(HostIrContainer& hic) {
  OptimizationPass<hir::AllocateAndDeallocate>::runPass(&hic);
}

} // namespace nvfuser::hir
