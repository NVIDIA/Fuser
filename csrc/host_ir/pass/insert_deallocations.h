// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <host_ir/container.h>
#include <host_ir/pass/optimization_pass.h>

namespace nvfuser::hir_pass {

// Inserts deallocation for memory efficiency.
class InsertDeallocations : public OptimizationPass<InsertDeallocations> {
  friend class OptimizationPass<InsertDeallocations>;

 protected:
  void passImplementation(Fusion* fusion);
  static constexpr std::string_view name() {
    return "InsertDeallocations";
  }
};

} // namespace nvfuser::hir_pass
