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

/* For each input in every expression in the container, find the index of its
 * last use and insert a new tensor directly after */
class InsertNewTensor : public OptimizationPass<InsertNewTensor> {
  friend class OptimizationPass<InsertNewTensor>;

 protected:
  void passImplementation(Fusion* fusion);
  static constexpr std::string_view name() {
    return "InsertNewTensor";
  }
};

} // namespace nvfuser::hir_pass
