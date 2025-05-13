// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <host_ir/lower.h>
#include <host_ir/lower_to_communication.h>
#include <host_ir/pass/optimization_pass.h>

namespace nvfuser::hir_pass {

class ConvertOpToCommunication
    : public OptimizationPass<ConvertOpToCommunication> {
  friend class OptimizationPass<ConvertOpToCommunication>;

 public:
  ConvertOpToCommunication(
      const HostIrLowerParams& params = HostIrLowerParams())
      : params_(params) {}

 protected:
  void passImplementation(Fusion* fusion);
  static constexpr std::string_view name() {
    return "ConvertOpToCommunication";
  }

 private:
  HostIrLowerParams params_;
};

} // namespace nvfuser::hir_pass
