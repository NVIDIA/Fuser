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
#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

class ConvertOpToCommunication
    : public OptimizationPass<ConvertOpToCommunication> {
  friend class OptimizationPass<ConvertOpToCommunication>;

 public:
  static std::vector<Expr*> ConvertSingleOpToCommunication(
      Expr* c,
      DeviceIdxType my_device_idx,
      const HostIrLowerParams& params);

  static void setParams(const HostIrLowerParams& params) {
    params_ = params;
  }

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "ConvertOpToCommunication";
  }

 private:
  static HostIrLowerParams params_;
};

} // namespace nvfuser::preseg_passes
