// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/container.h>
#include <host_ir/lower.h>
#include <host_ir/lower_to_communication.h>
#include <host_ir/pass/convert_op_to_communication.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <multidevice/communication.h>
#include <multidevice/utils.h>

namespace nvfuser::hir_pass {

void ConvertOpToCommunication::passImplementation(Fusion* fusion) {
  FusionGuard fg(fusion);
  hir::HostIrContainer* hic = dynamic_cast<hir::HostIrContainer*>(fusion);
  NVF_CHECK(hic, "Expected HostIrContainer");
  DeviceIdxType my_device_index = Communicator::getInstance().deviceId();

  auto handle_top_level_expr = [&](Expr* top_level_expr,
                                   std::vector<Expr*>& new_top_level_exprs) {
    if (!isResharding(top_level_expr)) {
      return new_top_level_exprs.push_back(top_level_expr);
    }
    for (auto* expr : nvfuser::convertSingleOpToCommunication(
             top_level_expr, my_device_index, params_)) {
      // Allocate the recv buffers of communications
      if (expr->isA<Communication>()) {
        auto* communication = expr->as<Communication>();
        TensorView* tv = communication->out();
        if (tv->getDeviceMesh().has(my_device_index) == 0) {
          auto* allocate =
              IrBuilder::create<kir::Allocate>(tv, MemoryType::Global);
          new_top_level_exprs.push_back(allocate);
        }
      }
      new_top_level_exprs.push_back(expr);
      if (expr->isA<Communication>()) {
        auto wait = IrBuilder::create<hir::Wait>(expr->as<Communication>());
        new_top_level_exprs.push_back(wait);
      }
    }
  };

  std::vector<Expr*> new_top_level_exprs;
  for (auto top_level_expr : hic->topLevelExprs()) {
    if (top_level_expr->isA<ForLoop>()) {
      auto* for_loop = top_level_expr->as<ForLoop>();
      std::vector<Expr*> new_for_loop_body;
      for (auto* expr : for_loop->body().exprs()) {
        handle_top_level_expr(expr, new_for_loop_body);
      }
      for_loop->body().clear();
      for (auto* expr : new_for_loop_body) {
        for_loop->body().push_back(expr);
      }
      new_top_level_exprs.push_back(for_loop);
    } else {
      handle_top_level_expr(top_level_expr, new_top_level_exprs);
    }
  }
  hic->resetTopLevelExprs(new_top_level_exprs);
}

} // namespace nvfuser::hir_pass
