// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/pass/insert_new.h>

namespace nvfuser::hir_pass {

void InsertNewTensor::passImplementation(Fusion* fusion) {
  FusionGuard fg(fusion);
  hir::HostIrContainer* hic = dynamic_cast<hir::HostIrContainer*>(fusion);
  NVF_CHECK(hic, "Expected HostIrContainer");
  std::unordered_set<TensorView*> allocated_tensors;

  const std::vector<Expr*>& top_level_exprs = hic->topLevelExprs();
  std::for_each(top_level_exprs.begin(), top_level_exprs.end(), [](Expr* expr) {
    NVF_ERROR(
        !expr->isA<hir::NewTensor>(),
        "Expected hostir container to not have new tensor, but found one "
        "anyways");
  });

  std::vector<std::pair<int64_t, TensorView*>> tensors_to_allocate;
  for (auto&& [i, expr] : enumerate(top_level_exprs)) {
    if (auto* allocated_tensor = dynamic_cast<kir::Allocate*>(expr)) {
       allocated_tensors.insert(allocated_tensor->buffer());
    }
    else if (BinaryOp* binary_op = dynamic_cast<BinaryOp*>(expr)) {
      for (auto* output : binary_op->outputs()) {
        if (output->isA<TensorView>()) {
          auto* tv = output->as<TensorView>();
          if (!allocated_tensors.contains(tv)) {
            tensors_to_allocate.emplace_back(i, tv);
            allocated_tensors.insert(tv);
          }
        }
      }
    }
  }

  std::sort(tensors_to_allocate.begin(), tensors_to_allocate.end(), 
            std::greater<std::pair<int64_t, TensorView*>>());

  for (auto&& [i, tv] : tensors_to_allocate) {
    auto* new_tensor = IrBuilder::create<hir::NewTensor>(tv);
    hic->insertExprBefore(i, new_tensor);
  }
}

} // namespace nvfuser::hir_pass
