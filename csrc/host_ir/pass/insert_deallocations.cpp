// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/pass/insert_deallocations.h>

#include <list>
#include <unordered_set>

#include <ir/iostream.h>
#include <ir/utils.h>

namespace nvfuser::hir_pass {

void InsertDeallocations::passImplementation(Fusion* fusion) {
  FusionGuard fg(fusion);
  auto* hic = dynamic_cast<hir::HostIrContainer*>(fusion);
  NVF_CHECK(hic, "Expected HostIrContainer");

  const std::list<Expr*>& top_level_exprs = hic->topLevelExprs();
  std::for_each(top_level_exprs.begin(), top_level_exprs.end(), [](Expr* expr) {
    NVF_ERROR(
        !expr->isA<hir::Deallocate>(),
        "Expected hostir container to not have deallocate, but found one "
        "anyways",
        expr);
  });

  std::unordered_set<TensorView*> fusion_inputs;
  for (auto* in : ir_utils::filterByType<TensorView>(hic->inputs())) {
    fusion_inputs.insert(in);
  }
  std::unordered_set<TensorView*> fusion_outputs;
  for (auto* out : ir_utils::filterByType<TensorView>(hic->outputs())) {
    fusion_outputs.insert(out);
  }

  std::unordered_set<TensorView*> last_use_found;
  for (auto insertion_point = top_level_exprs.end();
       insertion_point != top_level_exprs.begin();) {
    auto prev = std::prev(insertion_point);
    Expr* e = *prev;

    for (auto* val : e->inputs()) {
      if (!val->isA<TensorView>()) {
        continue;
      }
      auto* tv = val->as<TensorView>();

      if (fusion_inputs.count(tv) > 0 || fusion_outputs.count(tv) > 0) {
        continue;
      }

      if (!last_use_found.insert(tv).second) {
        continue;
      }

      auto* deallocate = IrBuilder::create<hir::Deallocate>(tv);
      hic->insertExprBefore(insertion_point, deallocate);
    }

    insertion_point = prev;
  }
}

} // namespace nvfuser::hir_pass
