// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/pass/insert_deallocations.h>
#include <ir/utils.h>

namespace nvfuser::hir_pass {

void InsertDeallocations::passImplementation(Fusion* fusion) {
  FusionGuard fg(fusion);
  hir::HostIrContainer* hic = dynamic_cast<hir::HostIrContainer*>(fusion);
  NVF_CHECK(hic, "Expected HostIrContainer");

  const std::vector<Expr*>& top_level_exprs = hic->topLevelExprs();
  std::for_each(top_level_exprs.begin(), top_level_exprs.end(), [](Expr* expr) {
    NVF_ERROR(
        !expr->isA<hir::Deallocate>(),
        "Expected hostir container to not have deallocate, but found one "
        "anyways");
  });
  std::unordered_map<TensorView*, int64_t> last_use;
  for (auto&& [i, expr] : enumerate(top_level_exprs)) {
    for (auto* val : expr->inputs()) {
      if (!val->isA<TensorView>()) {
        continue;
      }
      auto tv = val->as<TensorView>();
      last_use[tv] = i;
    }
  }

  // Remove inputs from last_use, they should not be deallocated
  for (auto* in : ir_utils::filterByType<TensorView>(hic->inputs())) {
    last_use.erase(in);
  }

  // Remove outputs from last_use, they should not be deallocated
  for (auto* out : ir_utils::filterByType<TensorView>(hic->outputs())) {
    last_use.erase(out);
  }

  std::vector<std::pair<int64_t, TensorView*>> last_use_by_index;
  last_use_by_index.reserve(last_use.size());
  for (auto&& [tv, i] : last_use) {
    last_use_by_index.emplace_back(i, tv);
  }
  std::sort(last_use_by_index.begin(), last_use_by_index.end());
  for (auto&& [i, tv] : last_use_by_index | std::views::reverse) {
    auto* deallocate = IrBuilder::create<hir::Deallocate>(tv);
    hic->insertExprAfter(i, deallocate);
  }
}

} // namespace nvfuser::hir_pass
