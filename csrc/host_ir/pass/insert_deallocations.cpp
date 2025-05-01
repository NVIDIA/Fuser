// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/pass/insert_deallocations.h>

namespace nvfuser::hir {

void insertDeallocations(HostIrContainer* hic) {
  const std::vector<Expr*>& top_level_exprs = hic->topLevelExprs();
  std::for_each(top_level_exprs.begin(), top_level_exprs.end(), [](Expr* expr) {
    NVF_ERROR(
        !expr->isA<Deallocate>(),
        "Expected hostir container to not have deallocate, but found one anyways");
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

  std::map<int64_t, std::vector<TensorView*>> rmap;
  for (auto p : last_use) {
    rmap[p.second].push_back(p.first);
  }

  for (auto iter = rmap.rbegin(); iter != rmap.rend(); ++iter) {
    // free all in iter->second at iter->first
    for (TensorView* tv : iter->second) {
      auto* deallocate = IrBuilder::create<Deallocate>(tv);
      hic->insertExprAfter(iter->first, deallocate);
    }
  }
}

} // namespace nvfuser::hir
