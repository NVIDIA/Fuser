// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/pass/deallocate.h>

namespace nvfuser::hir {

void insertDeallocations(HostIrContainer* hic) {
  const std::vector<Expr*>& topLevelExprs = hic->topLevelExprs();
  std::unordered_map<TensorView*, std::vector<nvfuser::Expr*>::size_type>
      lastUse;
  for (std::vector<nvfuser::Expr*>::size_type i = 0; i < topLevelExprs.size();
       i++) {
    Expr* expr = topLevelExprs[i];
    if (expr->isA<Deallocate>()) {
      auto* deallocate = expr->as<Deallocate>();
      auto* tv = deallocate->allocation();
      NVF_ERROR(
          lastUse.count(tv) > 0, "Tried to deallocate unknown TensorView");
      lastUse.erase(tv);
    } else {
      for (auto* val : expr->inputs()) {
        if (!val->isA<TensorView>()) {
          continue;
        }
        auto tv = val->as<TensorView>();
        lastUse[tv] = i;
      }
    }
  }

  std::map<std::vector<nvfuser::Expr*>::size_type, std::vector<TensorView*>>
      rmap;
  for (auto p : lastUse) {
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
