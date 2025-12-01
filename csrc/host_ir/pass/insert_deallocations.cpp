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

namespace nvfuser::hir {

void InsertDeallocations::runPass(Fusion* fusion) {
  NVF_CHECK(fusion->isA<HostIrContainer>());

  FusionGuard fg(fusion);
  auto& hic = *(fusion->as<HostIrContainer>());

  const std::list<Expr*>& top_level_exprs = hic.topLevelExprs();
  std::for_each(top_level_exprs.begin(), top_level_exprs.end(), [](Expr* expr) {
    NVF_ERROR(
        !expr->isA<hir::Deallocate>(),
        "Expected hostir container to not have deallocate, but found one "
        "anyways: ",
        expr);
  });

  std::unordered_set<TensorView*> last_use_found;
  for (auto insertion_point = top_level_exprs.end();
       insertion_point != top_level_exprs.begin();) {
    auto prev = std::prev(insertion_point);
    Expr* e = *prev;

    // Only tensors need to be allocated.
    for (auto* in : ir_utils::filterByType<TensorView>(e->inputs())) {
      // Deallocate `in` if `in` needs to be deallocated and its last use is
      // `e`.
      //
      // Fusion inputs are managed by the caller.
      if (in->isFusionInput()) {
        continue;
      }

      // Fusion outputs need to be kept alive for the caller.
      if (in->isFusionOutput()) {
        continue;
      }

      // Skip if `e` is not the last use.
      if (!last_use_found.insert(in).second) {
        continue;
      }

      auto* deallocate = IrBuilder::create<hir::Deallocate>(in);
      hic.insertExprBefore(insertion_point, deallocate);
    }

    // Don't `--insertion_point;` because we'd like to skip newly inserted
    // deallocations.
    insertion_point = prev;
  }
}

} // namespace nvfuser::hir
