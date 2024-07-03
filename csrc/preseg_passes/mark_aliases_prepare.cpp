// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <alias_analysis.h>
#include <debug.h>
#include <ir/utils.h>
#include <ops/alias.h>
#include <options.h>
#include <preseg_passes/mark_aliases_prepare.h>

namespace nvfuser::preseg_passes {

void MarkAliasesPreparePass::runPass(Fusion* fusion) {
  const AliasAnalysisResult analysis =
      findAliases(fusion, /*can_override_empty_allocation_domain=*/true);
  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Alias analysis result:" << std::endl;
    debug() << analysis.toString(/*indent_size=*/1) << std::endl;
  }

  // Materialize the alias-enabling allocation domain.
  for (TensorView* tv : ir_utils::allTvs(fusion)) {
    TensorView* aliased_io = analysis.getNearestAliasedIo(tv);
    if (aliased_io == nullptr) {
      continue;
    }

    // `AliasAnalysisResult::finalize` already checked the alias-enabling layout
    // is compliant with `tv`'s existing layout before adding `tv` to
    // `alias_to_root_`. So the existing layout can remain unchanged.
    if (tv->hasAllocation()) {
      continue;
    }

    // A scalar `tv` triggers a corner case that crashes
    // `validateDomainEquivalence`.
    if (tv->isZeroDim()) {
      continue;
    }

    const Layout preferred_layout = analysis.preferredLayout(tv);
    tv->setAllocationDomain(
        preferred_layout.allocation_domain, preferred_layout.contiguity);
    if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
      debug() << "Set the layout of " << ir_utils::varName(tv) << " to "
              << preferred_layout.toString() << std::endl;
    }
  }

  // FIXME: Fusion outputs that are (1) aliased by another fusion output, (2)
  // not aliases themselves, and (3) not fusion inputs (yes, a fusion may
  // trivially forward an input). Code will later add `segment_set` before them
  // so aliases are separated from non-aliases and more likely to be accepted by
  // the no-op scheduler.
  std::vector<Val*> non_alias_outs;
  non_alias_outs.reserve(fusion->outputs().size());
  std::unordered_set<TensorView*> aliased_ios;
  for (TensorView* out_tv :
       ir_utils::filterByType<TensorView>(fusion->outputs())) {
    if (TensorView* aliased_io = analysis.getNearestAliasedIo(out_tv)) {
      if (analysis.getNearestAliasedIo(aliased_io) == nullptr) {
        aliased_ios.insert(aliased_io);
      }
    } else {
      non_alias_outs.push_back(out_tv);
    }
  }

  auto used_for_non_alias =
      [](const std::vector<Expr*>& exprs) -> std::unordered_set<Expr*> {
    return {exprs.begin(), exprs.end()};
  }(StmtSort::getExprsTo(non_alias_outs));

  for (TensorView* aliased_io : aliased_ios) {
    std::vector<Expr*> users_for_non_alias;
    std::vector<Expr*> users_for_only_alias;
    for (Expr* e : aliased_io->uses()) {
      if (used_for_non_alias.count(e)) {
        users_for_non_alias.push_back(e);
      } else {
        users_for_only_alias.push_back(e);
      }
    }

    if (users_for_only_alias.empty()) {
      continue;
    }

    if (users_for_non_alias.empty()) {
      if (aliased_io->isFusionInput()) {
        continue;
      }
      if (LoadStoreOp* def =
              dynamic_cast<LoadStoreOp*>(aliased_io->definition())) {
        if (def->opType() == LoadStoreOpType::SegmenterSet) {
          continue;
        }
      }
      aliased_io->cacheBefore(LoadStoreOpType::SegmenterSet);
      continue;
    }

    TensorView* copy = segment_set(aliased_io);
    for (Expr* e : users_for_only_alias) {
      ir_utils::replaceValInExprInputs(e, aliased_io, copy);
    }
    if (aliased_io->isFusionOutput()) {
      fusion->replaceOutput(aliased_io, copy);
    }
  }

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion Transforms after " << name() << ":" << std::endl;
    fusion->printTransforms();
  }
}

} // namespace nvfuser::preseg_passes
