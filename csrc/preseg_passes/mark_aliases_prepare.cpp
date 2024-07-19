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
    if (analysis.getNearestAliasedIo(tv) == nullptr) {
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

  std::vector<Val*> non_alias_outs;
  non_alias_outs.reserve(fusion->outputs().size());
  // Fusion inputs/outputs that are (1) aliased by another output and (2) not
  // an alias itself. Code will later add `segment_set` around them so aliases
  // are separated from non-aliases and more likely to be accepted by the no-op
  // scheduler. See AliasTest.OutputAliasesAnotherOutput and
  // AliasTest.SegmentMetaOps.
  //
  // This algorithm is suboptimal in many cases. See
  // https://github.com/NVIDIA/Fuser/issues/2395#issuecomment-2207043749 for a
  // proposal that may work better.
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

  // Mark all expressions that are (transitively) used by non-alias outputs.
  auto used_by_non_aliases =
      [](const std::vector<Expr*>& exprs) -> std::unordered_set<Expr*> {
    return {exprs.begin(), exprs.end()};
  }(StmtSort::getExprsTo(non_alias_outs));

  for (TensorView* aliased_io : aliased_ios) {
    // Divide the users of aliased_io into two groups according to
    // `used_by_non_aliases`.
    std::vector<Expr*> users_used_by_non_aliases;
    std::vector<Expr*> users_used_only_by_aliases;
    for (Expr* e : aliased_io->uses()) {
      if (used_by_non_aliases.count(e)) {
        users_used_by_non_aliases.push_back(e);
      } else {
        users_used_only_by_aliases.push_back(e);
      }
    }

    // If all users are used by non-alias outputs, do nothing. Adding
    // segment_set around unlikely creates more no-op regions.
    if (users_used_only_by_aliases.empty()) {
      continue;
    }

    // If all users are used by aliases, put a `segment_set` before it.
    if (users_used_by_non_aliases.empty()) {
      if (aliased_io->isFusionInput()) {
        // A `segment_set` before a fusion input is useless.
        continue;
      }
      // Rarely, if `aliased_io` is already defined by `segment_set`, don't
      // create another `segment_set`.
      if (LoadStoreOp* def =
              dynamic_cast<LoadStoreOp*>(aliased_io->definition())) {
        if (def->opType() == LoadStoreOpType::SegmenterSet) {
          continue;
        }
      }
      aliased_io->cacheBefore(LoadStoreOpType::SegmenterSet);
      continue;
    }

    // Some users of `aliased_io` are used by non-aliases, and some are used
    // only by aliases. We put a `segment_set` after `aliased_io` and redirect
    // the latter group to use the `segment_set`.
    TensorView* copy = segment_set(aliased_io);
    for (Expr* e : users_used_only_by_aliases) {
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
