// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <alias_analysis.h>
#include <debug.h>
#include <ir/iostream.h> // for operator<<(ostream&, TensorView*)
#include <ir/utils.h>
#include <ops/alias.h>
#include <options.h>
#include <preseg_passes/mark_aliases_prepare.h>

namespace nvfuser::preseg_passes {

namespace {

std::pair<TensorView*, Expr*> findUseToSegment(
    TensorView* out,
    const AliasAnalysisResult& analysis,
    const std::unordered_set<Expr*>& used_by_non_aliases) {
  Expr* user = nullptr;
  while (true) {
    Expr* def = out->definition();
    if (analysis.getNearestAliasedIo(out) == nullptr ||
        used_by_non_aliases.count(def)) {
      return {out, user};
    }
    out = def->input(0)->as<TensorView>();
    user = def;
  }
}

} // namespace

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

  auto used_by_non_aliases = [&]() -> std::unordered_set<Expr*> {
    std::vector<Val*> non_aliases;
    for (TensorView* tv : ir_utils::allTvs(fusion)) {
      if (analysis.getNearestAliasedIo(tv) == nullptr) {
        non_aliases.push_back(tv);
      }
    }
    // Mark all expressions that are (transitively) used by non-alias outputs.
    std::vector<Expr*> used_by_non_aliases = StmtSort::getExprsTo(non_aliases);
    return {used_by_non_aliases.begin(), used_by_non_aliases.end()};
  }();

  std::vector<std::pair<TensorView*, Expr*>> uses_to_segment;
  uses_to_segment.reserve(fusion->outputs().size());
  for (auto* out : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    std::pair<TensorView*, Expr*> use_to_segment =
        findUseToSegment(out, analysis, used_by_non_aliases);
    if (use_to_segment.first != out) {
      uses_to_segment.push_back(use_to_segment);
    }
  }
  std::sort(uses_to_segment.begin(), uses_to_segment.end());
  uses_to_segment.erase(
      std::unique(uses_to_segment.begin(), uses_to_segment.end()),
      uses_to_segment.end());

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    for (const auto& [use_of, user] : uses_to_segment) {
      debug() << "Will put a segment_set at " << user << std::endl;
    }
  }

  auto i = uses_to_segment.begin();
  while (i != uses_to_segment.end()) {
    TensorView* use_of = i->first;
    auto j = i;
    while (j != uses_to_segment.end() && j->first == use_of) {
      j++;
    }

    auto insert_segment_set = [&]() {
      if (static_cast<size_t>(std::distance(i, j)) == use_of->uses().size()) {
        // Put a `segment_set` before `use_of`.
        if (use_of->isFusionInput()) {
          // A `segment_set` before a fusion input is useless.
          return;
        }
        // Rarely, if `aliased_io` is already defined by `segment_set`, don't
        // create another `segment_set`.
        if (LoadStoreOp* def =
                dynamic_cast<LoadStoreOp*>(use_of->definition())) {
          if (def->opType() == LoadStoreOpType::SegmenterSet) {
            return;
          }
        }
        use_of->cacheBefore(LoadStoreOpType::SegmenterSet);
      } else {
        // Some users of `use_of` are used by non-aliases, and some are used
        // only by aliases. We put a `segment_set` after `use_of` and redirect
        // the latter group to use the `segment_set`.
        TensorView* copy = segment_set(use_of);
        std::for_each(i, j, [&](const std::pair<TensorView*, Expr*>& use) {
          ir_utils::replaceValInExprInputs(use.second, use_of, copy);
        });
        if (use_of->isFusionOutput()) {
          fusion->replaceOutput(use_of, copy);
        }
      }
    };
    insert_segment_set();

    i = j;
  }

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << "Fusion Transforms after " << name() << ":" << std::endl;
    fusion->printTransforms();
  }
}

} // namespace nvfuser::preseg_passes
