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

struct Use {
  TensorView* use_of;
  Expr* user;

  bool operator<(const Use& other) const {
    return std::make_pair(use_of, user) <
        std::make_pair(other.use_of, other.user);
  }

  bool operator==(const Use& other) const {
    return use_of == other.use_of && user == other.user;
  }
};

// A helper function that walks up from `out` until reaching a non-meta op or a
// fusion input. Returns where it stops.
Use findUseToSegment(
    TensorView* out,
    const AliasAnalysisResult& analysis,
    const std::unordered_set<Expr*>& used_by_non_aliases) {
  Expr* user = nullptr;
  while (true) {
    Expr* def = out->definition();
    if (analysis.getRoot(out) == nullptr || used_by_non_aliases.count(def)) {
      return {out, user};
    }
    out = def->input(0)->as<TensorView>();
    user = def;
  }
}

bool isSegmentSet(Expr* e) {
  if (LoadStoreOp* ldst = dynamic_cast<LoadStoreOp*>(e)) {
    if (ldst->opType() == LoadStoreOpType::SegmenterSet) {
      return true;
    }
  }
  return false;
}

// Collects all expressions that are (transitively) used by non-alias
// TensorViews.
std::unordered_set<Expr*> exprsUsedByNonAliases(
    const AliasAnalysisResult& analysis,
    Fusion* fusion) {
  std::vector<Val*> non_aliases;
  for (TensorView* tv : ir_utils::allTvs(fusion)) {
    if (analysis.getRoot(tv) == nullptr) {
      non_aliases.push_back(tv);
    }
  }
  std::vector<Expr*> used_by_non_aliases = StmtSort::getExprsTo(non_aliases);
  return {used_by_non_aliases.begin(), used_by_non_aliases.end()};
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
    if (analysis.getRoot(tv) == nullptr) {
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

  // The following emulates the bookend optimization. Only the output end is
  // implemented at this moment. In general, the algorithm tries to walk up
  // from each fusion output until reaching a non-alias, and put a
  // `segment_set` there so the meta ops that are skipped form a no-op segment.
  //
  // An important detail: a meta op preceding a non-meta op (i.e. has a
  // non-alias output TensorView) is treated as non-meta for the purpose of
  // bookend. This is to avoid over segmentation. For example, in
  //
  //   N/M -> M1 -> N/M
  //          |
  //          -> M2
  //
  // we want to avoid putting a `segment_set` before M1, a meta op, because
  // that would lead to two kernels. See AliasTest.DoNotOverSegment_* for more
  // examples.
  const std::unordered_set<Expr*>& used_by_non_aliases =
      exprsUsedByNonAliases(analysis, fusion);
  std::vector<Use> uses_to_segment;
  uses_to_segment.reserve(fusion->outputs().size());
  for (auto* out : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    Use use_to_segment = findUseToSegment(out, analysis, used_by_non_aliases);
    if (use_to_segment.use_of != out) {
      uses_to_segment.push_back(use_to_segment);
    }
  }

  // The remaining are optimizations to reduce the number of `segment_set`s
  // inserted.
  //
  // Group `uses_to_segment` by `use_of` and remove duplicates.
  std::sort(uses_to_segment.begin(), uses_to_segment.end());
  uses_to_segment.erase(
      std::unique(uses_to_segment.begin(), uses_to_segment.end()),
      uses_to_segment.end());

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    for (const auto& use : uses_to_segment) {
      debug() << "Will put a segment_set at " << use.user << std::endl;
    }
  }

  auto i = uses_to_segment.begin();
  while (i != uses_to_segment.end()) {
    TensorView* use_of = i->use_of;
    auto j = i;
    while (j != uses_to_segment.end() && j->use_of == use_of) {
      j++;
    }

    auto insert_segment_set = [&]() {
      // Put a `segment_set` after `use_of` and redirect aliasing users to
      // use the `segment_set`.

      // There are a few corner cases where we don't need to add a
      // `segment_set`. If `use_of` is only used by aliases, ...
      if (static_cast<size_t>(std::distance(i, j)) == use_of->uses().size()) {
        if (use_of->isFusionInput()) {
          // Putting a `segment_set` between a fusion input and its users is
          // unnecessary.
          return;
        }

        // Rarely, if `use_of` is already defined by `segment_set`, don't
        // create another `segment_set`.
        if (isSegmentSet(use_of->definition())) {
          return;
        }
      }

      // If all aliasing users are `segment_set`, don't create another
      // `segment_set`.
      if (std::all_of(
              i, j, [](const Use& use) { return isSegmentSet(use.user); })) {
        return;
      }

      // The general case.
      TensorView* copy = segment_set(use_of);
      std::for_each(i, j, [&](const Use& use) {
        ir_utils::replaceValInExprInputs(use.user, use_of, copy);
      });
      if (use_of->isFusionOutput()) {
        fusion->replaceOutput(use_of, copy);
      }
    };
    insert_segment_set();

    i = j;
  }

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << std::endl
            << "Fusion Transforms after " << name() << ":" << std::endl;
    fusion->printTransforms();
  }
}

} // namespace nvfuser::preseg_passes
