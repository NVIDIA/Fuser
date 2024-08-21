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
#include <transform_replay.h>

namespace nvfuser::preseg_passes {

namespace {

// Represents a use of `use_of` by `user`. This is to mark locations to segment
// so meta ops form a no-op region. When `user` is not null, we expect to
// segment between `use_of` and `user`, e.g.,
//
//   use_of -> [segment_set] -> copy of use_of -> [user]
//
// This happens due to bookending from outputs.
//
// When `user` is null, we expect to segment between `use_of` and all its
// users, e.g.,
//
//   use_of -> [segment_set] -> copy of use_of -> [user_0]
//                                             |
//                                             +> [user_1]
//                                             |
//                                             +> [user_2]
//
// This happens due to bookending from inputs.
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

std::ostream& operator<<(std::ostream& os, const Use& use) {
  os << use.use_of;
  if (use.user != nullptr) {
    os << " used by " << use.user;
  }
  return os;
}

// A helper function that walks up from `out` until reaching a non-meta op or a
// fusion input. Returns where it stops.
Use findUseToSegment(
    TensorView* out,
    const AliasAnalysisResult& analysis,
    const std::unordered_set<Expr*>& depended_by_non_aliases) {
  Expr* user = nullptr;
  while (true) {
    Expr* def = out->definition();
    if (analysis.getRoot(out) == nullptr ||
        depended_by_non_aliases.count(def)) {
      return {out, user};
    }
    out = def->input(0)->as<TensorView>();
    user = def;
  }
}

// Collects all expressions that are depended (i.e. transitively used) by
// non-alias TensorViews.
std::unordered_set<Expr*> exprsDependedByNonAliases(
    const AliasAnalysisResult& analysis,
    Fusion* fusion) {
  std::vector<Val*> non_aliases;
  for (TensorView* tv : ir_utils::allTvs(fusion)) {
    if (analysis.getRoot(tv) == nullptr) {
      non_aliases.push_back(tv);
    }
  }
  std::vector<Expr*> depended_by_non_aliases =
      StmtSort::getExprsTo(non_aliases);
  return {depended_by_non_aliases.begin(), depended_by_non_aliases.end()};
}

// Inserts a `segment_set` after `use_of` to separate meta and non-meta ops.
template <typename InputIter>
void insertSegmentSetAfter(InputIter first_user, InputIter last_user) {
  TensorView* use_of = first_user->use_of;

  std::vector<Expr*> users;
  users.reserve(use_of->uses().size());
  // `uses_to_segment` is sorted so `nullptr` if exists appears first.
  if (first_user->user == nullptr) {
    // This is an optimization to make fewer segments. In the
    // following example, if bookending wants to segment (a) between `use_of`
    // and all its users, (b) between `use_of` and `user_0`, and (c) between
    // `use_of` and `user_1`. We can instead segment only between `use_of` and
    // `user_2`, the complement set of [`first_user`, `last_user`). This is
    // valid because the ops before `use_of`, those after `user_0`, and those
    // after `user_1` are all meta ops that can be merged into one no-op
    // segment.
    //
    //   use_of | -> | [user 0]
    //            |
    //            +> | [user 1]
    //            |
    //            +> [user 2]
    //
    //   ==>
    //
    //   use_of -> [user 0]
    //          |
    //          +> [user 1]
    //          |
    //          +> | [user 2]
    first_user++;
    std::unordered_set<Expr*> to_remove;
    std::for_each(first_user, last_user, [&](const Use& use) {
      to_remove.insert(use.user);
    });
    std::copy_if(
        use_of->uses().begin(),
        use_of->uses().end(),
        std::back_inserter(users),
        [&](Expr* user) { return to_remove.count(user) == 0; });
  } else {
    std::transform(
        first_user, last_user, std::back_inserter(users), [](const Use& use) {
          return use.user;
        });
  }

  // There are a few corner cases where we can avoid adding a
  // `segment_set`. If a segment_set is to be added between `use_of` and all
  // its users, ...
  if (users.size() == use_of->uses().size()) {
    if (use_of->isFusionInput()) {
      // Putting a `segment_set` between a fusion input and all its users is
      // unnecessary.
      return;
    }

    // Rarely, if `use_of` is already defined by `segment_set`, don't
    // create another `segment_set`.
    if (ir_utils::isSegmentSet(use_of->definition())) {
      return;
    }
  }

  // If all users to segment are `segment_set`, don't create another
  // `segment_set`.
  if (std::all_of(users.begin(), users.end(), ir_utils::isSegmentSet)) {
    return;
  }

  // The general case.
  TensorView* copy = segment_set(use_of);
  // Inherit the allocation domain from `use_of`. This is needed for cases like
  // AliasTest.Bookend_SegmentSetPreservesAllocation.
  TensorDomain* replayed_domain =
      TransformReplay::replayCasP(
          copy, use_of, -1, TransformReplayOptions().replayAllocation())
          .first;
  if (replayed_domain->hasAllocation()) {
    copy->setAllocationDomain(
        replayed_domain->allocation(), replayed_domain->contiguity());
  }
  // This is an optimization to make fewer segments. In the following example,
  // we could literally add two `segment_set`s, one before `user_0` and the
  // other before `user_1`. However, because these `segment_set`s are implied
  // by bookending, the ops after `user_0` and those after `user_1` are all
  // meta and can be merged into one no-op segment.
  //
  //   use_of -> | [user_0]
  //          |
  //          +> | [user_1]
  //          |
  //          +> [user_2]
  //
  //   =>
  //
  //   use_of -> [segment_set] -> copy -> [user_0]
  //          |                        |
  //          |                        +> [user 1]
  //          |
  //          +> [user_2]
  std::for_each(users.begin(), users.end(), [&](Expr* user) {
    ir_utils::replaceValInExprInputs(user, use_of, copy);
  });
  if (use_of->isFusionOutput()) {
    use_of->fusion()->replaceOutput(use_of, copy);
  }
}

bool isMetaOp(const AliasAnalysisResult& analysis, Expr* e) {
  return std::all_of(
      e->outputs().begin(), e->outputs().end(), [&analysis](Val* out) {
        if (auto* out_tv = dynamic_cast<TensorView*>(out)) {
          return analysis.getRoot(out_tv) != nullptr;
        }
        return false;
      });
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

  // The following emulates the bookend optimization. This is done in two
  // steps: the first step bookends the outputs and the second step does the
  // inputs. TODO(wujingyue): extract this into a function. I'm adding the new
  // logic in place just to make review easier.
  //
  // Step 1: for outputs, the algorithm tries to walk up
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
  // examples. This is the reason behind `depended_by_non_aliases`.
  const std::unordered_set<Expr*>& depended_by_non_aliases =
      exprsDependedByNonAliases(analysis, fusion);
  std::set<Use> uses_to_segment;
  for (auto* out : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    Use use_to_segment =
        findUseToSegment(out, analysis, depended_by_non_aliases);
    if (use_to_segment.use_of != out) {
      uses_to_segment.insert(use_to_segment);
    }
  }

  // Step 2: for inputs, the algorithm tries to walk down from each fusion
  // input until reaching a non-meta or a fork. Stopping at the fork is to
  // avoid feeding the same data via multiple inputs, e.g.,
  //
  //   in -> reshape_0 -> mul
  //    |                  ^
  //    +--> reshape_1 ----+
  //
  // If we separate `reshape_0` and `reshape_1` from `mul`, the pointwise
  // kernel would take double the input.
  std::queue<TensorView*> frontier;
  for (auto* tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    frontier.push(tv);
  }
  while (!frontier.empty()) {
    TensorView* tv = frontier.front();
    frontier.pop();

    auto should_enqueue_users = [&analysis](TensorView* tv) {
      // Stop at a non-meta op.
      if (!std::all_of(
              tv->uses().begin(), tv->uses().end(), [&analysis](Expr* e) {
                return isMetaOp(analysis, e);
              })) {
        return false;
      }

      // Stop at a fork.
      if (tv->uses().size() > 1 &&
          // The only exception is when the fork happens to be a split, which
          // is a common pattern in RoPE.
          !std::all_of(
              tv->uses().begin(),
              tv->uses().end(),
              std::mem_fn(&Expr::isA<SliceOp>))) {
        return false;
      }

      return true;
    };
    if (should_enqueue_users(tv)) {
      for (Expr* user : tv->uses()) {
        // If the use of `tv` by `user` is going to be segmented due to
        // bookending outputs, stop there. We could keep bookending but further
        // segmenting a meta-op region is useless.
        if (uses_to_segment.count(Use{tv, user})) {
          continue;
        }
        for (auto* user_out :
             ir_utils::filterByType<TensorView>(user->outputs())) {
          frontier.push(user_out);
        }
      }
    } else {
      // Will insert a segment_set between `tv` and all its users. See
      // Use::user for more details.
      uses_to_segment.insert(Use{tv, nullptr});
    }
  }

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    for (const auto& use : uses_to_segment) {
      debug() << "Will put a segment_set at " << use << std::endl;
    }
  }

  // Because `uses_to_segment` has been sorted by the TensorView being used, we
  // use a double nested while loop to find and process all the users for each
  // TensorView.
  auto first_user = uses_to_segment.begin();
  while (first_user != uses_to_segment.end()) {
    TensorView* use_of = first_user->use_of;
    auto last_user = first_user;
    do {
      last_user++;
    } while (last_user != uses_to_segment.end() && last_user->use_of == use_of);
    // At this point, <first_user,last_user> points the first user of `use_of`
    // and one past the last user.

    insertSegmentSetAfter(first_user, last_user);

    first_user = last_user;
  }

  if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
    debug() << std::endl
            << "Fusion Transforms after " << name() << ":" << std::endl;
    fusion->printTransforms();
  }
}

} // namespace nvfuser::preseg_passes
