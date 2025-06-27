// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/move_split_cat.h>

#include <vector>

#include <fusion.h>
#include <id_model/id_model.h>
#include <ir/builder.h>
#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <ops/alias.h>
#include <transform_replay.h>

namespace nvfuser::preseg_passes {

namespace {

class CancelSplitCat {
 public:
  CancelSplitCat(Fusion* fusion)
      : fusion_(fusion),
        id_model_(
            fusion,
            /*build_graphs=*/false,
            /*allow_self_mapping=*/true) {
    id_model_.buildExactGraph();
  }

  // Finds all cancellable <split,cat> pairs, cancels them and horizontallly
  // merges ops in between.
  void run();

 private:
  // Returns true when the def-use chain from slices[i] to pads[i] apply the
  // same IterDomain transforms as the one from slices[j] to pads[j]. This is a
  // necessary condition for horizontally merging the chains.
  //
  // Pre-condition: this is called after findPairingSplit so we know these
  // chains contain the same sequence of op types and attributes.
  bool sameIterDomainTransforms(
      const std::vector<SliceOp*>& slices,
      const std::vector<PadOp*>& pads,
      int64_t cat_axis);

  // Imagine we "zip" the cat upwards as following:
  //
  //   s0, s1 = split(in)
  //   s0 = unary_0(s0)
  //   s1 = unary_0(s1)
  //   ...
  //   s0 = unary_k(s0)
  //   s1 = unary_k(s1)
  //   s = cat({s0, s1})
  //
  //   ==>
  //
  //   s0, s1 = split(in)
  //   s = cat({s0, s1})
  //   s = unary_0(s)
  //   ...
  //   s = unary_k(s)
  //
  // This function returns the concatenated axis of the new cat so the above
  // transform preserves the semantics. This axis will then be compared with the
  // split axis to determine whether the split and the cat cancel out.
  //
  // If we can't zip the cat up to the split outputs (see one of the following
  // examples), this function returns -1.
  //
  // Before calling this function, we already checked the chains contain the
  // same sequence of op type and attributes and transform IterDomains in the
  // same way. So this function takes the logical domain of any one of the
  // slices and the catted IterDomain at the end of that chain.
  //
  // Example 1:
  //   t = permute(slice, {1, 2, 0})
  //   out = cat({t, ...}, 1)
  //
  // Returns 2 because the catted dimension (dimension 1 of `t1`) is permuted
  // from dimension 2 of `slice`.
  //
  // Example 2:
  //   t = reshape(slice, {2, 3, 5}, {6, 5})
  //   out = cat({t, ...}, 1}
  //
  // Returns 2 because the catted dimension comes from dimension 2 of `slice`.
  //
  // Example 3:
  //   t = reshape(slice, {2, 3}, {6})
  //   out = cat({t, ...}, 0}
  //
  // Returns 0 because `slice`'s dimension 0 is the outer dimension.

  // Example 4:
  //   t = reshape(slice, {6}, {2, 3})  // inner split by 3
  //   out = cat({t, ...}, 0}
  //
  // Returns 0 because `t` is an inner split and `out`'s dimension 0 is the
  // outer dimension. See #2142 for why checking the split is inner/outer.
  //
  // Example 5:
  //   t = reshape(slice, {6}, {2, 3})  // outer split by 2
  //   out = cat({t, ...}, 0}
  //
  // Returns -1 because `t` is an outer split. See #2142 for why checking the
  // split is inner/outer.
  //
  // Example 6:
  //   t = reshape(slice, {6}, {2, 3})
  //   out = cat({t, ...}, 1}
  //
  // Returns -1 because `out`'s dimension 1 is the inner dimension. Consider
  //   in = arange(12)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  //   x, y = in.split([6, 6]) # [0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]
  //   x = x.view([2, 3])  # [[0, 1, 2], [3, 4, 5]]
  //   y = y.view([2, 3])  # [[6, 7, 8], [9, 10, 11]]
  //   out = cat([x, y], axis=1)  # [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]]
  // It's impossible to set strides to make `out` a view of `in`.
  int64_t computeCatAxisAfterZipping(
      const std::vector<IterDomain*>& slice_rfactor,
      IterDomain* cat_id);

  // Finds the canceling split of `cat` and returns the input TensorView of the
  // split. A split (implemented as multiple `slice`s) and a cat cancel when
  // they work on the same dimension. For example, when
  //
  //   s0 = in[:, :5]
  //   s1 = in[:, 5:]
  //   out = cat([s0, s1], dim=-1)
  //
  // findCancelingSplit(out) returns `in`.
  //
  // `cat` doesn't have to immediately follow the split. For example, when
  //
  //   s0 = in[:, :5]
  //   s1 = in[:, 5:]
  //   t0 = permute(s0)
  //   t1 = permute(s1)
  //   out = cat([t0, t1], dim=0)
  //
  // In addition to returning `in`, findCancelingSplit(out) puts `t0`'s defining
  // `permute` into `def_use_chain` so the caller can reconstruct `out` by
  // replaying `def_use_chain` on `in`.
  TensorView* findCancelingSplit(CatOp* cat, std::vector<Expr*>& def_use_chain);

  Fusion* fusion_;

  // `id_model_` is supposed to be read-only and reflect the
  // original fusion.
  IdModel id_model_;
};

bool sameOp(const std::vector<Expr*>& frontier) {
  return std::adjacent_find(
             frontier.begin(), frontier.end(), [](Expr* lhs, Expr* rhs) {
               return !lhs->sameOp(rhs);
             }) == frontier.end();
}

bool CancelSplitCat::sameIterDomainTransforms(
    const std::vector<SliceOp*>& slices,
    const std::vector<PadOp*>& pads,
    const int64_t cat_axis) {
  NVF_ERROR(slices.size() == pads.size());
  NVF_ERROR(!slices.empty());

  // This clones the exact graph so that `mapVals` are done without affecting
  // other split/cat pairs in consideration. See
  // MoveSplitCatTest.MultipleCatsOnSameSplit for why this independence is
  // important.
  ValGraph exact_graph = id_model_.idGraph(IdMappingMode::EXACT);
  {
    // Map pads[i0].root[cat_axis] and pads[i1].root[cat_axis]. Other axes were
    // already mapped due to the `cat` when the IdModel was built.
    const std::vector<IterDomain*>& first_root =
        pads[0]->out()->as<TensorView>()->getMaybeRootDomain();
    for (size_t i = 1; i < pads.size(); i++) {
      const std::vector<IterDomain*>& other_root =
          pads[i]->out()->as<TensorView>()->getMaybeRootDomain();
      NVF_ERROR(first_root.size() == other_root.size());
      exact_graph.mapVals(first_root[cat_axis], other_root[cat_axis]);
    }
  }

  // The above code block only maps IterDomains across chains. If a self mapping
  // is detected at this point, it's likely due to some IterDomains are permuted
  // diffrently between two chains. See
  // MoveSplitCatTest.Noncancellable_PermutedDifferently for an example.
  for (auto* slice : slices) {
    if (hasSelfMapping(slice->out(), exact_graph)) {
      return false;
    }
  }

  {
    // Check slices[i0][j] and slices[i1][j] are mapped.
    const std::vector<IterDomain*>& first_logical =
        slices[0]->out()->getLogicalDomain();
    size_t num_dims = first_logical.size();
    for (size_t i = 1; i < slices.size(); i++) {
      const std::vector<IterDomain*>& other_logical =
          slices[i]->out()->getLogicalDomain();
      if (other_logical.size() != num_dims) {
        return false;
      }
      for (size_t j = 0; j < num_dims; j++) {
        if (!exact_graph.disjointValSets().strictAreMapped(
                first_logical[j], other_logical[j])) {
          return false;
        }
      }
    }
    return true;
  }
}

// If `slices` form a split, returns the base tensor of the
// split. Returns null otherwise.
TensorView* slicesFormSplit(
    const std::vector<SliceOp*>& slices,
    const int64_t split_axis) {
  // Checks that all exprs are slices and are based on the
  // same tensor. Otherwise, they don't form a split.
  TensorView* split_in = nullptr;
  for (auto* slice : slices) {
    if (split_in == nullptr) {
      split_in = slice->in();
    } else if (split_in != slice->in()) {
      return nullptr;
    }
  }
  NVF_ERROR(split_in != nullptr);

  // Check that `exprs` (already known to be `SliceOp`s) form a split along
  // `split_axis`.
  //
  // `split_ranges[i]` is the slice range of `exprs[i]` for the split axis.
  std::vector<Slice> split_ranges;
  split_ranges.reserve(slices.size());
  for (auto* slice : slices) {
    const std::vector<Slice>& slice_ranges = slice->getRanges();
    // Check the steps are all one.
    if (std::any_of(
            slice_ranges.begin(),
            slice_ranges.end(),
            [](const Slice& slice_range) {
              return !slice_range.step->isOne();
            })) {
      return nullptr;
    }

    // Check only the split axis is sliced.
    for (auto j : arange(
             static_cast<int64_t>(slice->out()->getMaybeRootDomain().size()))) {
      const bool sliced =
          (slice->out()->getMaybeRootDomain()[j] !=
           slice->out()->getLogicalDomain()[j]);
      if ((j == split_axis) != sliced) {
        return nullptr;
      }
    }

    // Collect the slice range for the split axis.
    split_ranges.push_back(slice_ranges[split_axis]);
  }

  if (!split_ranges.front().start->isZero()) {
    return nullptr;
  }
  // Due to the limitation of `sameAs` mentioned in #1859, I can't check
  // split_ranges.back().stop is the same as the dimension size. Below is a
  // slightly lengthy workaround.
  if (!slices.back()
           ->out()
           ->getLogicalDomain()[split_axis]
           ->definition()
           ->as<Resize>()
           ->rightExpand()
           ->isZero()) {
    return nullptr;
  }
  for (size_t i = 0; i + 1 < slices.size(); i++) {
    if (!split_ranges[i].stop->sameAs(split_ranges[i + 1].start)) {
      return nullptr;
    }
  }

  return split_in;
}

int64_t CancelSplitCat::computeCatAxisAfterZipping(
    const std::vector<IterDomain*>& slice_rfactor,
    IterDomain* cat_id) {
  const ValGraph& exact_graph = id_model_.idGraph(IdMappingMode::EXACT);
  ValGroup cat_group = exact_graph.toGroup(cat_id);
  while (cat_group != nullptr) {
    // If `cat_group` contains a slice rfactor ID, return the index of that ID.
    auto i = std::find_if(
        slice_rfactor.begin(), slice_rfactor.end(), [&](IterDomain* id) {
          return exact_graph.toGroup(id) == cat_group;
        });
    if (i != slice_rfactor.end()) {
      return i - slice_rfactor.begin();
    }

    // Conceptually zip `cat_group` over its definition.
    auto cat_group_after_zipping = [&](ValGroup cat_group) -> ValGroup {
      const ExprGroups& defining_groups = exact_graph.getDefinitions(cat_group);
      if (defining_groups.size() != 1) {
        return nullptr;
      }
      ExprGroup defining_group = defining_groups.front();
      // Pick an arbitrary Expr from defining_group as the representative.
      Expr* def = defining_group->front();

      if (Split* split = dynamic_cast<Split*>(def)) {
        // Check `split` is an inner split to avoid #2142. If we allow an outer
        // split, `replayExprWithNewInputs` will incorrectly use the same split
        // factor as the extent of the horizontally merged dimension. We could
        // instead make `replayExprWithNewInputs` smarter, but I haven't given
        // that alternative enough thought.
        if (exact_graph.toGroup(split->outer()) == cat_group &&
            split->innerSplit()) {
          return exact_graph.toGroup(split->in());
        }
        return nullptr;
      }

      if (Merge* merge = dynamic_cast<Merge*>(def)) {
        return exact_graph.toGroup(merge->outer());
      }

      return nullptr;
    };
    cat_group = cat_group_after_zipping(cat_group);
  }

  return -1;
}

// Finds the pairing split of `cat` by traversing the use-def chains. If found,
// returns the slices of the pairing split and `cat`'s preceding `PadOp`s. This
// function does some basic checks like:
// 1. Ops between the chains must have the same op type and attributes.
// 2. Chains must end with slices.
// However, these checks are necessary but not sufficient to guarantee the
// pairing split is canceling. To make that decision, the caller has to further
// inspect the ops in between.
std::optional<std::pair<std::vector<SliceOp*>, std::vector<PadOp*>>>
findPairingSplit(CatOp* cat) {
  NVF_CHECK(!cat->inputs().empty(), "`cat` has zero inputs: ", cat);

  // `PadOp`s that produce `cat`'s inputs.
  std::vector<PadOp*> pads;
  pads.reserve(cat->inputs().size());

  // `frontier` initially contains the `Expr`s that precede `pads`. Then, we
  // repeatedly try to move the frontier up in lockstep as long as Exprs in the
  // frontier can be horizontally merged and applied on the unsplit tensor.
  std::vector<Expr*> frontier;
  frontier.reserve(cat->inputs().size());
  for (Val* in : cat->inputs()) {
    auto* pad = in->definition()->as<PadOp>();
    pads.push_back(pad);
    frontier.push_back(pad->in()->definition());
  }

  // Exit the loop when any Expr in `frontier` is a slice or a null.
  while (std::none_of(frontier.begin(), frontier.end(), [](Expr* e) {
    return e == nullptr || e->isA<SliceOp>();
  })) {
    if (!sameOp(frontier)) {
      return std::nullopt;
    }

    // We can probably extend this list to include many other unary ops.
    // Currently, I limit this to only reshapes and permutes to reduce blast
    // radius.
    auto supported = [](Expr* e) -> bool {
      if (e->isA<ViewOp>()) {
        return true;
      }
      if (auto* set = dynamic_cast<LoadStoreOp*>(e)) {
        if (set->opType() == LoadStoreOpType::Set) {
          return true;
        }
      }
      return false;
    };
    if (!supported(frontier[0])) {
      return std::nullopt;
    }

    // Advance the frontier in lockstep.
    for (Expr*& e : frontier) {
      NVF_ERROR(
          e->inputs().size() == 1,
          "All mergeable Exprs should be unary at this moment, but found: ",
          e);
      e = e->input(0)->definition();
    }
  }

  std::vector<SliceOp*> slices;
  slices.reserve(frontier.size());
  for (Expr* e : frontier) {
    auto* slice = dynamic_cast<SliceOp*>(e);
    if (slice == nullptr) {
      return std::nullopt;
    }
    slices.push_back(slice);
  }

  return std::make_pair(slices, pads);
}

TensorView* CancelSplitCat::findCancelingSplit(
    CatOp* cat,
    std::vector<Expr*>& def_use_chain) {
  auto heads_and_tails = findPairingSplit(cat);
  if (!heads_and_tails.has_value()) {
    return nullptr;
  }
  std::vector<SliceOp*> slices;
  std::vector<PadOp*> pads;
  std::tie(slices, pads) = *heads_and_tails;

  int64_t cat_axis = cat->concatenatedDim();
  if (!sameIterDomainTransforms(slices, pads, cat_axis)) {
    return nullptr;
  }

  cat_axis = computeCatAxisAfterZipping(
      slices[0]->out()->getLogicalDomain(),
      pads[0]->out()->as<TensorView>()->getMaybeRootDomain()[cat_axis]);
  if (cat_axis == -1) {
    return nullptr;
  }

  TensorView* split_in = slicesFormSplit(slices, cat_axis);
  if (split_in == nullptr) {
    return nullptr;
  }

  std::vector<Expr*> first_chain =
      StmtSort::getExprsBetween({slices[0]->out()}, {pads[0]->in()});
  def_use_chain.swap(first_chain);
  return split_in;
}

void CancelSplitCat::run() {
  std::vector<Expr*> exprs = fusion_->exprs();
  for (auto* cat : ir_utils::filterByType<CatOp>(exprs)) {
    std::vector<Expr*> def_use_chain;
    TensorView* split_in = findCancelingSplit(cat, std::ref(def_use_chain));
    if (split_in == nullptr) {
      continue;
    }

    Val* merged_out = split_in;
    for (Expr* e : def_use_chain) {
      Expr* merged = replayExprWithNewInput(e, merged_out);
      NVF_ERROR(
          merged->outputs().size() == 1,
          "Currently, we merge only unary ops, so it would be a programming "
          "mistake when the number of outputs is ",
          merged->outputs().size());
      merged_out = merged->output(0);
    }
    // `cat->output(0)` may be a fusion output with allocation domain.
    // Therefore, instead of replacing the output, we create a Set to preserve
    // the output allocation domain.
    IrBuilder::create<LoadStoreOp>(
        LoadStoreOpType::Set, cat->output(0), merged_out);
  }
}

} // namespace

void MoveSplitCatPass::runPass(Fusion* fusion) {
  CancelSplitCat(fusion).run();
}

} // namespace nvfuser::preseg_passes
