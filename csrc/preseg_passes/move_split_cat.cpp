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
        id_model_for_merging_(
            fusion,
            /*build_graphs=*/true,
            /*allow_self_mapping=*/true),
        id_model_for_propagation_(
            fusion,
            /*build_graphs=*/true,
            /*allow_self_mapping=*/true) {}

  // Finds all cancellable <split,cat> pairs, cancels them and horizontallly
  // merges ops in between.
  void run();

 private:
  // Returns true when Exprs between `slices` and `pads` can be horizontally
  // merged and applied on the input of the split.
  bool horizontallyMergeable(
      const std::vector<SliceOp*>& slices,
      const std::vector<PadOp*>& pads);

  int64_t propagateCatAxis(
      const std::vector<IterDomain*>& source,
      const std::vector<IterDomain*>& destination,
      int64_t cat_axis);

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
  // `permute` into `use_def_chain` so the caller can reconstruct `out` by
  // replaying `use_def_chain` (in reverse order) on `in`.
  TensorView* findCancelingSplit(CatOp* cat, std::vector<Expr*>& use_def_chain);

  Fusion* fusion_;

  // TODO(wujingyue): keep two `IdGraph`s not two `IdModel`s. An `IdModel`
  // contains multiple graphs and we only care about the exact graph in it.
  IdModel id_model_for_merging_;
  IdModel id_model_for_propagation_;
};

bool sameOp(const std::vector<Expr*>& frontier) {
  return std::adjacent_find(
             frontier.begin(), frontier.end(), [](Expr* lhs, Expr* rhs) {
               return !lhs->sameOp(rhs);
             }) == frontier.end();
}

bool CancelSplitCat::horizontallyMergeable(
    const std::vector<SliceOp*>& slices,
    const std::vector<PadOp*>& pads) {
  NVF_ERROR(slices.size() == pads.size());
  NVF_ERROR(!slices.empty());

  // FIXME: make it a class member.
  ValGraph& exact_graph = id_model_for_merging_.idGraph(IdMappingMode::EXACT);
  {
    const std::vector<IterDomain*>& first_rfactor =
        slices[0]->output(0)->as<TensorView>()->getMaybeRFactorDomain();
    size_t num_dims = first_rfactor.size();
    for (size_t i = 1; i < slices.size(); i++) {
      const std::vector<IterDomain*>& rfactor =
          slices[i]->output(0)->as<TensorView>()->getMaybeRFactorDomain();
      if (rfactor.size() != num_dims) {
        return false;
      }
      for (size_t j = 0; j < num_dims; j++) {
        exact_graph.mapVals(first_rfactor[j], rfactor[j]);
      }
    }
  }

  for (PadOp* pad : pads) {
    auto* pad_out = pad->out()->as<TensorView>();
    if (id_model_for_merging_.hasSelfMapping(pad_out)) {
      return false;
    }
  }

  {
    const std::vector<IterDomain*>& first_root =
        pads[0]->out()->as<TensorView>()->getRootDomain();
    size_t num_dims = first_root.size();
    for (size_t i = 1; i < pads.size(); i++) {
      const std::vector<IterDomain*>& root =
          pads[i]->out()->as<TensorView>()->getRootDomain();
      if (root.size() != num_dims) {
        return false;
      }
      for (size_t j = 0; j < num_dims; j++) {
        if (!exact_graph.disjointValSets().strictAreMapped(
                first_root[j], root[j])) {
          return false;
        }
      }
    }
  }

  return true;
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
    for (auto j : c10::irange(
             static_cast<int64_t>(slice->out()->getRootDomain().size()))) {
      const bool sliced =
          (slice->out()->getRootDomain()[j] !=
           slice->out()->getMaybeRFactorDomain()[j]);
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
           ->getMaybeRFactorDomain()[split_axis]
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

int64_t CancelSplitCat::propagateCatAxis(
    const std::vector<IterDomain*>& source,
    const std::vector<IterDomain*>& destination,
    int64_t cat_axis) {
  ValGraph& exact_graph =
      id_model_for_propagation_.idGraph(IdMappingMode::EXACT);
  ValGroup cat_dim = exact_graph.toGroup(destination[cat_axis]);
  while (
      std::none_of(source.begin(), source.end(), [&](IterDomain* source_dim) {
        return exact_graph.toGroup(source_dim) == cat_dim;
      })) {
    const ExprGroups& defining_groups = exact_graph.getDefinitions(cat_dim);
    if (defining_groups.size() != 1) {
      return -1;
    }
    ExprGroup defining_group = defining_groups.front();
    Expr* def = defining_group->front();
    // FIXME: make this a function so we can early return.
    if (Split* split = dynamic_cast<Split*>(def)) {
      if (exact_graph.toGroup(split->outer()) == cat_dim) {
        cat_dim = exact_graph.toGroup(split->in());
      } else {
        return -1;
      }
    } else if (Merge* merge = dynamic_cast<Merge*>(def)) {
      cat_dim = exact_graph.toGroup(merge->outer());
    } else {
      return -1;
    }
  }

  cat_axis = std::find_if(
                 source.begin(),
                 source.end(),
                 [&](IterDomain* source_dim) {
                   return exact_graph.toGroup(source_dim) == cat_dim;
                 }) -
      source.begin();
  return cat_axis;
}

TensorView* CancelSplitCat::findCancelingSplit(
    CatOp* cat,
    std::vector<Expr*>& use_def_chain) {
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
      return nullptr;
    }

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
      return nullptr;
    }

    use_def_chain.push_back(frontier[0]);

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
      return nullptr;
    }
    slices.push_back(slice);
  }

  if (!horizontallyMergeable(slices, pads)) {
    return nullptr;
  }

  // Find the corresponding split_axis.
  const int64_t split_axis = propagateCatAxis(
      slices[0]->out()->getMaybeRFactorDomain(),
      pads[0]->out()->as<TensorView>()->getRootDomain(),
      cat->concatenatedDim());
  if (split_axis == -1) {
    return nullptr;
  }

  TensorView* split_in = slicesFormSplit(slices, split_axis);
  return split_in;
}

void CancelSplitCat::run() {
  std::vector<Expr*> exprs = fusion_->exprs();
  for (auto* cat : ir_utils::filterByType<CatOp>(exprs)) {
    std::vector<Expr*> use_def_chain;
    TensorView* split_in = findCancelingSplit(cat, std::ref(use_def_chain));
    if (split_in == nullptr) {
      continue;
    }

    Val* merged_out = split_in;
    for (auto i = use_def_chain.rbegin(), end = use_def_chain.rend(); i != end;
         i++) {
      Expr* merged = replayExprWithNewInput(*i, merged_out);
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
