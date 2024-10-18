// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <deque>
#include <vector>

#include <fusion.h>
#include <id_model/id_model.h>
#include <id_model/to_string.h>
#include <ir/utils.h>
#include <ops/alias.h>
#include <preseg_passes/segment_inplace_update.h>

namespace nvfuser::preseg_passes {
// When an intermediate tensorview is aliased to a fusion input,
// a RW race occurs, when the intermediate tensorview or
// the aliased input is in path of a broadcast.
// This preseg pass :
// 1. Finds any tensorviews used in inplace updates (AllocationType:ReuseBuffer)
// in the fusion
// 2. Traverses the fusion graph starting from broadcast ops and stores all
// direct/indirect producer/consumer tensorviews.
// 3. For all aliased tensorviews, if the aliased tensorview or the aliased
// input is present in the set of visited tensorviews in step 2, we insert a
// segment set and set to force a separate copy kernel. Additionally,
// we check for implict broadcasts if any aliased input already has a
// broadcast dimension that is concretized later in the fusion. This ensures
// that all write operations to the fusion inputs occur after the read
// operations have completed. See Issue #2664: https://
// github.com/NVIDIA/Fuser/issues/2664
namespace {
void insertSegmentSet(Fusion* fusion) {
  std::vector<TensorView*> aliased_tvs;

  // Find all tensorviews which are used in inplace updates.
  // Aliases will always be fusion outputs.
  for (Val* out : fusion->outputs()) {
    if (fusion->getOutputAlias(out->as<TensorView>()).type ==
        AllocationType::ReuseBuffer) {
      aliased_tvs.push_back(out->as<TensorView>());
    }
  }

  // Return early if there is no inplace update
  if (aliased_tvs.empty()) {
    return;
  }

  // fusion->exprs() is a topologically sorted list. Filter out the broadcast
  // ops from the list.
  auto all_exprs = fusion->exprs();
  auto all_bcast_ops = ir_utils::filterByType<BroadcastOp>(all_exprs);

  // Traverse and store all direct/indirect consumer tensorviews of these
  // broadcast nodes. If the tensorview has been visited, return --> this means
  // that we have already traversed that branch
  std::unordered_set<TensorView*> visited_tvs;
  for (auto bcast_op : all_bcast_ops) {
    std::deque<TensorView*> tvs_to_visit;
    tvs_to_visit.push_back(bcast_op->output(0)->as<TensorView>());
    while (!tvs_to_visit.empty()) {
      TensorView* current_tv = tvs_to_visit.front();
      tvs_to_visit.pop_front();
      if (visited_tvs.count(current_tv)) {
        continue;
      }
      visited_tvs.insert(current_tv);
      std::vector<Expr*> current_tv_uses = current_tv->uses();
      for (Expr* use : current_tv_uses) {
        for (auto output_tv :
             ir_utils::filterByType<TensorView>(use->outputs())) {
          tvs_to_visit.push_back(output_tv->as<TensorView>());
        }
      }
    }
  }

  // Traverse and store the direct/indirect producer tensorviews of these
  // broadcast nodes If that tensorview has been visited, return.
  for (auto bcast_op : all_bcast_ops) {
    std::deque<TensorView*> tvs_to_visit;
    tvs_to_visit.push_back(bcast_op->input(0)->as<TensorView>());
    while (!tvs_to_visit.empty()) {
      TensorView* current_tv = tvs_to_visit.front();
      tvs_to_visit.pop_front();
      if (visited_tvs.count(current_tv)) {
        continue;
      }
      visited_tvs.insert(current_tv);
      auto definition = current_tv->definition();
      if (definition != nullptr) {
        for (auto input_tv :
             ir_utils::filterByType<TensorView>(definition->inputs())) {
          tvs_to_visit.push_back(input_tv->as<TensorView>());
        }
      }
    }
  }

  // Use permissive IdModel graph to identify any concretized broadcast
  // iterdomain in any aliased input.
  auto id_model = IdModel(fusion, /*build_graphs=*/false);
  id_model.buildPermissiveGraph();
  const ValGraph& permissive_graph =
      id_model.idGraph(IdMappingMode::PERMISSIVE);

  auto hasConcretizedBroadcast = [&](TensorView* tv) -> bool {
    if (!tv->hasBroadcast()) {
      return false;
    }
    for (IterDomain* id : tv->getLogicalDomain()) {
      if (!id->isBroadcast()) {
        continue;
      }
      if (!permissive_graph.hasGroup(id)) {
        continue;
      }
      const ValGroup& val_group = permissive_graph.toGroup(id);
      for (auto other_id : val_group.get()->vector()) {
        if (!other_id->as<IterDomain>()->isBroadcast()) {
          return true;
        }
      }
    }
    return false;
  };

  // For all aliased tensorviews:
  // 1) if that tv or the corresponding aliased input is a producer/consumer of
  // a broadcast op, or 2) the aliased input has a concretized broadcast, insert
  // a (segment_set + set) to force the inplace update into a separate copy
  // kernel. NOTE: We cannot use a segment_set alone. Since, there will be no
  // data flow across this segment_set (the output of segment_set is an output
  // of given fusion with no uses), it will be merged with other segments.
  // https://github.com/NVIDIA/Fuser/blob/92b635125ae509cc6b2ccbe29e957586a9cbb059/csrc/fusion_segmenter.cpp#L2331-L2346
  for (auto aliased_tv : aliased_tvs) {
    TensorView* aliased_input =
        fusion->getOutputAlias(aliased_tv).aliased_io->as<TensorView>();
    if (visited_tvs.count(aliased_tv) || visited_tvs.count(aliased_input) ||
        hasConcretizedBroadcast(aliased_input)) {
      TensorView* alias_seg = segment_set(aliased_tv);
      TensorView* alias_copy = set(alias_seg);
      fusion->replaceOutput(aliased_tv, alias_copy);
    }
  }
}
} // namespace

void SegmentInplaceUpdatePass::runPass(Fusion* fusion) {
  insertSegmentSet(fusion);
}
} // namespace nvfuser::preseg_passes
