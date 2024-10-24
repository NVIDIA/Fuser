// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <id_model/id_model.h>
#include <ir/cloner.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <scheduler/tools/loop_domain_scheduler.h>
#include <scheduler/tools/resize_utils.h>
#include <val_graph_visitor.h>

namespace nvfuser {
namespace scheduler_tools {

void propagateCatToInputs(CatOp* cat_op) {
  Fusion* fusion = cat_op->fusion();

  DisjointSets<TensorView*> input_sets;

  std::cerr << "propagateResizeToCatInputs: " << cat_op->toString();

  auto get_inputs = [&](Val* tv) -> std::vector<Val*> {
    auto dep_inputs = DependencyCheck::getAllValsBetween(
        {fusion->inputs().begin(), fusion->inputs().end()}, {tv});
    dep_inputs.erase(
        std::remove_if(
            dep_inputs.begin(),
            dep_inputs.end(),
            [&](Val* dep_tv) {
              return dep_tv == tv || dep_tv->isFusionInput();
            }),
        dep_inputs.end());
    return dep_inputs;
  };

  auto privatize_cat_input = [&](TensorView* cat_input) -> TensorView* {
    auto private_copy = RecomputeTv::recompute(cat_input);
    DisjointSets<TensorView*>::DisjointSet& input_set =
        input_sets.initializeSet(private_copy).first->second;
    std::stringstream ss;
    ss << private_copy->toString();
    for (auto val : get_inputs(private_copy)) {
      ss << " " << val->toString();
      input_sets.appendToSet(val->as<TensorView>(), input_set);
    }
    std::cerr << "recomputed: " << ss.str() << "\n";
    return private_copy;
  };

  auto has_overlap = [&input_sets](TensorView* cat_input) -> bool {
    Fusion* fusion = cat_input->fusion();
    std::cerr << "Cat input: " << cat_input->toString() << "\n";
    if (input_sets.mappingExists(cat_input)) {
      // Overlapped cat inputs
      std::cerr << "Overlapped input: " << cat_input->toString() << "\n";
      return true;
    }
    DisjointSets<TensorView*>::DisjointSet& input_set =
        input_sets.initializeSet(cat_input).first->second;
    auto dep_inputs = DependencyCheck::getAllValsBetween(
        {fusion->inputs().begin(), fusion->inputs().end()}, {cat_input});
    dep_inputs.erase(
        std::remove_if(
            dep_inputs.begin(),
            dep_inputs.end(),
            [&](Val* tv) { return tv == cat_input || tv->isFusionInput(); }),
        dep_inputs.end());
    std::cerr << "Dep input: " << toDelimitedString(dep_inputs) << "\n";
    for (auto tv : ir_utils::filterByType<TensorView>(dep_inputs)) {
      if (input_sets.mappingExists(tv)) {
        // Overlapped cat inputs
        std::cerr << "Overlapped input: " << tv->toString() << "\n";
        return true;
      }
      input_sets.appendToSet(tv, input_set);
    }
    return false;
  };

  std::vector<std::pair<TensorView*, TensorView*>> replaement_map;
  for (auto inp_tv : ir_utils::filterByType<TensorView>(cat_op->inputs())) {
    bool overlap = has_overlap(inp_tv);
    if (overlap) {
      auto private_copy = privatize_cat_input(inp_tv);
      replaement_map.emplace_back(inp_tv, private_copy);
    }
  }

  auto updated_cat_op = cat_op;
  for (const auto& [original, clone] : replaement_map) {
    std::cerr << "Replacing " << original->toString() << " with "
              << clone->toString() << "\n";
    updated_cat_op =
        ir_utils::replaceValInExprInputs(updated_cat_op, original, clone)
            ->as<CatOp>();
  }

  std::cerr << "New cat op: " << updated_cat_op->toString();

  std::cerr << "Num disjoint sets: " << input_sets.size() << "\n";

  std::cerr << "Propagating cat resizes to each disjoint set\n";

  for (auto inp_tv :
       ir_utils::filterByType<TensorView>(updated_cat_op->inputs())) {
    std::cerr << "Cat input: " << inp_tv->toString() << "\n";
    const auto& inp_dep_set = input_sets.getDisjointSetOf(inp_tv);
    std::cerr << "Dep: " << toDelimitedString(inp_dep_set.vector()) << "\n";
    IterDomain* cat_id =
        inp_tv->getLogicalDomain().at(updated_cat_op->concatenatedDim());
    std::vector<TensorView*> tvs_to_schedule;
    tvs_to_schedule.reserve(inp_dep_set.size() - 1);
    std::copy_if(
        inp_dep_set.vector().begin(),
        inp_dep_set.vector().end(),
        std::back_inserter(tvs_to_schedule),
        [inp_tv](TensorView* tv) { return tv != inp_tv; });
    std::cerr << "Scheduling: " << toDelimitedString(tvs_to_schedule) << "\n";
    scheduler_tools::scheduleLoopDomainsLike(tvs_to_schedule, cat_id);
  }
}

bool propagateCatToInputs(Fusion* fusion) {
  const auto exprs = fusion->exprs();
  for (auto expr : exprs) {
    auto cat = dynamic_cast<CatOp*>(expr);
    if (cat == nullptr) {
      continue;
    }

    std::cerr << "propagate cat: " << cat->toString();
    propagateCatToInputs(cat);
  }

  // TODO: do error check and return something else if failed
  return true;
}

bool propagateSliceToOutputs(Fusion* fusion) {
  IdModel id_model(fusion, /*build_models=*/false);
  const auto& graph = id_model.buildExactGraph();

  const auto exprs = fusion->exprs();
  for (auto it = exprs.rbegin(); it != exprs.rend(); ++it) {
    auto slice = dynamic_cast<SliceOp*>(*it);
    if (slice == nullptr) {
      continue;
    }

    std::cerr << "propagateSliceToOutputs: " << slice->toString();

    TensorView* out = slice->out();

    auto dep_outputs = DependencyCheck::getAllValsBetween(
        {out}, {fusion->outputs().begin(), fusion->outputs().end()});

    std::vector<TensorView*> tvs_to_schedule;
    tvs_to_schedule.reserve(dep_outputs.size());
    std::transform(
        dep_outputs.begin(),
        dep_outputs.end(),
        std::back_inserter(tvs_to_schedule),
        [](Val* val) { return val->as<TensorView>(); });

    ValGroups cat_ids;
    for (const auto tv : tvs_to_schedule) {
      CatOp* cat_op = dynamic_cast<CatOp*>(tv->definition());
      if (cat_op == nullptr) {
        continue;
      }

      cat_ids.pushBack(graph.toGroup(
          cat_op->output(0)->as<TensorView>()->getLogicalDomain().at(
              cat_op->concatenatedDim())));
    }

    const auto logical_groups = graph.toGroups(out->getLogicalDomain());
    for (const auto i : c10::irange(out->getLogicalDomain().size())) {
      auto logical_id = out->getLogicalDomain().at(i);
      auto resize = dynamic_cast<Resize*>(logical_id->definition());
      if (resize == nullptr) {
        continue;
      }
      auto root_id = resize->in();
      std::cerr << "Slice ID: " << logical_id->toString() << ", "
                << root_id->toString() << "\n";

      auto path_to_reachable_cat_ids = ValGraphBFS::getExprsBetween(
          graph, logical_groups, cat_ids, false, Direction::Forward);
      if (!path_to_reachable_cat_ids.empty() &&
          getInputsOfExprPath(graph, path_to_reachable_cat_ids)
              .has(graph.toGroup(logical_id))) {
        std::cerr << "Skipping as consumed by concat: "
                  << logical_id->toString() << "\n";
        continue;
      }

      std::cerr << "propagate slice: " << root_id->toString() << " of "
                << slice->toString();
      scheduler_tools::scheduleLoopDomainsLike(tvs_to_schedule, root_id);
    }
  }

  // TODO: do error check and return something else if failed
  return true;
}

} // namespace scheduler_tools
} // namespace nvfuser
