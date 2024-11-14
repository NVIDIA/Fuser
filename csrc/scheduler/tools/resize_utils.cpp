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
    // Needs to be privatized if dep tensors are used by non-cat
    // consumers. For now, just always privatize
    if (true || overlap) {
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

// Merge this with propagateCatToInputs?
void propagateSliceToInputs(SliceOp* resize_op) {
  Fusion* fusion = resize_op->fusion();

  DisjointSets<TensorView*> input_sets;

  std::cerr << "propagateSliceToInputs: " << resize_op->toString();

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

  // TODO: Avoid privatize if the dep chain is exclusively used by input
  auto privatize_input = [&](TensorView* input) -> TensorView* {
    auto private_copy = RecomputeTv::recompute(input);
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

  auto has_overlap = [&input_sets](TensorView* input) -> bool {
    Fusion* fusion = input->fusion();
    std::cerr << "Input: " << input->toString() << "\n";
    if (input_sets.mappingExists(input)) {
      // Overlapped inputs
      std::cerr << "Overlapped input: " << input->toString() << "\n";
      return true;
    }
    DisjointSets<TensorView*>::DisjointSet& input_set =
        input_sets.initializeSet(input).first->second;
    auto dep_inputs = DependencyCheck::getAllValsBetween(
        {fusion->inputs().begin(), fusion->inputs().end()}, {input});
    dep_inputs.erase(
        std::remove_if(
            dep_inputs.begin(),
            dep_inputs.end(),
            [&](Val* tv) { return tv->isFusionInput(); }),
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

  auto inp_tv = resize_op->input(0)->as<TensorView>();
  std::cerr << "Input: " << inp_tv->toString() << "\n";

  if (inp_tv->isFusionInput()) {
    return;
  }

  TensorView* original = inp_tv;
  TensorView* clone = nullptr;
  [[maybe_unused]] bool overlap = has_overlap(inp_tv);
  // Needs to be privatized if dep tensors are used by non-cat
  // consumers. For now, just always privatize
  clone = privatize_input(inp_tv);

  auto updated_op = resize_op;
  std::cerr << "Replacing " << original->toString() << " with "
            << clone->toString() << "\n";
  updated_op = ir_utils::replaceValInExprInputs(updated_op, original, clone)
                   ->as<SliceOp>();

  std::cerr << "New op: " << updated_op->toString();
  std::cerr << "Num disjoint sets: " << input_sets.size() << "\n";
  std::cerr << "Propagating slice resizes to each disjoint set\n";

  const auto& inp_dep_set = input_sets.getDisjointSetOf(clone);
  std::cerr << "Dep: " << toDelimitedString(inp_dep_set.vector()) << "\n";
  auto out_tv = updated_op->output(0)->as<TensorView>();
  for (const auto i : c10::irange(out_tv->getLogicalDomain().size())) {
    auto slice_out_id = out_tv->getLogicalDomain().at(i);
    auto slice = dynamic_cast<Resize*>(slice_out_id->definition());
    if (slice == nullptr) {
      continue;
    }

    auto tvs_to_schedule = inp_dep_set.vector();
    for (auto& tv : tvs_to_schedule) {
      if (tv == original) {
        tv = clone;
      }
    }

    std::cerr << "Scheduling " << toDelimitedString(tvs_to_schedule) << " with "
              << slice_out_id->toString() << "\n";
    scheduler_tools::scheduleLoopDomainsLike(tvs_to_schedule, slice_out_id);
  }
}

bool propagateSliceToInputs(Fusion* fusion) {
  const auto exprs = fusion->exprs();
  for (auto expr : exprs) {
    if (auto slice = dynamic_cast<SliceOp*>(expr)) {
      propagateSliceToInputs(slice);
    }
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

bool propagateSqueezedSliceToOutputs(Fusion* fusion) {
  std::cerr << "propagateSqueezedSliceToOutputs\n";

  fusion->printMath();
  std::cout << std::endl;

  IdModel id_model(fusion, /*build_models=*/false);
  const auto& graph = id_model.buildExactGraph();

  auto squeezed_slices = ir_utils::getSqueezedSlices(fusion);
  std::unordered_set<IterDomain*> squeezed_slice_set{
      squeezed_slices.begin(), squeezed_slices.end()};

  std::cerr << "All squeezed slices: " << toDelimitedString(squeezed_slices)
            << "\n";

  // Each tensor should not need to be updated multiple times
  std::unordered_set<TensorView*> already_updated;

  const auto exprs = fusion->exprs();
  for (auto it = exprs.rbegin(); it != exprs.rend(); ++it) {
    auto slice = dynamic_cast<SliceOp*>(*it);
    if (slice == nullptr) {
      continue;
    }

    std::cerr << "Slice: " << slice->toString();

    auto slice_out = slice->output(0)->as<TensorView>();

    auto dep_outputs = DependencyCheck::getAllValsBetween(
        {slice_out}, {fusion->outputs().begin(), fusion->outputs().end()});

    for (const auto logical_id : slice_out->getLogicalDomain()) {
      if (squeezed_slice_set.count(logical_id) == 0) {
        continue;
      }

      std::cerr << "propagateSqueezedSliceToOutputs: slice candidate: "
                << slice->toString() << ", " << logical_id->toString() << "\n";

      std::cerr << "All dep outputs: " << toDelimitedString(dep_outputs)
                << std::endl;

      // squeezed slice found
      // Assume this ID remains in the loop domain
      NVF_ERROR(
          std::find(
              slice_out->getLoopDomain().begin(),
              slice_out->getLoopDomain().end(),
              logical_id) != slice_out->getLoopDomain().end());

      std::vector<TensorView*> tvs_to_schedule;
      tvs_to_schedule.reserve(dep_outputs.size());
      for (Val* dep_output : dep_outputs) {
        auto tv = dep_output->as<TensorView>();
        if (std::find_if(
                tv->getLogicalDomain().begin(),
                tv->getLogicalDomain().end(),
                [&](IterDomain* id) {
                  return graph.disjointValSets().strictAreMapped(
                      id, logical_id);
                }) != tv->getLogicalDomain().end()) {
          // Not yet squeezed
          std::cerr << "Not yet squeezed: " << tv->toString() << "\n";
          continue;
        }
        tvs_to_schedule.push_back(tv);
      }

      if (tvs_to_schedule.empty()) {
        continue;
      }

      std::cerr << "propagate squeezed slice: " << logical_id->toString()
                << " of " << slice->toString();
      std::cerr << "To tensors: " << toDelimitedString(tvs_to_schedule) << "\n";

      [[maybe_unused]] auto has_no_overlap = std::all_of(
          tvs_to_schedule.begin(),
          tvs_to_schedule.end(),
          [&](TensorView* tv_to_schedule) {
            if (already_updated.count(tv_to_schedule)) {
              std::cerr << "Already updated: " << tv_to_schedule->toString()
                        << "\n";
            }
            return already_updated.count(tv_to_schedule) == 0;
          });

      // NVF_ERROR(has_no_overlap);

      scheduler_tools::scheduleLoopDomainsLike(tvs_to_schedule, logical_id);

      already_updated.insert(tvs_to_schedule.begin(), tvs_to_schedule.end());
    }
  }

  std::cout << "propagateSqueezedSliceToOutputs done\n";
  fusion->printMath();
  std::cout << std::endl;

  // TODO: do error check and return something else if failed
  return true;
}

void propagatePadToInputs(PadOp* pad_op) {
  // Fusion* fusion = pad_op->fusion();
}

void propagateResizeTensorOpToInputs(Expr* resize_op) {
  DebugStreamGuard dsg(std::cerr);

  NVF_ERROR(
      resize_op->isA<SliceOp>() || resize_op->isA<PadOp>(),
      "Unexpected resize tensor op: ",
      resize_op->toString());

  Fusion* fusion = resize_op->fusion();

  DisjointSets<TensorView*> input_sets;

  std::cerr << "propagateResizeTensorOpToInputs: " << resize_op->toString();

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

  // TODO: Avoid privatize if the dep chain is exclusively used by input
  auto privatize_input = [&](TensorView* input) -> TensorView* {
    auto private_copy = RecomputeTv::recompute(input);
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

  auto has_overlap = [&input_sets](TensorView* input) -> bool {
    Fusion* fusion = input->fusion();
    std::cerr << "Input: " << input->toString() << "\n";
    if (input_sets.mappingExists(input)) {
      // Overlapped inputs
      std::cerr << "Overlapped input: " << input->toString() << "\n";
      return true;
    }
    DisjointSets<TensorView*>::DisjointSet& input_set =
        input_sets.initializeSet(input).first->second;
    auto dep_inputs = DependencyCheck::getAllValsBetween(
        {fusion->inputs().begin(), fusion->inputs().end()}, {input});
    dep_inputs.erase(
        std::remove_if(
            dep_inputs.begin(),
            dep_inputs.end(),
            [&](Val* tv) { return tv->isFusionInput(); }),
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

  auto inp_tv = resize_op->input(0)->as<TensorView>();
  std::cerr << "Input: " << inp_tv->toString() << "\n";

  if (inp_tv->isFusionInput()) {
    return;
  }

  TensorView* original = inp_tv;
  TensorView* clone = nullptr;
  [[maybe_unused]] bool overlap = has_overlap(inp_tv);
  // Needs to be privatized if dep tensors are used by non-cat
  // consumers. For now, just always privatize
  clone = privatize_input(inp_tv);

  Expr* updated_op = resize_op;
  std::cerr << "Replacing " << original->toString() << " with "
            << clone->toString() << "\n";
  updated_op = ir_utils::replaceValInExprInputs(updated_op, original, clone);

  std::cerr << "New op: " << updated_op->toString();
  std::cerr << "Num disjoint sets: " << input_sets.size() << "\n";
  std::cerr << "Propagating slice resizes to each disjoint set\n";

  fusion->printMath();

  const auto& inp_dep_set = input_sets.getDisjointSetOf(clone);
  std::cerr << "Dep: " << toDelimitedString(inp_dep_set.vector()) << "\n";
  auto out_tv = updated_op->output(0)->as<TensorView>();

  auto tvs_to_schedule = inp_dep_set.vector();
  for (auto& tv : tvs_to_schedule) {
    if (tv == original) {
      tv = clone;
    }
  }

  std::cerr << "Propagating pre-resize producer loop domain "
            << toDelimitedString(tvs_to_schedule) << " with "
            << clone->toString() << "\n";

  scheduler_tools::scheduleLoopDomainsLike(
      tvs_to_schedule, clone->getLoopDomain());

  for (const auto i : c10::irange(out_tv->getLogicalDomain().size())) {
    auto out_logical_id = out_tv->getLogicalDomain().at(i);
    auto resize = dynamic_cast<Resize*>(out_logical_id->definition());
    if (resize == nullptr) {
      continue;
    }

    std::cerr << "Scheduling " << toDelimitedString(tvs_to_schedule) << " with "
              << out_logical_id->toString() << "\n";
    // scheduler_tools::scheduleLoopDomainsLike(tvs_to_schedule,
    // out_logical_id);
    scheduler_tools::scheduleLoopDomainsBy(tvs_to_schedule, resize);
  }
}

} // namespace scheduler_tools
} // namespace nvfuser
