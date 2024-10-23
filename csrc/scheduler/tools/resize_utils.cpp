// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/cloner.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <scheduler/tools/loop_domain_scheduler.h>
#include <scheduler/tools/resize_utils.h>

namespace nvfuser {
namespace scheduler_tools {

std::vector<CatOp*> getRepresentativeCatOps(Fusion* fusion) {
  const auto& exprs = fusion->exprs();
#if 0
  std::unordered_set<Expr*> representative_set;

  std::unordered_set<Val*> deps;

  for (auto it = exprs.rbegin(); it != exprs.rend(); ++it) {
    CatOp* cat = dynamic_cast<CatOp*>(*it);
    if (cat == nullptr) {
      continue;
    }

    if (deps.find(cat->output(0)) != deps.end()) {
      continue;
    }

    representative_set.insert(cat);
    auto all_inp_dep = DependencyCheck::getAllValsBetween(
        {fusion->inputs().begin(), fusion->inputs().end()}, {cat->input(0)});
    for (auto val : all_inp_dep) {
      deps.insert(val);
    }
  }

  std::vector<CatOp*> representative_vec;
  for (auto expr : exprs) {
    if (representative_set.find(expr) != representative_set.end()) {
      representative_vec.push_back(expr->as<CatOp>());
    }
  }

  return representative_vec;
#else
  std::vector<CatOp*> cat_ops;
  for (auto expr: exprs) {
    if (expr->isA<CatOp>()) {
      cat_ops.push_back(expr->as<CatOp>());
    }
  }
  return cat_ops;
#endif
}

bool propagateResizeToCatInputs(CatOp* cat_op) {
  Fusion* fusion = cat_op->fusion();

  DisjointSets<TensorView*> input_sets;

  std::cerr << "propagateResizeToCatInputs: " << cat_op->toString();

  auto get_inputs = [&] (Val* tv) -> std::vector<Val*> {
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

  auto privatize_cat_input =
      [&](TensorView* cat_input) -> TensorView* {
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
    dep_inputs.erase(std::remove_if(
        dep_inputs.begin(), dep_inputs.end(), [&](Val* tv) {
          return tv == cat_input || tv->isFusionInput();
        }),
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
    std::cerr << "Replacing " << original->toString() << " with " << clone->toString() << "\n";
    updated_cat_op =
        ir_utils::replaceValInExprInputs(updated_cat_op, original, clone)->as<CatOp>();
  }

  std::cerr << "New cat op: " << updated_cat_op->toString();
  
  std::cerr << "Num disjoint sets: " << input_sets.size() << "\n";

  std::cerr << "Propagating cat resizes to each disjoint set\n";

  for (auto inp_tv : ir_utils::filterByType<TensorView>(updated_cat_op->inputs())) {
    std::cerr << "Cat input: " << inp_tv->toString() << "\n";
    const auto& inp_dep_set = input_sets.getDisjointSetOf(inp_tv);
    scheduler_tools::scheduleLoopDomainsLike(
        inp_dep_set.vector(), inp_tv->getLogicalDomain());
  }

  return true;
}

} // namespace scheduler_tools
} // namespace nvfuser
