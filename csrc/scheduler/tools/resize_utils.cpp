// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <scheduler/tools/loop_domain_scheduler.h>
#include <scheduler/tools/resize_utils.h>
#include <iter_visitor.h>
#include <ir/utils.h>

namespace nvfuser {
namespace scheduler_tools {

bool propagateResizeToCatInputs(CatOp* cat_op) {
  Fusion* fusion = cat_op->fusion();

  DisjointSets<TensorView*> input_sets;

  for (auto inp_tv : ir_utils::filterByType<TensorView>(cat_op->inputs())) {
    std::cerr << "Cat input: " << inp_tv->toString() << "\n";
    if (input_sets.mappingExists(inp_tv)) {
      // Overlapped cat inputs
      std::cerr << "Overlapped input: " << inp_tv->toString() << "\n";
      return false;
    }
    DisjointSets<TensorView*>::DisjointSet& input_set = input_sets.initializeSet(inp_tv).first->second;
    auto dep_inputs = DependencyCheck::getAllValsBetween(
        {fusion->inputs().begin(), fusion->inputs().end()}, {inp_tv});
    dep_inputs.erase(
        std::remove(dep_inputs.begin(), dep_inputs.end(), inp_tv),
        dep_inputs.end());
    std::cerr << "Dep input: " << toDelimitedString(dep_inputs) << "\n";
    for (auto tv : ir_utils::filterByType<TensorView>(dep_inputs)) {
      if (input_sets.mappingExists(tv)) {
        // Overlapped cat inputs
        std::cerr << "Overlapped input: " << tv->toString() << "\n";
        return false;
      }
      input_sets.appendToSet(tv, input_set);
    }
  }

  std::cerr << "Num disjoint sets: " << input_sets.size() << "\n";

  NVF_ERROR(input_sets.size() <= cat_op->inputs().size());

  if (input_sets.size() < cat_op->inputs().size()) {
    // Overlapped inputs are detected.
    return false;
  }

  std::cerr << "Propagating cat resizes to each disjoint set\n";
  
  for (auto inp_tv : ir_utils::filterByType<TensorView>(cat_op->inputs())) {
    std::cerr << "Cat input: " << inp_tv->toString() << "\n";
    const auto& inp_dep_set = input_sets.getDisjointSetOf(inp_tv);
    scheduler_tools::scheduleLoopDomainsLike(
        inp_dep_set.vector(), inp_tv->getLogicalDomain());
  }
  
  return false;
}

} // namespace scheduler_tools
} // namespace nvfuser
