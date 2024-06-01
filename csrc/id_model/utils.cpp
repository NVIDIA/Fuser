// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/utils.h>
#include <val_graph.h>

namespace nvfuser {

ValGroup merge(ValGraph* graph, const ValGroup& g0, const ValGroup& g1) {
  NVF_ERROR(g0->size() > 0, "ValGroup can not be empty");
  NVF_ERROR(g1->size() > 0, "ValGroup can not be empty");
  auto g0_id = g0->front()->as<IterDomain>();
  auto g1_id = g1->front()->as<IterDomain>();
  NVF_ERROR(
      graph->hasGroup(g0_id) && graph->toGroup(g0_id) == g0,
      "Invalid g0 given: g0 is not in the given ValGraph");
  NVF_ERROR(
      graph->hasGroup(g1_id) && graph->toGroup(g1_id) == g1,
      "Invalid g1 given: g1 is not in the given ValGraph");
  // If there is already an existing merge in the ValGraph, just use it.
  auto g0_uses = graph->getUses(g0);
  for (const ExprGroup& use : g0_uses) {
    if (!use->front()->isA<Merge>()) {
      continue;
    }
    auto input_groups = graph->inputGroups(use);
    NVF_ERROR(input_groups.size() == 2);
    if (input_groups == std::vector<ValGroup>{g0, g1}) {
      auto output_groups = graph->outputGroups(use);
      NVF_ERROR(output_groups.size() == 1);
      return output_groups[0];
    }
  }
  // There is no such merge, then create one
  g0_id = g0_id->cloneWithoutRFactor();
  g1_id = g1_id->cloneWithoutRFactor();
  auto output_id = IterDomain::merge(g0_id, g1_id);
  graph->initializeVal(g0_id, {}, {});
  graph->initializeVal(g1_id, {}, {});
  graph->mapVals(g0->front(), g0_id);
  graph->mapVals(g1->front(), g1_id);
  graph->initializeVal(output_id, {}, {});
  graph->registerExpr(output_id->definition());
  return graph->toGroup(output_id);
}

} // namespace nvfuser
