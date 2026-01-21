// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/schedule.h>
#include <val_graph.h>

namespace nvfuser {

IterDomain* representativeId(const ValGroup& vg) {
  IterDomain* rep = nullptr;

  auto preferNewIterType = [&rep](IterDomain* new_id) {
    if (rep->isReduction()) {
      return new_id->isBroadcast() || new_id->isIteration();
    } else if (rep->isBroadcast()) {
      return new_id->isIteration();
    } else if (rep->isIteration()) {
      return false;
    } else {
      // Prefer anything else to a non-iter, non-bcast, non-reduction ID
      return true;
    }
  };

  for (Val* v : *vg) {
    if (auto id = dynamic_cast<IterDomain*>(v); id &&
        (rep == nullptr || preferNewIterType(id) ||
         (id->getIterType() == rep->getIterType() &&
          id->name() < rep->name()))) {
      rep = id;
      continue;
    }
  }

  NVF_ERROR(rep != nullptr);
  return rep;
}

ValGroup merge(ValGraph* graph, const ValGroup& g0, const ValGroup& g1) {
  NVF_ERROR(!g0->empty() && !g1->empty(), "ValGroup can not be empty");
  auto g0_id = representativeId(g0);
  auto g1_id = representativeId(g1);
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
  graph->initializeVal(g0_id, g0);
  graph->initializeVal(g1_id, g1);
  graph->initializeVal(output_id, {}, {});
  graph->registerExpr(output_id->definition());
  return graph->toGroup(output_id);
}

std::pair<ValGroup, ValGroup> split(
    ValGraph* graph,
    const ValGroup& g,
    Val* factor,
    bool inner_split) {
  NVF_ERROR(!g->empty(), "ValGroup can not be empty");
  auto g_id = representativeId(g);
  NVF_ERROR(
      graph->hasGroup(g_id) && graph->toGroup(g_id) == g,
      "Invalid g given: g is not in the given ValGraph");
  // If there is already an existing split in the ValGraph, just use it.
  auto g_uses = graph->getUses(g);
  for (const ExprGroup& use : g_uses) {
    if (!use->front()->isA<Split>()) {
      continue;
    }
    auto input_groups = graph->inputGroups(use);
    NVF_ERROR(input_groups.size() == 1);
    if (input_groups != std::vector<ValGroup>{g}) {
      continue;
    }
    bool split_match = false;
    for (auto expr : *use) {
      if (auto split = dynamic_cast<Split*>(expr); split != nullptr &&
          split->innerSplit() == inner_split &&
          split->factor()->sameAs(factor)) {
        split_match = true;
        break;
      }
    }
    if (!split_match) {
      continue;
    }
    auto output_groups = graph->outputGroups(use);
    NVF_ERROR(output_groups.size() == 2);
    return {output_groups[0], output_groups[1]};
  }
  // There is no such split, then create one
  g_id = g_id->cloneWithoutRFactor();
  auto [outer_id, inner_id] = IterDomain::split(g_id, factor, inner_split);
  graph->initializeVal(g_id, g);
  graph->initializeVal(outer_id, {}, {});
  graph->initializeVal(inner_id, {}, {});
  graph->registerExpr(inner_id->definition());
  return {graph->toGroup(outer_id), graph->toGroup(inner_id)};
}

std::pair<ValGroup, ValGroup> split(
    ValGraph* graph,
    const ValGroup& g,
    int64_t factor,
    bool inner_split) {
  return split(
      graph,
      g,
      IrBuilder::createInContainer<Val>(
          g->front()->fusion(), factor, DataType::Index),
      inner_split);
}

std::pair<ValGroup, ValGroup> swizzle(
    ValGraph* graph,
    SwizzleType swizzle_type,
    const ValGroup& g0,
    const ValGroup& g1) {
  NVF_ERROR(!g0->empty() && !g1->empty(), "ValGroup can not be empty");
  auto g0_id = representativeId(g0);
  auto g1_id = representativeId(g1);
  NVF_ERROR(
      graph->hasGroup(g0_id) && graph->toGroup(g0_id) == g0,
      "Invalid g0 given: g0 is not in the given ValGraph");
  NVF_ERROR(
      graph->hasGroup(g1_id) && graph->toGroup(g1_id) == g1,
      "Invalid g1 given: g1 is not in the given ValGraph");
  // If there is already an existing swizzle in the ValGraph, just use it.
  auto g0_uses = graph->getUses(g0);
  for (const ExprGroup& use : g0_uses) {
    auto swizzle = dynamic_cast<Swizzle*>(use->front());
    if (swizzle == nullptr || swizzle->swizzleType() != swizzle_type) {
      continue;
    }
    auto input_groups = graph->inputGroups(use);
    NVF_ERROR(input_groups.size() == 2);
    if (input_groups == std::vector<ValGroup>{g0, g1}) {
      auto output_groups = graph->outputGroups(use);
      NVF_ERROR(output_groups.size() == 2);
      return {output_groups[0], output_groups[1]};
    }
  }
  // There is no such merge, then create one
  g0_id = g0_id->cloneWithoutRFactor();
  g1_id = g1_id->cloneWithoutRFactor();
  auto [out_x, out_y] = IterDomain::swizzle(swizzle_type, g0_id, g1_id);
  graph->initializeVal(g0_id, g0);
  graph->initializeVal(g1_id, g1);
  graph->initializeVal(out_x, {}, {});
  graph->initializeVal(out_y, {}, {});
  graph->registerExpr(out_x->definition());
  return {graph->toGroup(out_x), graph->toGroup(out_y)};
}

} // namespace nvfuser
