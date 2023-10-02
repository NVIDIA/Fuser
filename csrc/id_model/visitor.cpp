// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/visitor.h>

namespace nvfuser {

void IdGraphVisitor::traverse() {
  IdGroups all_ids;
  ExprGroups all_exprs;
  {
    if (sub_selection_.empty()) {
      all_ids = IdGroups(
          graph().disjointIdSets().disjointSets().begin(),
          graph().disjointIdSets().disjointSets().end());
    } else {
      for (auto id : sub_selection_) {
        if (graph().hasGroup(id)) {
          all_ids.pushBack(graph().toGroup(id));
        }
      }
    }

    if (sub_selection_.empty()) {
      all_exprs = ExprGroups(
          graph().disjointExprSets().disjointSets().begin(),
          graph().disjointExprSets().disjointSets().end());
    } else {
      for (auto id_group : all_ids) {
        for (auto def : graph().getUniqueDefinitions(id_group)) {
          if (all_exprs.has(def)) {
            continue;
          }
          auto inp_groups = IdGroups(graph().inputGroups(def));
          auto out_groups = IdGroups(graph().outputGroups(def));
          if (inp_groups.subtract(all_ids).empty() &&
              out_groups.subtract(all_ids).empty()) {
            all_exprs.pushBack(def);
          }
        }
      }
    }
  }
  // There could be IterDomains in from or to that are between other from and
  // to nodes. Make sure to clear those out.
  IdGroups terminating_inputs;
  IdGroups terminating_outputs;

  {
    IdGroups not_inputs;
    IdGroups not_outputs;
    for (auto expr_group : all_exprs) {
      auto inp_groups = IdGroups(graph().inputGroups(expr_group));
      auto out_groups = IdGroups(graph().outputGroups(expr_group));

      if (inp_groups.intersect(out_groups).size() > 0) {
        // Expression is just a loop to its current group, ignore
        continue;
      }

      not_inputs.pushBack(out_groups);
      not_outputs.pushBack(inp_groups);
    }

    terminating_inputs =
        IdGroups(all_ids.begin(), all_ids.end()).subtract(not_inputs);

    terminating_outputs =
        IdGroups(all_ids.begin(), all_ids.end()).subtract(not_outputs);
  }

  IdGroups to_visit_ids = terminating_inputs;
  IdGroups visited_ids;

  ExprGroups to_visit_exprs;
  ExprGroups visited_exprs;

  auto is_expr_ready = [&](ExprGroup expr_group) {
    auto inp_groups = graph().inputGroups(expr_group);
    return std::all_of(
        inp_groups.begin(), inp_groups.end(), [&](IdGroup id_group) {
          return visited_ids.has(id_group) || id_group->empty();
        });
  };

  auto is_id_ready = [&](IdGroup id_group) {
    auto unique_defs = graph().getUniqueDefinitions(id_group);
    return std::all_of(
        unique_defs.begin(), unique_defs.end(), [&](ExprGroup expr_group) {
          return expr_group->empty() || visited_exprs.has(expr_group) ||
              graph().isTrivialExprGroup(expr_group);
        });
  };

  while (to_visit_ids.size() > 0 || to_visit_exprs.size() > 0) {
    // Process expressions first as all definitions of iter domains have to be
    // processed before we can process that iter domain.

    // Detect if nothing has been processed which would put us in an infinite
    // loop
    bool something_was_processed = false;
    ExprGroups still_to_visit_exprs;

    while (to_visit_exprs.size() > 0) {
      auto current_expr_group = to_visit_exprs.popFront();
      if (visited_exprs.has(current_expr_group)) {
        continue;
      }

      if (is_expr_ready(current_expr_group)) {
        handle(current_expr_group);

        something_was_processed = true;
        visited_exprs.pushBack(current_expr_group);

        auto out_groups = graph().outputGroups(current_expr_group);
        for (auto out_group : out_groups) {
          to_visit_ids.pushBack(out_group);
        }
      } else {
        still_to_visit_exprs.pushBack(current_expr_group);
      }
    }

    std::swap(to_visit_exprs, still_to_visit_exprs);

    IdGroups still_to_visit_ids;
    while (to_visit_ids.size() > 0) {
      auto current_id_group = to_visit_ids.popFront();
      if (visited_ids.has(current_id_group)) {
        continue;
      }

      if (is_id_ready(current_id_group)) {
        handle(current_id_group);

        something_was_processed = true;
        visited_ids.pushBack(current_id_group);

        if (!terminating_outputs.has(current_id_group)) {
          auto uses_pair = graph().getUses(current_id_group);
          if (uses_pair.second) {
            to_visit_exprs.pushBack(uses_pair.first);
          }
        }
      } else {
        still_to_visit_ids.pushBack(current_id_group);
      }
    }
    std::swap(to_visit_ids, still_to_visit_ids);

    NVF_ERROR(
        something_was_processed ||
            (to_visit_ids.size() == 0 && to_visit_exprs.size() == 0),
        "Infinite loop entered.");
  }
}
} // namespace nvfuser
