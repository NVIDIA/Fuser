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
    // Initialize IDs to traverse. If sub_selection is provided, only
    // traverse IDs that are included in the set are traversed.
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

    // Initialize exprs to traverse. If sub_selection is provided,
    // only traverse exprs that are strictly contained within the provided
    // sub_selection. Exprs are excluded if any of inputs or outputs
    // is not in sub_selection.
    if (sub_selection_.empty()) {
      all_exprs = ExprGroups(
          graph().disjointExprSets().disjointSets().begin(),
          graph().disjointExprSets().disjointSets().end());
    } else {
      for (const IdGroup& id_group : all_ids) {
        for (const ExprGroup& def : graph().getUniqueDefinitions(id_group)) {
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
    for (const ExprGroup& expr_group : all_exprs) {
      if (graph().isTrivialExprGroup(expr_group)) {
        // Expression is just a loop to its current group, ignore
        continue;
      }

      not_inputs.pushBack(graph().outputGroups(expr_group));
      not_outputs.pushBack(graph().inputGroups(expr_group));
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

  auto is_expr_ready = [&](const ExprGroup& expr_group) {
    auto inp_groups = graph().inputGroups(expr_group);
    return std::all_of(
        inp_groups.begin(), inp_groups.end(), [&](IdGroup id_group) {
          return visited_ids.has(id_group) || id_group->empty();
        });
  };

  auto is_id_ready = [&](const IdGroup& id_group) {
    auto unique_defs = graph().getUniqueDefinitions(id_group);
    return std::all_of(
        unique_defs.begin(), unique_defs.end(), [&](ExprGroup expr_group) {
          return expr_group->empty() || visited_exprs.has(expr_group) ||
              graph().isTrivialExprGroup(expr_group);
        });
  };

  while (!to_visit_ids.empty() || !to_visit_exprs.empty()) {
    // Process expressions first as all definitions of iter domains have to be
    // processed before we can process that iter domain.

    // Detect if nothing has been processed which would put us in an infinite
    // loop
    bool something_was_processed = false;
    ExprGroups still_to_visit_exprs;

    while (!to_visit_exprs.empty()) {
      ExprGroup current_expr_group = to_visit_exprs.popFront();
      NVF_ERROR(!current_expr_group->empty());
      if (visited_exprs.has(current_expr_group)) {
        continue;
      }

      if (is_expr_ready(current_expr_group)) {
        handle(current_expr_group);

        something_was_processed = true;
        visited_exprs.pushBack(current_expr_group);

        to_visit_ids.pushBack(graph().outputGroups(current_expr_group));
      } else {
        still_to_visit_exprs.pushBack(current_expr_group);
      }
    }

    std::swap(to_visit_exprs, still_to_visit_exprs);

    IdGroups still_to_visit_ids;
    while (!to_visit_ids.empty()) {
      auto current_id_group = to_visit_ids.popFront();
      NVF_ERROR(!current_id_group->empty());
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
            (to_visit_ids.empty() && to_visit_exprs.empty()),
        "Infinite loop entered.");
  }
}
} // namespace nvfuser
