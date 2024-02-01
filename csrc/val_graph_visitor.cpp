// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <val_graph_visitor.h>

namespace nvfuser {

void ValGraphVisitor::traverse() {
  ValGroups all_vals;
  ExprGroups all_exprs;
  {
    // Initialize Vals to traverse. If sub_selection is provided, only
    // traverse Vals that are included in the set are traversed.
    if (sub_selection_.empty()) {
      all_vals = ValGroups(
          graph().disjointValSets().disjointSets().begin(),
          graph().disjointValSets().disjointSets().end());
    } else {
      for (auto val : sub_selection_) {
        if (graph().hasGroup(val)) {
          all_vals.pushBack(graph().toGroup(val));
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
      for (const ValGroup& val_group : all_vals) {
        for (const ExprGroup& def : graph().getDefinitions(val_group)) {
          if (all_exprs.has(def)) {
            continue;
          }
          auto inp_groups = ValGroups(graph().inputGroups(def));
          auto out_groups = ValGroups(graph().outputGroups(def));
          if (inp_groups.computeSubtract(all_vals).empty() &&
              out_groups.computeSubtract(all_vals).empty()) {
            all_exprs.pushBack(def);
          }
        }
      }
    }
  }
  // There could be Vals in from or to that are between other from and
  // to nodes. Make sure to clear those out.
  ValGroups terminating_inputs;
  ValGroups terminating_outputs;

  {
    ValGroups not_inputs;
    ValGroups not_outputs;
    for (const ExprGroup& expr_group : all_exprs) {
      if (graph().isTrivialExprGroup(expr_group)) {
        // Expression is just a loop to its current group, ignore
        continue;
      }

      not_inputs.pushBack(graph().outputGroups(expr_group));
      not_outputs.pushBack(graph().inputGroups(expr_group));
    }

    terminating_inputs = all_vals.computeSubtract(not_inputs);

    terminating_outputs = all_vals.computeSubtract(not_outputs);
  }

  ValGroups to_visit_ids = terminating_inputs;
  ValGroups visited_ids;

  ExprGroups to_visit_exprs;
  ExprGroups visited_exprs;

  auto is_expr_ready = [&](const ExprGroup& expr_group) -> bool {
    auto inp_groups = graph().inputGroups(expr_group);
    return std::all_of(
        inp_groups.begin(), inp_groups.end(), [&](ValGroup id_group) {
          return visited_ids.has(id_group) || id_group->empty();
        });
  };

  auto is_val_ready = [&](const ValGroup& val_group) -> bool {
    const ExprGroups& unique_defs = graph().getDefinitions(val_group);
    return std::all_of(
        unique_defs.begin(), unique_defs.end(), [&](ExprGroup expr_group) {
          return expr_group->empty() || visited_exprs.has(expr_group) ||
              graph().isTrivialExprGroup(expr_group);
        });
  };

  while (!to_visit_ids.empty() || !to_visit_exprs.empty()) {
    // Process expressions first as all definitions of vals have to be
    // processed before we can process that val.

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

    ValGroups still_to_visit_ids;
    while (!to_visit_ids.empty()) {
      auto current_id_group = to_visit_ids.popFront();
      NVF_ERROR(!current_id_group->empty());
      if (visited_ids.has(current_id_group)) {
        continue;
      }

      if (is_val_ready(current_id_group)) {
        handle(current_id_group);

        something_was_processed = true;
        visited_ids.pushBack(current_id_group);

        if (!terminating_outputs.has(current_id_group)) {
          const ExprGroups& uses = graph().getUses(current_id_group);
          to_visit_exprs.pushBack(uses);
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
