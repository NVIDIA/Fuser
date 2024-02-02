// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <val_graph_visitor.h>

#include <id_model/to_string.h>

namespace nvfuser {

void ValGraphVisitor::traverse() {
  ValGroups to_visit_ids = graph().getTerminatingInputs(sub_selection_);
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

        //if (true || !terminating_outputs.has(current_id_group)) {
        if (true) {
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
