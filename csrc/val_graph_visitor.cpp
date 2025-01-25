// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <val_graph_visitor.h>

#include <id_model/to_string.h>

#include <variant>

namespace nvfuser {

bool ValGraphVisitor::traverse() {
  if (graph().disjointValSets().size() == 0) {
    return true;
  }
  //const ValGroups terminating_inputs =
  //graph().getTerminatingInputs();
  const ValGroups terminating_inputs = starting_groups_;

  // If no terminating input is found, that should mean there's a
  // cycle.
  if (terminating_inputs.empty()) {
    std::stringstream ss;
    ss << "Unsupported graph: No terminating input found, likely a cyclic graph: ";
    ss << graph().toString();
    error_message_ = ss.str();
    return false;
  }

  {
    std::cerr << "Inputs:\n";
    for (const auto& g: terminating_inputs) {
      std::cerr << nvfuser::toString(g) << "\n";
    }
  }


  std::deque<ValGroup> to_visit_vals(
      terminating_inputs.begin(), terminating_inputs.end());
  ValGroups visited_vals;

  std::deque<ExprGroup> to_visit_exprs;
  ExprGroups visited_exprs;

  auto is_expr_ready = [&](const ExprGroup& expr_group) -> bool {
    const auto inp_groups = graph().inputGroups(expr_group);
    return std::all_of(
        inp_groups.begin(), inp_groups.end(), [&](ValGroup val_group) {
          return visited_vals.has(val_group) || val_group->empty();
        });
  };

  // If any input of the def expr is mapped with the val
  // group itself, i.e., a trivial expr, allow visiting the
  // val group first. The trivial expr group will be visited
  // after the val group.
  //
  // Example:
  //
  // [i0, 1]
  // merge
  // [i0*1]
  // map i0 and i0*1
  // ValGroups: {{i0, i0*1}, {1}}
  //
  // Then, {i0, i0*1} and {1} would be visited first, then the merge
  // expr group would be visited. {i0, i0*1} is also an output group
  // of the merge but since it's already in the visited set, it would
  // not be visited again.
  //
  // See also IdModelTest.ValGraphStmtSort3 for a concrete example.
  auto is_val_ready = [&](const ValGroup& val_group) -> bool {
    if (terminating_inputs.has(val_group)) {
      return true;
    }
    const ExprGroups& unique_defs = graph().getDefinitions(val_group);
    return std::all_of(
        unique_defs.begin(), unique_defs.end(), [&](ExprGroup expr_group) {
          if (expr_group->empty() || visited_exprs.has(expr_group)) {
            return true;
          }

          if (!allow_cycle_) {
            return false;
          }

          auto reachable_nodes = getReachableNodesFrom<ValGraphPermissiveBFS>(
              {expr_group}, {val_group}, Direction::Backward,
              graph());
          if (!reachable_nodes.empty()) {
            // cycle. ignore
            std::cerr << "Cycle detected. Should be safe to ignore. "
                      << nvfuser::toString(val_group)
                      << ", " << nvfuser::toString(expr_group)
                      << "\n";
            return true;
          }
          
          return false;
        });
  };

  // Detect if nothing has been processed which would put us in an infinite
  // loop
  bool something_was_processed = false;

  do {
    something_was_processed = false;

    // Process expressions first as all definitions of vals have to be
    // processed before we can process that val.

    while (!to_visit_exprs.empty()) {
      ExprGroup current_expr_group = to_visit_exprs.front();
      to_visit_exprs.pop_front();
      NVF_ERROR(!current_expr_group->empty());
      if (visited_exprs.has(current_expr_group)) {
        continue;
      }

      if (is_expr_ready(current_expr_group)) {
        handle(current_expr_group);

        something_was_processed = true;
        visited_exprs.pushBack(current_expr_group);

        for (const ValGroup& output_group :
             graph().outputGroups(current_expr_group)) {
          to_visit_vals.push_back(output_group);
        }
      }
    }

    std::deque<ValGroup> still_to_visit_vals;
    while (!to_visit_vals.empty()) {
      auto current_val_group = to_visit_vals.front();
      to_visit_vals.pop_front();
      NVF_ERROR(!current_val_group->empty());
      if (visited_vals.has(current_val_group)) {
        continue;
      }

      if (is_val_ready(current_val_group)) {
        handle(current_val_group);

        something_was_processed = true;
        visited_vals.pushBack(current_val_group);

        for (const ExprGroup& use_group : graph().getUses(current_val_group)) {
          to_visit_exprs.push_back(use_group);
        }
      } else {
        still_to_visit_vals.push_back(current_val_group);
      }
    }

    std::swap(to_visit_vals, still_to_visit_vals);

  } while (something_was_processed);

  if (!to_visit_vals.empty()) {
    std::stringstream ss;
    ss << "The graph has an infinite loop. The following Vals should be visited but are never ready:";
    for (const ValGroup& vg : to_visit_vals) {
      ss << " " << nvfuser::toString(vg);
    }
    error_message_ = ss.str();
    return false;
  }

  if (!to_visit_exprs.empty()) {
    std::stringstream ss;
    ss << "The graph has an infinite loop. The following Exprs should be visited but are never ready:";
    for (const ExprGroup& eg : to_visit_exprs) {
      ss << " " << nvfuser::toString(eg);
    }
    error_message_ = ss.str();
    return false;
  }

#if 0
  // If not all exprs are visited, that should mean there must be a
  // cyclic subgraph. The subgraph should have no terminating input,
  // so it should not be visited at all. Note that some Val groups may
  // not be visited, which should be fine.
  if (visited_exprs.size() != graph().disjointExprSets().size()) {
    std::stringstream ss;
    ss << "The graph has an infinite loop. The following Exprs should be visited but are never ready:";
    for (const ExprGroup& eg : graph().disjointExprSets().disjointSets()) {
      if (!visited_exprs.has(eg)) {
        ss << " " << nvfuser::toString(eg);
      }
    }
    error_message_ = ss.str();
    graph().dumpGraphvizDotGraph("stmt_sort_cycle.dot");
    return false;
  }
#endif

  return true;
}

} // namespace nvfuser
