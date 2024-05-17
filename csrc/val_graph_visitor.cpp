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

void ValGraphVisitor::traverse() {
  const ValGroups terminating_inputs = graph().getTerminatingInputs();
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
          // Handle ExprGroups that return one or some of its input ValGroups as
          // output. This expr_group is not visited yet, which means there're
          // input ValGroups that are not yet visited. If those not-visited
          // inputs are actually the same as val_group, visit val_group at this
          // point to resolve the circular dependency.
          for (const ValGroup& input_group : graph().inputGroups(expr_group)) {
            if (input_group != val_group && !visited_vals.has(input_group) &&
                input_group->empty()) {
              return false;
            }
          }
          return true;
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
    NVF_ERROR(false, ss.str());
  }

  if (!to_visit_exprs.empty()) {
    std::stringstream ss;
    ss << "The graph has an infinite loop. The following Exprs should be visited but are never ready:";
    for (const ExprGroup& eg : to_visit_exprs) {
      ss << " " << nvfuser::toString(eg);
    }
    NVF_ERROR(false, ss.str());
  }
}

std::ostream& operator<<(std::ostream& os, const Direction direction) {
  switch (direction) {
    case Direction::Forward:
      os << "Forward";
      break;
    case Direction::Backward:
      os << "Backward";
      break;
    case Direction::Undefined:
      os << "Undefined";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const ExprPath& path) {
  for (const auto& [expr_group, direction] : path) {
    NVF_ERROR(!expr_group->empty());
    Expr* expr = expr_group->front();
    NVF_ERROR(expr != nullptr);
    os << direction << " " << expr->toString();
  }
  return os;
}

ExprPath ValGraphBFS::getExprsBetween(
    const ValGraph& graph,
    const ValGroups& from,
    const ValGroups& to) {
  ValGraphBFS bfs(
      graph,
      {from.vector().begin(), from.vector().end()},
      {to.vector().begin(), to.vector().end()});

  bfs.traverse();

  return bfs.getShortestExprPath();
}

namespace {

std::string toString(const ValGraphBFS::GroupType& g) {
  if (const ExprGroup* eg = std::get_if<ExprGroup>(&g)) {
    std::stringstream ss;
    ss << nvfuser::toString(*eg)
       << ", input: " << (*eg)->front()->input(0)->name();
    return ss.str();
  } else if (const ValGroup* vg = std::get_if<ValGroup>(&g)) {
    return nvfuser::toString(*vg);
  } else {
    NVF_ERROR(false);
  }
}

} // namespace

bool ValGraphBFS::allToGroupsVisited() const {
  return std::all_of(
      to_groups_.begin(),
      to_groups_.end(),
      [&](const GroupType& group) -> bool { return isVisited(group); });
};

void ValGraphBFS::traverse() {
  for (const auto& g : from_groups_) {
    setVisited(g);
    addNewNeighbors(g);
  }

  while (!allToGroupsVisited()) {
    bool something_was_processed = false;
    std::deque<GroupType> not_ready_;
    while (!allToGroupsVisited() && !to_visit_.empty()) {
      const auto g = to_visit_.front();
      to_visit_.pop_front();

      if (isVisited(g)) {
        continue;
      }

      auto ready_direction = isReady(g);
      if (!ready_direction.has_value()) {
        // To stop an infinite loop, the not-ready group is not moved
        // back to the to_visit_ queue but kept in the separate
        // queue. This way, if all groups in to_visit_ are not ready,
        // the queue would eventually become empty, which would then
        // break the inner while loop. The something_was_processed
        // flag is used to remember if there's any progress.
        not_ready_.emplace_back(g);
        continue;
      }

      // Visit this group and add its neighbors to to_visit if not
      // visited yet
      setVisited(g);
      setPrevGroups(g, *ready_direction);
      addNewNeighbors(g);
      something_was_processed = true;
    }

    // If nothing was processed, break out of the loop
    if (!something_was_processed) {
      break;
    }

    // Something was processed. Redo the traversal.
    to_visit_.insert(to_visit_.end(), not_ready_.begin(), not_ready_.end());
  };

  if (!allToGroupsVisited()) {
    std::stringstream ss;
    for (const auto& to_group : to_groups_) {
      if (!isVisited(to_group)) {
        ss << " " << toString(to_group);
        if (const ExprGroup* eg = std::get_if<ExprGroup>(&to_group)) {
          ss << " " << (*eg)->front()->toString();
        }
      }
    }
    NVF_ERROR(false, "BFS traversal could not visit some nodes: ", ss.str());
  }
}

std::optional<std::pair<Direction, std::vector<ValGraphBFS::GroupType>>>
ValGraphBFS::isReady(const GroupType& group) const {
  if (const ExprGroup* eg = std::get_if<ExprGroup>(&group)) {
    return isReady(*eg);
  } else if (const ValGroup* vg = std::get_if<ValGroup>(&group)) {
    return isReady(*vg);
  } else {
    NVF_ERROR(false);
  }
}

std::optional<std::pair<Direction, std::vector<ValGraphBFS::GroupType>>>
ValGraphBFS::isReady(const ExprGroup& expr_group) const {
  // Either all inputs or all outputs must have been visited
  auto inputs = graph_.inputGroups(expr_group);
  if (!inputs.empty() &&
      std::all_of(
          inputs.begin(), inputs.end(), [&](const ValGroup& input) -> bool {
            return isDependencySatisfied(input);
          })) {
    std::vector<GroupType> prev_groups;
    std::copy_if(
        inputs.begin(),
        inputs.end(),
        std::back_inserter(prev_groups),
        [&](const ValGroup& input) -> bool { return isVisited(input); });
    return std::make_pair(Direction::Forward, prev_groups);
  }

  auto outputs = graph_.outputGroups(expr_group);
  if (!outputs.empty() &&
      std::all_of(
          outputs.begin(), outputs.end(), [&](const ValGroup& output) -> bool {
            return isDependencySatisfied(output);
          })) {
    std::vector<GroupType> prev_groups;
    std::copy_if(
        outputs.begin(),
        outputs.end(),
        std::back_inserter(prev_groups),
        [&](const ValGroup& output) -> bool { return isVisited(output); });

    return std::make_pair(Direction::Backward, prev_groups);
  }

  return std::nullopt;
}

std::optional<std::pair<Direction, std::vector<ValGraphBFS::GroupType>>>
ValGraphBFS::isReady(const ValGroup& val_group) const {
  // In the case of Val, requires one def or use expr.
  // Check if any use is visited
  if (!graph_.getUses(val_group).empty()) {
    auto it = std::find_if(
        graph_.getUses(val_group).begin(),
        graph_.getUses(val_group).end(),
        [&](const ExprGroup& use_eg) -> bool {
          return isDependencySatisfied(use_eg);
        });
    if (it != graph_.getUses(val_group).end()) {
      return std::make_pair(Direction::Backward, std::vector<GroupType>{*it});
    }
  }
  // Check if all defs are visited
  if (!graph_.getDefinitions(val_group).empty()) {
    auto it = std::find_if(
        graph_.getDefinitions(val_group).begin(),
        graph_.getDefinitions(val_group).end(),
        [&](const ExprGroup& def_eg) -> bool {
          return isDependencySatisfied(def_eg);
        });
    if (it != graph_.getDefinitions(val_group).end()) {
      return std::make_pair(Direction::Forward, std::vector<GroupType>{*it});
    }
  }

  return std::nullopt;
}

bool ValGraphBFS::isDependencySatisfied(const GroupType& group) const {
  return isVisited(group);
}

bool ValGraphBFS::isVisited(const GroupType& g) const {
  return visited_.find(g) != visited_.end();
}

void ValGraphBFS::setVisited(const GroupType& g) {
  visited_.emplace(g);
}

void ValGraphBFS::addNewNeighbors(const GroupType& g) {
  auto add_to_visit_list = [&](const GroupType& g) -> void {
    if (!isVisited(g) && excludeFromTraversal(g)) {
      return;
    }
    to_visit_.emplace_back(g);
  };

  if (const ExprGroup* eg = std::get_if<ExprGroup>(&g)) {
    for (const auto& vg : graph_.inputGroups(*eg)) {
      add_to_visit_list(vg);
    }
    for (const auto& vg : graph_.outputGroups(*eg)) {
      add_to_visit_list(vg);
    }
  } else if (const ValGroup* vg = std::get_if<ValGroup>(&g)) {
    for (const auto& eg : graph_.getUses(*vg)) {
      add_to_visit_list(eg);
    }
    for (const auto& eg : graph_.getDefinitions(*vg)) {
      add_to_visit_list(eg);
    }
  } else {
    NVF_ERROR(false);
  }
}

void ValGraphBFS::setPrevGroups(
    const GroupType& group,
    const std::pair<Direction, std::vector<GroupType>>& prev_groups) {
  NVF_ERROR(
      prev_groups_.emplace(group, prev_groups).second,
      "Previous group already set for ",
      toString(group));
}

} // namespace nvfuser

namespace std {
template <>
struct hash<pair<nvfuser::ExprGroup, nvfuser::Direction>> {
  std::size_t operator()(
      const std::pair<nvfuser::ExprGroup, nvfuser::Direction>& directed_group)
      const {
    using std::hash;
    return hash<nvfuser::ExprGroup>()(directed_group.first);
  }
};
} // namespace std

namespace nvfuser {

// Going backwards from to_groups_ to from_groups_ to discover the
// shortest path.
ExprPath ValGraphBFS::getShortestExprPath() {
  NVF_ERROR(allToGroupsVisited(), "Traveral is either not done or failed");

  ExprPath path;

  std::deque<std::pair<GroupType, Direction>> to_visit;
  for (const GroupType& to_group : to_groups_) {
    to_visit.emplace_back(to_group, Direction::Undefined);
  }

  while (!to_visit.empty()) {
    const auto [group, direction] = to_visit.front();
    to_visit.pop_front();

    if (const ExprGroup* eg = std::get_if<ExprGroup>(&group)) {
      path.emplace_back(*eg, direction);
    }

    if (std::find(from_groups_.begin(), from_groups_.end(), group) !=
        from_groups_.end()) {
      continue;
    }

    auto prev_groups_it = prev_groups_.find(group);

    // Some groups may be considered visited without actually
    // visited. No previous path set for such groups.
    // TODO: really?
#if 1
    NVF_ERROR(prev_groups_it != prev_groups_.end());
#else
    if (prev_groups_it == prev_groups_.end()) {
      continue;
    }
#endif

    const Direction dir = prev_groups_it->second.first;
    for (const auto& prev_group : prev_groups_it->second.second) {
      to_visit.emplace_back(prev_group, dir);
    }
  }

  // At this point, we have the reverse path, but it may have multiple exprs
  // that need to be filtered out. Let's say theare are domains 0, 1 and 2, and
  // domains 1 and 2 are merged to produce domain 3, and then domains
  // 0 and 3 are merged to produce domain 4.
  //
  // 0       1         2
  //
  // |       |         |
  // |       |         |
  // |       +-->   <--+
  // |            3
  // |            |
  // |            |
  // +----> 4 <---+
  //
  // Suppose we want to find the shortest path from {4} to {0, 1,
  // 2}. The correct answer should be:
  //
  //   Backward merge of 0, 3 -> 4
  //   Backward merge of 1, 2 -> 3
  //
  // However, the above traversal would produce a path of:
  //
  //   Backward merge of 0, 3 -> 4
  //   Backward merge of 1, 2 -> 3
  //   Backward merge of 1, 2 -> 3
  //   Backward merge of 0, 3 -> 4
  //
  // This is because, since nodes 0, 1 and 2 are the starting nodes,
  // we would first visit 4 from 0, and then 3 from 1 and again 3 from
  // 2. Since node 3 would be visited twice, the path from 3 to 4
  // would be traversed twice as well. Obviously, just reversing this
  // path wouldn't give the correct path. There are two issues here:
  //
  // - The first visit to node 4 from node 0 should not be taken since
  //   node 4 must appear after node 3
  // - Visiting the same node multiple times is redundant and should
  //   be removed
  //
  // Both problems could be solved by taking into considerations if
  // nodes are ready to visit and also are already visited, just like
  // done in the forward traversal. However, there's an additional
  // complexity in this case because the following graph is also valid:
  //
  //         1         2
  //
  // |       |         |
  // |       |         |
  // |       +-->   <--+
  // |            3
  // |            |
  // |            |
  // +----> 4 <---+
  //
  // Notice that node 0 is missing, meaning the shortest path problem
  // in this case is  from node 4 to nodes 1 and 2, and node 0 is not
  // of interest. The correct path is still the same, i.e., first
  // backward merge of 0 and 3 and then another backward merge of 1
  // and 2. It is just node 0 is discarded as it is not of
  // interest. In this case, however, if the
  // traversal was enforced to honor the dependency of each node,
  // it would not be able to visit the backward merge of 0 and 3 as
  // node 0 is missing.
  //
  // A straightforward solution here is simply first generating the
  // path as above and for each node, take the last visit only. Note
  // that the last visit is always guaranteed to satisfy its
  // dependencies.
  //
  // Here, instead of finding the last appearance of each node, the
  // path is first reversed and then only the first appearance is
  // taken since the path needs to be reversed anyway.
  //
  // See the ValGraphBFS2 test for a concrete example.

  std::reverse(path.begin(), path.end());

  VectorOfUniqueEntries<std::pair<ExprGroup, Direction>> unique_path(
      path.begin(), path.end());

  return unique_path.vector();
}

} // namespace nvfuser
