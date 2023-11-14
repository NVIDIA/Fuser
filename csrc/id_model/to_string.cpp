// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/to_string.h>

namespace nvfuser {

// Printing utilities to show critical uniqueness information. i.e. being able
// to tell slight differences between groups we're working with.
namespace {
// Sometimes it can be helpful to directly check the pointer addresses of the
// groups. As one group might look exactly like another group but are in
// different disjoint sets. Leaving commented out by default.
template <typename T>
std::string toString(const T* ptr, bool enable) {
  if (!enable) {
    return "";
  }
  std::stringstream ss;
  ss << ptr;
  return "[0x." + ss.str().substr(9) + "]";
}

std::string indent(int size = 0) {
  std::stringstream ss;
  for (auto i : c10::irange(size)) {
    // Unused variable error
    if (i >= 0) {
      ss << "  ";
    }
  }
  return ss.str();
}
} // namespace

std::string toString(const std::vector<Val*>& id_group, int indent_size) {
  std::vector<unsigned int> names;
  names.reserve(id_group.size());
  for (auto id : id_group) {
    names.push_back(id->name());
  }
  std::sort(names.begin(), names.end());

  std::stringstream ss;
  ss << indent(indent_size) << "{" << names << "}";
  return ss.str();
}

std::string toString(
    const std::vector<IterDomain*>& id_group,
    int indent_size) {
  std::vector<unsigned int> names;
  names.reserve(id_group.size());
  for (auto id : id_group) {
    names.push_back(id->name());
  }
  std::sort(names.begin(), names.end());

  std::stringstream ss;
  ss << indent(indent_size) << "{" << names << "}";
  return ss.str();
}

std::string toString(const ValGroup& id_group, int indent_size, bool with_ptr) {
  std::stringstream ss;
  ss << indent(indent_size) << "idg" << (with_ptr ? "(" : "")
     << toString(id_group.get(), with_ptr) << (with_ptr ? ")" : "")
     << toString(id_group->vector());
  return ss.str();
}

std::string toString(
    const std::vector<ValGroup>& id_groups,
    int indent_size,
    bool with_ptr) {
  std::stringstream ss;

  // Track position in id_groups and its min iter domain name in the set
  std::vector<std::pair<unsigned int, unsigned int>> group_name_info;

  unsigned int pos = 0;

  for (const ValGroup& id_group : id_groups) {
    unsigned int min_id_name = std::numeric_limits<unsigned int>::max();
    for (auto id : *id_group) {
      if (id->name() < min_id_name) {
        min_id_name = id->name();
      }
    }
    group_name_info.emplace_back(min_id_name, pos++);
  }

  ss << indent(indent_size) << "(idgs){\n";

  // Sort based on minimum id in the group
  std::sort(group_name_info.begin(), group_name_info.end());

  for (auto i : c10::irange(group_name_info.size())) {
    auto pos = group_name_info[i].second;
    ss << toString(id_groups[pos], indent_size + 1, with_ptr) << "\n";
  }

  ss << "}";
  return ss.str();
}

std::string toString(
    const ValGroups& id_groups,
    int indent_size,
    bool with_ptr) {
  std::stringstream ss;

  // Track position in id_groups and its min iter domain name in the set
  std::vector<std::pair<unsigned int, unsigned int>> group_name_info;

  unsigned int pos = 0;

  for (const ValGroup& id_group : id_groups) {
    unsigned int min_id_name = std::numeric_limits<unsigned int>::max();
    for (auto id : *id_group) {
      if (id->name() < min_id_name) {
        min_id_name = id->name();
      }
    }
    group_name_info.emplace_back(min_id_name, pos++);
  }

  ss << indent(indent_size) << "(idgs){\n";

  // Sort based on minimum id in the group
  std::sort(group_name_info.begin(), group_name_info.end());

  for (auto i : c10::irange(group_name_info.size())) {
    auto pos = group_name_info[i].second;
    ss << toString(id_groups.vector()[pos], indent_size + 1, with_ptr) << "\n";
  }

  ss << "}";
  return ss.str();
}

std::string toInlineString(const std::vector<ValGroup>& id_groups) {
  // Track position in id_groups and its min iter domain name in the set
  std::vector<std::pair<unsigned int, unsigned int>> group_name_info;

  unsigned int pos = 0;

  for (const ValGroup& id_group : id_groups) {
    unsigned int min_id_name = std::numeric_limits<unsigned int>::max();
    for (auto id : *id_group) {
      if (id->name() < min_id_name) {
        min_id_name = id->name();
      }
    }
    group_name_info.emplace_back(min_id_name, pos++);
  }

  // Sort based on minimum id in the group
  std::sort(group_name_info.begin(), group_name_info.end());

  std::stringstream ss;

  ss << "(idgs){";
  bool first = true;
  for (auto i : c10::irange(group_name_info.size())) {
    if (first) {
      first = false;
    } else {
      ss << ", ";
    }
    auto pos = group_name_info[i].second;
    ss << toString(id_groups[pos]);
  }
  ss << "}";

  return ss.str();
}

std::string toString(const std::vector<Expr*>& expr_group, int indent_size) {
  std::vector<unsigned int> names;
  names.reserve(expr_group.size());
  for (auto expr : expr_group) {
    names.push_back(expr->name());
  }
  std::sort(names.begin(), names.end());

  std::stringstream ss;
  ss << indent(indent_size) << "{" << names << "}";
  return ss.str();
}

std::string toString(
    const ExprGroup& expr_group,
    int indent_size,
    bool with_ptr) {
  std::stringstream ss;
  ss << indent(indent_size) << "exprg" << (with_ptr ? "(" : "")
     << toString(expr_group.get(), with_ptr) << (with_ptr ? ")" : "")
     << toString(expr_group->vector());
  return ss.str();
}

std::string toString(
    const ValGraph& id_graph,
    const std::vector<ExprGroup>& expr_groups,
    int indent_size,
    bool with_ptr) {
  std::stringstream ss;

  // Track position in expr_groups and its min iter domain name in the set
  std::vector<std::pair<unsigned int, unsigned int>> group_name_info;

  unsigned int pos = 0;

  for (const ExprGroup& expr_group : expr_groups) {
    unsigned int min_expr_name = std::numeric_limits<unsigned int>::max();
    for (auto expr : *expr_group) {
      if (expr->name() < min_expr_name) {
        min_expr_name = expr->name();
      }
    }
    group_name_info.emplace_back(min_expr_name, pos++);
  }

  ss << indent(indent_size) << "(exprgs){\n";

  // Sort based on minimum id in the group
  std::sort(group_name_info.begin(), group_name_info.end());

  for (auto i : c10::irange(group_name_info.size())) {
    auto pos = group_name_info[i].second;
    const ExprGroup& expr_group = expr_groups[pos];

    auto inputs = ValGroups(id_graph.inputGroups(expr_group));
    auto outputs = ValGroups(id_graph.outputGroups(expr_group));

    ss << indent(indent_size + 1) << toInlineString(inputs.vector()) << " --"
       << toString(expr_group, 0, with_ptr) << "--> "
       << toInlineString(outputs.vector()) << "\n";
  }

  ss << indent(indent_size) << "}";
  return ss.str();
}

std::string toString(
    const ValGraph& id_graph,
    const ExprGroups& expr_groups,
    int indent_size,
    bool with_ptr) {
  std::stringstream ss;

  // Track position in expr_groups and its min iter domain name in the set
  std::vector<std::pair<unsigned int, unsigned int>> group_name_info;

  unsigned int pos = 0;

  for (const ExprGroup& expr_group : expr_groups) {
    unsigned int min_id_name = std::numeric_limits<unsigned int>::max();
    for (auto id : *expr_group) {
      if (id->name() < min_id_name) {
        min_id_name = id->name();
      }
    }
    group_name_info.emplace_back(min_id_name, pos++);
  }

  ss << indent(indent_size) << "(exprgs){\n";

  // Sort based on minimum id in the group
  std::sort(group_name_info.begin(), group_name_info.end());

  for (auto i : c10::irange(group_name_info.size())) {
    auto pos = group_name_info[i].second;
    auto expr_group = expr_groups.vector()[pos];

    auto inputs = ValGroups(id_graph.inputGroups(expr_group));
    auto outputs = ValGroups(id_graph.outputGroups(expr_group));

    ss << indent(indent_size + 1) << toInlineString(inputs.vector()) << " --"
       << toString(expr_group, 0, with_ptr) << "--> "
       << toInlineString(outputs.vector()) << "\n";
  }

  ss << indent(indent_size) << "}";
  return ss.str();
}

std::string idGroupsString(
    const ValGraph& id_graph,
    int indent_size,
    bool with_ptr) {
  ValGroups id_groups(
      id_graph.disjointValSets().disjointSets().begin(),
      id_graph.disjointValSets().disjointSets().end());
  return toString(id_groups, indent_size, with_ptr);
}
std::string exprGroupsString(
    const ValGraph& id_graph,
    int indent_size,
    bool with_ptr) {
  ExprGroups expr_groups(
      id_graph.disjointExprSets().disjointSets().begin(),
      id_graph.disjointExprSets().disjointSets().end());
  return toString(id_graph, expr_groups, indent_size, with_ptr);
}

std::string definitionsString(
    const ValGraph& id_graph,
    int indent_size,
    bool with_ptr) {
  ExprGroups defs;
  for (const ValGroup& id_group : id_graph.disjointValSets().disjointSets()) {
    auto definition_pair = id_graph.getDefinitions(id_group);
    if (definition_pair.second) {
      for (const ExprGroup& expr_group : definition_pair.first) {
        defs.pushBack(expr_group);
      }
    }
  }
  return toString(id_graph, defs, indent_size, with_ptr);
}

std::string usesString(
    const ValGraph& id_graph,
    int indent_size,
    bool with_ptr) {
  ExprGroups uses;
  for (const ValGroup& id_group : id_graph.disjointValSets().disjointSets()) {
    auto definition_pair = id_graph.getUses(id_group);
    if (definition_pair.second) {
      for (const ExprGroup& expr_group : definition_pair.first) {
        uses.pushBack(expr_group);
      }
    }
  }
  return toString(id_graph, uses, indent_size, with_ptr);
}

} // namespace nvfuser
