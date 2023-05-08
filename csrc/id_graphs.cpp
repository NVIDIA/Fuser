#include <id_graphs.h>

#include <disjoint_set.h>
#include <ir_utils.h>
#include <lower2device.h>
#include <lower_trivial_broadcast.h>
#include <lower_utils.h>
#include <root_domain_map.h>
#include <transform_iter.h>

#include <tuple>
#include <typeinfo>

namespace nvfuser {

// Printing utilities to show critical uniqueness information. i.e. being able
// to tell slight differences between groups we're working with.
namespace debug {

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

std::string toString(
    const std::vector<IterDomain*>& id_group,
    int indent_size) {
  std::vector<unsigned int> names;
  for (auto id : id_group) {
    names.push_back(id->name());
  }
  std::sort(names.begin(), names.end());

  std::stringstream ss;
  ss << indent(indent_size) << "{" << names << "}";
  return ss.str();
}

std::string toString(const IdGroup& id_group, int indent_size, bool with_ptr) {
  std::stringstream ss;
  ss << indent(indent_size) << "idg" << (with_ptr ? "(" : "")
     << toString(id_group.get(), with_ptr) << (with_ptr ? ")" : "")
     << toString(id_group->vector());
  return ss.str();
}

std::string toString(
    const std::vector<IdGroup>& id_groups,
    int indent_size,
    bool with_ptr) {
  std::stringstream ss;

  // Track position in id_groups and its min iter domain name in the set
  std::vector<std::pair<unsigned int, unsigned int>> group_name_info;

  unsigned int pos = 0;

  for (auto id_group : id_groups) {
    unsigned int min_id_name = std::numeric_limits<unsigned int>::max();
    for (auto id : *id_group) {
      if (id->name() < min_id_name) {
        min_id_name = id->name();
      }
    }
    group_name_info.push_back(std::make_pair(min_id_name, pos++));
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
    const IdGroups& id_groups,
    int indent_size,
    bool with_ptr) {
  std::stringstream ss;

  // Track position in id_groups and its min iter domain name in the set
  std::vector<std::pair<unsigned int, unsigned int>> group_name_info;

  unsigned int pos = 0;

  for (auto id_group : id_groups) {
    unsigned int min_id_name = std::numeric_limits<unsigned int>::max();
    for (auto id : *id_group) {
      if (id->name() < min_id_name) {
        min_id_name = id->name();
      }
    }
    group_name_info.push_back(std::make_pair(min_id_name, pos++));
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

std::string toInlineString(const std::vector<IdGroup>& id_groups) {
  // Track position in id_groups and its min iter domain name in the set
  std::vector<std::pair<unsigned int, unsigned int>> group_name_info;

  unsigned int pos = 0;

  for (auto id_group : id_groups) {
    unsigned int min_id_name = std::numeric_limits<unsigned int>::max();
    for (auto id : *id_group) {
      if (id->name() < min_id_name) {
        min_id_name = id->name();
      }
    }
    group_name_info.push_back(std::make_pair(min_id_name, pos++));
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

  return ss.str();
}

std::string toString(const std::vector<Expr*>& expr_group, int indent_size) {
  std::vector<unsigned int> names;
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
    const IdGraph& id_graph,
    const std::vector<ExprGroup>& expr_groups,
    int indent_size,
    bool with_ptr) {
  std::stringstream ss;

  // Track position in expr_groups and its min iter domain name in the set
  std::vector<std::pair<unsigned int, unsigned int>> group_name_info;

  unsigned int pos = 0;

  for (auto expr_group : expr_groups) {
    unsigned int min_expr_name = std::numeric_limits<unsigned int>::max();
    for (auto expr : *expr_group) {
      if (expr->name() < min_expr_name) {
        min_expr_name = expr->name();
      }
    }
    group_name_info.push_back(std::make_pair(min_expr_name, pos++));
  }

  ss << indent(indent_size) << "(exprgs){\n";

  // Sort based on minimum id in the group
  std::sort(group_name_info.begin(), group_name_info.end());

  for (auto i : c10::irange(group_name_info.size())) {
    auto pos = group_name_info[i].second;
    auto expr_group = expr_groups[pos];

    auto inputs = IdGroups(id_graph.inputGroups(expr_group));
    auto outputs = IdGroups(id_graph.outputGroups(expr_group));

    ss << indent(indent_size + 1) << toInlineString(inputs.vector()) << " --"
       << toString(expr_group, 0, with_ptr) << "--> "
       << toInlineString(outputs.vector()) << "\n";
  }

  ss << indent(indent_size) << "}";
  return ss.str();
}

std::string toString(
    const IdGraph& id_graph,
    const ExprGroups& expr_groups,
    int indent_size,
    bool with_ptr) {
  std::stringstream ss;

  // Track position in expr_groups and its min iter domain name in the set
  std::vector<std::pair<unsigned int, unsigned int>> group_name_info;

  unsigned int pos = 0;

  for (auto expr_group : expr_groups) {
    unsigned int min_id_name = std::numeric_limits<unsigned int>::max();
    for (auto id : *expr_group) {
      if (id->name() < min_id_name) {
        min_id_name = id->name();
      }
    }
    group_name_info.push_back(std::make_pair(min_id_name, pos++));
  }

  ss << indent(indent_size) << "(exprgs){\n";

  // Sort based on minimum id in the group
  std::sort(group_name_info.begin(), group_name_info.end());

  for (auto i : c10::irange(group_name_info.size())) {
    auto pos = group_name_info[i].second;
    auto expr_group = expr_groups.vector()[pos];

    auto inputs = IdGroups(id_graph.inputGroups(expr_group));
    auto outputs = IdGroups(id_graph.outputGroups(expr_group));

    ss << indent(indent_size + 1) << toInlineString(inputs.vector()) << " --"
       << toString(expr_group, 0, with_ptr) << "--> "
       << toInlineString(outputs.vector()) << "\n";
  }

  ss << indent(indent_size) << "}";
  return ss.str();
}

std::string idGroupsString(
    const IdGraph& id_graph,
    int indent_size,
    bool with_ptr) {
  IdGroups id_groups(
      id_graph.disjointIdSets().disjointSets().begin(),
      id_graph.disjointIdSets().disjointSets().end());
  return toString(id_groups, indent_size, with_ptr);
}
std::string exprGroupsString(
    const IdGraph& id_graph,
    int indent_size,
    bool with_ptr) {
  ExprGroups expr_groups(
      id_graph.disjointExprSets().disjointSets().begin(),
      id_graph.disjointExprSets().disjointSets().end());
  return toString(id_graph, expr_groups, indent_size, with_ptr);
}

std::string definitionsString(
    const IdGraph& id_graph,
    int indent_size,
    bool with_ptr) {
  ExprGroups defs;
  for (auto id_group : id_graph.disjointIdSets().disjointSets()) {
    auto definition_pair = id_graph.iterDomainGroupDefinitions(id_group);
    if (definition_pair.second) {
      for (auto expr_group : definition_pair.first) {
        defs.pushBack(expr_group);
      }
    }
  }
  return toString(id_graph, defs, indent_size, with_ptr);
}

std::string usesString(
    const IdGraph& id_graph,
    int indent_size,
    bool with_ptr) {
  ExprGroups uses;
  for (auto id_group : id_graph.disjointIdSets().disjointSets()) {
    auto definition_pair = id_graph.iterDomainGroupUses(id_group);
    if (definition_pair.second) {
      for (auto expr_group : definition_pair.first) {
        uses.pushBack(expr_group);
      }
    }
  }
  return toString(id_graph, uses, indent_size, with_ptr);
}

} // namespace debug

namespace {

bool transformAtributesMatch(Expr* first, Expr* second) {
  if (first == nullptr || second == nullptr) {
    return false;
  }

  TORCH_INTERNAL_ASSERT(
      first->isA<Merge>() || first->isA<Split>() || first->isA<Swizzle2D>() ||
          first->isA<Resize>(),
      "Merge and split are the only expressions supported through rfactor operations in compute at map, but found:\n",
      first->toString());

  if (typeid(*first) != typeid(*second)) {
    return false;
  }

  if (first->isA<Split>()) {
    auto first_split = first->as<Split>();
    auto second_split = second->as<Split>();
    if (!first_split->factor()->sameAs(second_split->factor()) ||
        first_split->innerSplit() != second_split->innerSplit() ||
        !first_split->startOffset()->sameAs(second_split->startOffset()) ||
        !first_split->stopOffset()->sameAs(second_split->stopOffset())) {
      return false;
    }
  }

  if (first->isA<Swizzle2D>()) {
    auto first_swizzle = first->as<Swizzle2D>();
    auto second_swizzle = second->as<Swizzle2D>();
    if (first_swizzle->swizzleMode() != second_swizzle->swizzleMode() ||
        first_swizzle->swizzleType() != second_swizzle->swizzleType()) {
      return false;
    }
  }

  return true;
}
} // namespace

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
        auto disjoint_pair = graph().disjointIdSet(id);
        if (disjoint_pair.second) {
          all_ids.pushBack(disjoint_pair.first);
        }
      }
    }

    if (sub_selection_.empty()) {
      all_exprs = ExprGroups(
          graph().disjointExprSets().disjointSets().begin(),
          graph().disjointExprSets().disjointSets().end());
    } else {
      for (auto id_group : all_ids) {
        for (auto def : graph().uniqueDefinitions(id_group)) {
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
    auto unique_defs = graph().uniqueDefinitions(id_group);
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
          auto uses_pair = graph().iterDomainGroupUses(current_id_group);
          if (uses_pair.second) {
            to_visit_exprs.pushBack(uses_pair.first);
          }
        }
      } else {
        still_to_visit_ids.pushBack(current_id_group);
      }
    }
    std::swap(to_visit_ids, still_to_visit_ids);

    TORCH_INTERNAL_ASSERT(
        something_was_processed ||
            (to_visit_ids.size() == 0 && to_visit_exprs.size() == 0),
        "Infinite loop entered.");
  }
}

IdGraph::IdGraph(const IdGraph& other) {
  disjoint_ids_ = other.disjoint_ids_;
  disjoint_exprs_ = other.disjoint_exprs_;
  view_rfactor_ids_ = other.view_rfactor_ids_;

  for (auto orig_unique_def_pair : other.unique_definitions_) {
    auto orig_id_group = orig_unique_def_pair.first;
    auto orig_expr_groups = orig_unique_def_pair.second;

    auto new_id_group_pair = disjointIdSet(orig_id_group->front());
    TORCH_INTERNAL_ASSERT(new_id_group_pair.second);
    auto new_id_group = new_id_group_pair.first;

    ExprGroups new_expr_groups;
    for (auto orig_expr_group : orig_expr_groups) {
      auto new_expr_group_pair = disjointExprSet(orig_expr_group->front());
      TORCH_INTERNAL_ASSERT(new_expr_group_pair.second);
      new_expr_groups.pushBack(new_expr_group_pair.first);
    }

    unique_definitions_[new_id_group] = new_expr_groups;
  }

  for (auto orig_unique_use_pair : other.unique_uses_) {
    auto orig_id_group = orig_unique_use_pair.first;
    auto orig_expr_groups = orig_unique_use_pair.second;

    auto new_id_group_pair = disjointIdSet(orig_id_group->front());
    TORCH_INTERNAL_ASSERT(new_id_group_pair.second);
    auto new_id_group = new_id_group_pair.first;

    ExprGroups new_expr_groups;
    for (auto orig_expr_group : orig_expr_groups) {
      auto new_expr_group_pair = disjointExprSet(orig_expr_group->front());
      TORCH_INTERNAL_ASSERT(new_expr_group_pair.second);
      new_expr_groups.pushBack(new_expr_group_pair.first);
    }

    unique_uses_[new_id_group] = new_expr_groups;
  }
}

IdGraph& IdGraph::operator=(const IdGraph& other) {
  disjoint_ids_.clear();
  disjoint_exprs_.clear();
  unique_definitions_.clear();
  unique_uses_.clear();
  view_rfactor_ids_.clear();
  IdGraph copy(other);
  std::swap(*this, copy);
  return *this;
}

const DisjointSets<IterDomain*>& IdGraph::disjointIdSets() const {
  return disjoint_ids_;
}

DisjointSets<IterDomain*>& IdGraph::disjointIdSets() {
  return disjoint_ids_;
}

std::pair<IdGroup, bool> IdGraph::disjointIdSet(IterDomain* id) const {
  auto disjoint_set_it = disjoint_ids_.disjointSetMap().find(id);
  if (disjoint_set_it == disjoint_ids_.disjointSetMap().end()) {
    return std::make_pair(IdGroup(nullptr), false);
  }
  return std::make_pair(disjoint_set_it->second, true);
}

const DisjointSets<Expr*>& IdGraph::disjointExprSets() const {
  return disjoint_exprs_;
}

DisjointSets<Expr*>& IdGraph::disjointExprSets() {
  return disjoint_exprs_;
}

std::pair<ExprGroup, bool> IdGraph::disjointExprSet(Expr* expr) const {
  auto disjoint_set_it = disjoint_exprs_.disjointSetMap().find(expr);
  if (disjoint_set_it == disjoint_exprs_.disjointSetMap().end()) {
    return std::make_pair(ExprGroup(nullptr), false);
  }
  return std::make_pair(disjoint_set_it->second, true);
}

ExprGroup IdGraph::toGroup(Expr* expr) const {
  auto disjoint_set_pair = disjointExprSet(expr);
  TORCH_INTERNAL_ASSERT(
      disjoint_set_pair.second,
      "\nExpr group could not be found in graph associated with: ",
      expr->toString());
  return disjoint_set_pair.first;
}

IdGroup IdGraph::toGroup(IterDomain* id) const {
  auto disjoint_set_pair = disjointIdSet(id);
  TORCH_INTERNAL_ASSERT(
      disjoint_set_pair.second,
      "\nId group could not be found in graph associated with: ",
      id->toString(),
      "\n");
  return disjoint_set_pair.first;
}

ExprGroups IdGraph::toGroups(const VectorOfUniqueEntries<Expr*>& exprs) const {
  ExprGroups expr_groups;
  for (auto expr : exprs) {
    expr_groups.pushBack(toGroup(expr));
  }
  return expr_groups;
}

IdGroups IdGraph::toGroups(
    const VectorOfUniqueEntries<IterDomain*>& ids) const {
  IdGroups id_groups;
  for (auto id : ids) {
    id_groups.pushBack(toGroup(id));
  }
  return id_groups;
}

std::vector<IdGroup> IdGraph::outputGroups(ExprGroup expr) const {
  std::vector<IdGroup> output_groups;
  for (auto id_output :
       ir_utils::filterByType<IterDomain>(expr->front()->outputs())) {
    output_groups.push_back(toGroup(id_output));
  }
  return output_groups;
}

std::vector<IdGroup> IdGraph::inputGroups(ExprGroup expr) const {
  std::vector<IdGroup> input_groups;
  for (auto id_input :
       ir_utils::filterByType<IterDomain>(expr->front()->inputs())) {
    input_groups.push_back(toGroup(id_input));
  }
  return input_groups;
}

ExprGroups IdGraph::allUsesOf(const IdGroups& of) const {
  ExprGroups to_visit;
  for (auto of_id_group : of) {
    auto group_uses_pair = iterDomainGroupUses(of_id_group);
    if (group_uses_pair.second) {
      to_visit.pushBack(group_uses_pair.first);
    }
  }

  ExprGroups visited;
  while (to_visit.size() > 0) {
    auto current_expr = to_visit.popFront();
    visited.pushBack(current_expr);
    auto output_ids = outputGroups(current_expr);
    for (auto output_id : output_ids) {
      auto group_uses_pair = iterDomainGroupUses(output_id);
      if (!group_uses_pair.second) {
        continue;
      }
      for (auto group_use : group_uses_pair.first) {
        if (visited.has(group_use)) {
          continue;
        }
        to_visit.pushBack(group_use);
      }
    }
  }

  return visited;
}

ExprGroups IdGraph::allDefinitionsOf(const IdGroups& of) const {
  ExprGroups to_visit;
  for (auto of_id_group : of) {
    auto group_defs_pair = iterDomainGroupDefinitions(of_id_group);
    if (group_defs_pair.second) {
      to_visit.pushBack(group_defs_pair.first);
    }
  }

  ExprGroups visited;
  while (to_visit.size() > 0) {
    auto current_expr = to_visit.popFront();
    visited.pushBack(current_expr);
    auto input_ids = inputGroups(current_expr);
    for (auto input_id : input_ids) {
      auto group_defs_pair = iterDomainGroupDefinitions(input_id);
      if (!group_defs_pair.second) {
        continue;
      }
      for (auto group_def : group_defs_pair.first) {
        if (visited.has(group_def)) {
          continue;
        }
        to_visit.pushBack(group_def);
      }
    }
  }

  return visited;
}

ExprGroups IdGraph::getExprsBetween(const IdGroups& from, const IdGroups& to)
    const {
  auto all_uses_of_from = allUsesOf(from);
  auto all_definitions_of_to = allDefinitionsOf(to);

  // All of the expressions between from and to. Not all will be used as we
  // just want to define each iter domain group once.
  auto all_exprs = all_uses_of_from.intersect(all_definitions_of_to);

  // There could be IterDomains in from or to that are between other from and
  // to nodes. Make sure to clear those out.
  IdGroups terminating_inputs;
  IdGroups terminating_outputs;
  {
    IdGroups not_inputs;
    IdGroups not_outputs;
    IdGroups all_id_groups;

    for (auto expr_group : all_exprs) {
      auto inp_groups = inputGroups(expr_group);
      auto out_groups = outputGroups(expr_group);
      if (IdGroups(inp_groups).intersect(IdGroups(out_groups)).size() > 0) {
        // Expression is just a loop to its current group, ignore
        continue;
      }

      all_id_groups.pushBack(inp_groups);

      if (!inp_groups.empty()) {
        not_outputs.pushBack(inp_groups);
      }

      all_id_groups.pushBack(out_groups);

      if (!out_groups.empty()) {
        not_inputs.pushBack(out_groups);
      }
    }
    terminating_inputs = all_id_groups.subtract(not_inputs);
    terminating_outputs = all_id_groups.subtract(not_outputs);
  }

  // Track all expressions to get from outputs to this IterDomain. We
  // traverse backwards as that's the direction of indexing expressions. An
  // index is assigned to each leaf of a domain and as we traverse backwards
  // we're effectively accumulating indexing math. We'll only keep the fewest
  // expression lists to get to the iter domain.
  std::unordered_map<IdGroup, ExprGroups> required_ind_exprs_ids;
  std::unordered_map<ExprGroup, ExprGroups> required_ind_exprs_exprs;

  // Return if all output IterDomain groups of an expression group have
  // already been visited
  auto outputsVisited = [&](ExprGroup expr) {
    for (auto id_group : outputGroups(expr)) {
      if (required_ind_exprs_ids.find(id_group) ==
          required_ind_exprs_ids.end()) {
        return false;
      }
    }
    return true;
  };

  auto allIdUsesVisisted = [&](IdGroup id) {
    auto uses_pair = iterDomainGroupUses(id);
    if (!uses_pair.second) {
      return true;
    }
    for (auto use_group : uses_pair.first) {
      if (all_exprs.has(use_group)) {
        if (required_ind_exprs_exprs.find(use_group) ==
            required_ind_exprs_exprs.end()) {
          return false;
        }
      }
    }
    return true;
  };

  // Returns all expression groups in required_ind_exprs_ids of outputs
  auto requiredExprsOutputs = [&](ExprGroup expr) {
    ExprGroups all_output_required_exprs;
    for (auto id_group : outputGroups(expr)) {
      auto id_group_exprs_it = required_ind_exprs_ids.find(id_group);
      TORCH_INTERNAL_ASSERT(
          id_group_exprs_it != required_ind_exprs_ids.end(),
          "Failure in Iter Domain Graph index resolution, count expected for group: ",
          id_group->toString());
      all_output_required_exprs.pushBack(id_group_exprs_it->second);
    }
    return all_output_required_exprs;
  };

  auto processExpr = [&](ExprGroup expr) {
    if (!outputsVisited(expr)) {
      return false;
    }
    // Accumulate expressions from all outputs add this expression and set it
    // as current expressions required indexing expressions.
    required_ind_exprs_exprs[expr] = requiredExprsOutputs(expr);
    return true;
  };

  auto processId = [&](IdGroup id) {
    // Track if we've grabed any of the uses required indexing expressions.
    bool initialized = false;
    // Expression group of all indexing expressions required for this iter
    // domain coming back from any of its uses.
    ExprGroups min_groups;

    auto uses_pair = iterDomainGroupUses(id);
    if (!uses_pair.second) {
      // No expressions required for this iter domain, it must be a
      // terminating output.
      required_ind_exprs_ids[id] = min_groups;
      return true;
    }

    // Only worry about expressions between inputs and outputs we're
    // looking at.
    for (auto use_group : uses_pair.first.intersect(all_exprs)) {
      auto use_required_ind_exprs_it = required_ind_exprs_exprs.find(use_group);
      if (use_required_ind_exprs_it == required_ind_exprs_exprs.end()) {
        // If there isn't an entry for the use expression it wasn't
        // processed, so don't try to process this iter domain yet.
        return false;
      }
      if (!initialized) {
        // If first use found initialize the minimum expression group
        min_groups =
            use_required_ind_exprs_it->second.computeUnion({use_group});
        initialized = true;
      } else if (
          use_required_ind_exprs_it->second.size() + 1 < min_groups.size()) {
        // If current use has fewer expressions use that, make sure to add the
        // use expression.
        min_groups =
            use_required_ind_exprs_it->second.computeUnion({use_group});
      }
    }
    required_ind_exprs_ids[id] = min_groups;
    return true;
  };

  IdGroups to_visit_ids = terminating_outputs;
  ExprGroups to_visit_exprs;

  while (to_visit_ids.size() > 0 || to_visit_exprs.size() > 0) {
    // Process expressions first as all uses of iter domains have to be
    // processed before we can process that iter domain.

    // Try to detect when nothing has been processed which would put us in an
    // infinite loop
    bool something_was_processed = false;
    ExprGroups still_to_visit_exprs;
    while (to_visit_exprs.size() > 0) {
      auto currently_visiting = to_visit_exprs.popFront();
      if (required_ind_exprs_exprs.find(currently_visiting) !=
          required_ind_exprs_exprs.end()) {
        continue;
      }
      if (processExpr(currently_visiting)) {
        something_was_processed = true;
        auto inp_groups = inputGroups(currently_visiting);
        for (auto inp_group : inp_groups) {
          to_visit_ids.pushBack(inp_group);
        }
      } else {
        still_to_visit_exprs.pushBack(currently_visiting);
      }
    }

    std::swap(to_visit_exprs, still_to_visit_exprs);

    IdGroups still_to_visit_ids;
    while (to_visit_ids.size() > 0) {
      auto currently_visiting = to_visit_ids.popFront();
      if (required_ind_exprs_ids.find(currently_visiting) !=
          required_ind_exprs_ids.end()) {
        continue;
      }

      if (processId(currently_visiting)) {
        something_was_processed = true;
        auto definitions_pair = iterDomainGroupDefinitions(currently_visiting);
        if (definitions_pair.second) {
          for (auto def : definitions_pair.first) {
            if (!all_exprs.has(def)) {
              continue;
            }
            if (required_ind_exprs_exprs.find(def) ==
                required_ind_exprs_exprs.end()) {
              to_visit_exprs.pushBack(def);
            }
          }
        }
      } else {
        still_to_visit_ids.pushBack(currently_visiting);
      }
    }

    TORCH_INTERNAL_ASSERT(
        something_was_processed ||
            (to_visit_ids.size() == 0 && to_visit_exprs.size() == 0),
        "Infinite loop entered.");
  }

  // We want to traverse the expressions registered in required_ind_exprs_ids,
  // let's create a strict "uses path"
  std::unordered_map<IdGroup, ExprGroups> uses_path;
  for (auto entry : required_ind_exprs_ids) {
    auto id = entry.first;
    auto traverse_exprs = entry.second;
    auto all_uses = iterDomainGroupUses(id);
    if (all_uses.second) {
      uses_path[id] = traverse_exprs.intersect(all_uses.first);
    } else {
      uses_path[id] = {};
      continue;
    }
  }

  // Topologically sort the uses_path.
  ExprGroups sorted_exprs;
  ExprGroups to_visit;

  for (auto inp : terminating_inputs) {
    auto use_it = uses_path.find(inp);
    if (use_it == uses_path.end()) {
      // This can happen for a trivial traversal where inputs and outputs are
      // exactly the same.
      continue;
    }
    auto uses = use_it->second;
    for (auto use : uses) {
      to_visit.pushBack(use);
    }
  }

  IdGroups visited = terminating_inputs;

  while (to_visit.size() > 0) {
    bool something_processed = false;
    ExprGroups still_to_visit;
    while (to_visit.size() > 0) {
      auto currently_visiting = to_visit.popFront();
      auto inputs = inputGroups(currently_visiting);
      if (std::all_of(inputs.begin(), inputs.end(), [&](IdGroup inp_id) {
            return visited.has(inp_id);
          })) {
        something_processed = true;
        sorted_exprs.pushBack(currently_visiting);
        auto outputs = outputGroups(currently_visiting);
        for (auto out_id : outputs) {
          visited.pushBack(out_id);
          auto use_pair = iterDomainGroupUses(out_id);
          if (!use_pair.second) {
            continue;
          }
          still_to_visit.pushBack(use_pair.first.intersect(all_exprs));
        }
      } else {
        still_to_visit.pushBack(currently_visiting);
      }
    }
    std::swap(to_visit, still_to_visit);
    TORCH_INTERNAL_ASSERT(something_processed, "Infinite loop entered.");
  }

  return sorted_exprs;
}

std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>> IdGraph::
    buildMapBetween(
        const std::vector<IterDomain*>& from,
        const std::vector<IterDomain*>& to) const {
  std::unordered_map<IterDomain*, IdGroup> from_ids2set;

  for (auto from_id : from) {
    auto from_disjoint_set_pair = disjointIdSet(from_id);
    if (!from_disjoint_set_pair.second) {
      continue;
    }
    from_ids2set[from_id] = from_disjoint_set_pair.first;
  }

  // Map from the sets associated with the IterDomains in to, to those iter
  // domains
  std::unordered_map<IdGroup, VectorOfUniqueEntries<IterDomain*>> set2to_ids;

  for (auto to_id : to) {
    auto to_disjoint_set_pair = disjointIdSet(to_id);
    if (!to_disjoint_set_pair.second) {
      continue;
    }
    auto to_set = to_disjoint_set_pair.first;
    auto set2to_ids_it = set2to_ids.find(to_set);

    if (set2to_ids_it == set2to_ids.end()) {
      set2to_ids[to_set] = {to_id};
    } else {
      set2to_ids[to_set].pushBack(to_id);
    }
  }

  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      from_ids2to_ids;
  for (auto from_id : from) {
    from_ids2to_ids[from_id] = VectorOfUniqueEntries<IterDomain*>();

    auto from_it = from_ids2set.find(from_id);
    TORCH_INTERNAL_ASSERT(from_it != from_ids2set.end());

    auto from_set = from_it->second;
    auto to_entry_it = set2to_ids.find(from_set);
    if (to_entry_it == set2to_ids.end()) {
      continue;
    }
    from_ids2to_ids[from_id] = to_entry_it->second;
  }
  return from_ids2to_ids;
}

std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>> IdGraph::
    buildMapBetween(
        const VectorOfUniqueEntries<IterDomain*>& from,
        const VectorOfUniqueEntries<IterDomain*>& to) const {
  return buildMapBetween(from.vector(), to.vector());
}

std::pair<ExprGroups, bool> IdGraph::iterDomainGroupDefinitions(
    IdGroup id_group) const {
  auto null_return = std::make_pair(ExprGroups(), false);

  if (id_group == nullptr) {
    return null_return;
  }

  auto definitions_it = unique_definitions_.find(id_group);
  if (definitions_it == unique_definitions_.end()) {
    return null_return;
  }

  return std::make_pair(definitions_it->second, true);
}

std::pair<ExprGroups, bool> IdGraph::iterDomainGroupUses(
    IdGroup id_group) const {
  auto null_return = std::make_pair(ExprGroups(), false);

  if (id_group == nullptr) {
    return null_return;
  }

  auto uses_it = unique_uses_.find(id_group);
  if (uses_it == unique_uses_.end()) {
    return null_return;
  }

  return std::make_pair(uses_it->second, true);
}

std::string IdGraph::toString() const {
  std::stringstream ss;
  ss << "IdGraph { \n";
  ss << "Disjoint Ids:\n"
     << debug::idGroupsString(*this, 1) << "\n\nDisjoint Expression groups:\n"
     << debug::exprGroupsString(*this, 1) << std::endl;
  ss << " } IdGraph\n" << std::endl;
  return ss.str();
}

std::vector<std::vector<IterDomain*>> IdGraph::isTrivialExpr(Expr* expr) {
  std::vector<std::vector<IterDomain*>> mapped_ids;
  if (auto merge = dynamic_cast<Merge*>(expr)) {
    if (merge->inner()->extent()->isOneInt()) {
      mapped_ids.push_back({merge->outer(), merge->out()});
    }
    if (merge->outer()->extent()->isOneInt()) {
      mapped_ids.push_back({merge->inner(), merge->out()});
    }
  } else if (auto split = dynamic_cast<Split*>(expr)) {
    if (split->factor()->isOneInt() && split->startOffset()->isZeroInt() &&
        split->stopOffset()->isZeroInt()) {
      if (split->innerSplit()) {
        mapped_ids.push_back({split->in(), split->outer()});
      } else {
        mapped_ids.push_back({split->in(), split->inner()});
      }
    }
  } else if (auto swizzle = dynamic_cast<Swizzle2D*>(expr)) {
    if (swizzle->swizzleType() == Swizzle2DType::NoSwizzle ||
        swizzle->swizzleMode() == SwizzleMode::NoSwizzle) {
      mapped_ids.push_back({swizzle->inX(), swizzle->outX()});
      mapped_ids.push_back({swizzle->inY(), swizzle->outY()});
    }
  }
  return mapped_ids;
}

void IdGraph::initializeId(
    IterDomain* id,
    const VectorOfUniqueEntries<Expr*>& definitions,
    const VectorOfUniqueEntries<Expr*>& uses) {
  auto id_disjoint_set = disjointIdSets().initializeSet(id).first->second;

  ExprGroups def_groups;
  for (auto def : definitions) {
    auto expr_set = disjointExprSets().initializeSet(def).first->second;
    def_groups.pushBack(expr_set);
  }
  unique_definitions_[id_disjoint_set] = def_groups;

  ExprGroups use_groups;
  for (auto use : uses) {
    auto expr_set = disjointExprSets().initializeSet(use).first->second;
    use_groups.pushBack(expr_set);
  }
  unique_uses_[id_disjoint_set] = use_groups;
}

bool IdGraph::exprsMap(Expr* first, Expr* second, bool forward) const {
  if (!transformAtributesMatch(first, second)) {
    return false;
  }

  auto first_ids = ir_utils::filterByType<IterDomain>(
                       forward ? first->inputs() : first->outputs())
                       .vector();

  auto second_ids = ir_utils::filterByType<IterDomain>(
                        forward ? second->inputs() : second->outputs())
                        .vector();

  TORCH_INTERNAL_ASSERT(
      first_ids.size() == second_ids.size(),
      "Expected number of ",
      (forward ? "inputs" : "outputs"),
      " to match for\n",
      first->toString(),
      second->toString());

  {
    std::vector<std::pair<IterDomain*, IterDomain*>> zipped_ids;

    std::transform(
        first_ids.begin(),
        first_ids.end(),
        second_ids.begin(),
        std::back_inserter(zipped_ids),
        [](IterDomain* first, IterDomain* second) {
          return std::make_pair(first, second);
        });

    if (std::any_of(
            zipped_ids.begin(),
            zipped_ids.end(),
            [&](std::pair<IterDomain*, IterDomain*> id_pair) {
              return !disjointIdSets().permissiveAreMapped(
                  id_pair.first, id_pair.second);
            })) {
      return false;
    }
  }

  // Special handling for backprop of merge
  if (first->isA<Merge>() && !forward) {
    // Can't back prop through merge without making sure one input actually
    // matches. This can be done on a map or extent basis.
    auto merge0 = first->as<Merge>();
    auto merge1 = second->as<Merge>();

    auto extent_0o = merge0->outer()->extent();
    auto extent_0i = merge0->inner()->extent();
    auto extent_1o = merge1->outer()->extent();
    auto extent_1i = merge1->inner()->extent();

    auto extent_0_match = extent_0o->sameAs(extent_1o) ||
        (extent_0o->isConstInt() && extent_1o->isConstInt() &&
         extent_0o->evaluateInt() == extent_1o->evaluateInt()) ||
        disjointIdSets().permissiveAreMapped(merge0->outer(), merge1->outer());

    auto extent_1_match = extent_0i->sameAs(extent_1i) ||
        (extent_0i->isConstInt() && extent_1i->isConstInt() &&
         extent_0i->evaluateInt() == extent_1i->evaluateInt()) ||
        disjointIdSets().permissiveAreMapped(merge0->inner(), merge1->inner());

    if (!(extent_0_match || extent_1_match)) {
      return false;
    }
  }

  // TODO: For now we're using same as, however we could know what val's are
  // exactly the same given the exact map. We might want to pipe that
  // information through to here.
  if (first->isA<Resize>()) {
    if (!first->as<Resize>()->leftExpand()->sameAs(
            second->as<Resize>()->leftExpand())) {
      return false;
    }

    if (!first->as<Resize>()->rightExpand()->sameAs(
            second->as<Resize>()->rightExpand())) {
      return false;
    }
  }

  return true;
}

ExprGroups IdGraph::uniqueDefinitions(IdGroup group) const {
  auto unique_defs_it = unique_definitions_.find(group);
  TORCH_INTERNAL_ASSERT(
      unique_defs_it != unique_definitions_.end(),
      "Definition not found for IdGroup: ",
      group->toString());
  return unique_defs_it->second;
}

ExprGroups IdGraph::uniqueUses(IdGroup group) const {
  auto unique_uses_it = unique_uses_.find(group);
  TORCH_INTERNAL_ASSERT(
      unique_uses_it != unique_uses_.end(),
      "Uses not found for IdGroup: ",
      group->toString());
  return unique_uses_it->second;
}

void IdGraph::maybeMapThroughExprs(Expr* expr0, Expr* expr1, bool forward) {
  if (exprsMap(expr0, expr1, forward)) {
    if (propagate_exprs_) {
      mapExprs(expr0, expr1);
      mapThroughExpr(expr0, expr1, forward);
    } else if (
        inputGroups(toGroup(expr0)) == inputGroups(toGroup(expr1)) &&
        outputGroups(toGroup(expr0)) == outputGroups(toGroup(expr1))) {
      mapExprs(expr0, expr1);
    }
  }
}

void IdGraph::mapExprs(Expr* expr0, Expr* expr1) {
  if (expr0 == expr1) {
    return;
  }

  if (disjointExprSets().strictAreMapped(expr0, expr1)) {
    return;
  }

  ExprGroup expr0_orig_group = toGroup(expr0);
  ExprGroup expr1_orig_group = toGroup(expr1);

  disjointExprSets().mapEntries(expr0, expr1);

  auto expr_new_group = toGroup(expr0);

  // Update unique uses of producers
  IdGroups producers;
  for (auto expr : std::vector<Expr*>{expr0, expr1}) {
    for (auto input_id : ir_utils::filterByType<IterDomain>(expr->inputs())) {
      producers.pushBack(toGroup(input_id));
    }
  }

  for (auto producer_group : producers) {
    uniqueUses().at(producer_group).erase(expr0_orig_group);
    uniqueUses().at(producer_group).erase(expr1_orig_group);
    uniqueUses().at(producer_group).pushBack(expr_new_group);
  }

  // Update unique definitinos of consumers
  IdGroups consumers;
  for (auto expr : std::vector<Expr*>{expr0, expr1}) {
    for (auto output_id : ir_utils::filterByType<IterDomain>(expr->outputs())) {
      consumers.pushBack(toGroup(output_id));
    }
  }

  for (auto consumer_group : consumers) {
    uniqueDefinitions().at(consumer_group).erase(expr0_orig_group);
    uniqueDefinitions().at(consumer_group).erase(expr1_orig_group);
    uniqueDefinitions().at(consumer_group).pushBack(expr_new_group);
  }
}

void IdGraph::mapIds(IterDomain* id0, IterDomain* id1) {
  if (id0 == id1) {
    return;
  }

  if (disjointIdSets().strictAreMapped(id0, id1)) {
    return;
  }
  // Definitions and uses are based on the groups of id0 and id1, don't merge
  // them into a single group until we grab all definitions and uses for later
  // processing.
  auto orig_id_group0 = disjointIdSet(id0).first;
  auto orig_id_group1 = disjointIdSet(id1).first;
  ExprGroups orig_defs0 = uniqueDefinitions(orig_id_group0);
  ExprGroups orig_defs1 = uniqueDefinitions(orig_id_group1);
  ExprGroups orig_uses0 = uniqueUses(orig_id_group0);
  ExprGroups orig_uses1 = uniqueUses(orig_id_group1);

  // Map the iter domains together before we traverse across definitions and
  // uses. Traversing definitions and uses could use the new property of id0 and
  // id1 being mapped.
  disjointIdSets().mapEntries(id0, id1);
  auto new_id_group = disjointIdSet(id0).first;

  unique_definitions_.erase(orig_id_group0);
  unique_definitions_.erase(orig_id_group1);
  unique_uses_.erase(orig_id_group0);
  unique_uses_.erase(orig_id_group1);

  unique_definitions_[new_id_group] = orig_defs0.computeUnion(orig_defs1);
  unique_uses_[new_id_group] = orig_uses0.computeUnion(orig_uses1);

  // Propagate on uses
  if (orig_uses0.size() > 0 || orig_uses1.size() > 0) {
    if (orig_uses0.size() > 0 && orig_uses1.size() > 0) {
      for (auto use_group_1 : orig_uses1) {
        if (orig_uses0.has(use_group_1)) {
          continue;
        }

        for (auto use_group_0 : orig_uses0) {
          auto use0 = use_group_0->front();
          auto use1 = use_group_1->front();
          maybeMapThroughExprs(use0, use1, true);
        }
      }
    }
  }

  // Propagate on definitions
  if (orig_defs0.size() > 0 || orig_defs1.size() > 0) {
    if (orig_defs0.size() > 0 && orig_defs1.size() > 0) {
      for (auto def_group_1 : orig_defs1) {
        if (orig_defs0.has(def_group_1)) {
          continue;
        }

        for (auto def_group_0 : orig_defs0) {
          auto def0 = def_group_0->front();
          auto def1 = def_group_1->front();
          maybeMapThroughExprs(def0, def1, false);
        }
      }
    }
  }
}

bool IdGraph::mapThroughExpr(Expr* first, Expr* second, bool forward) {
  if (first == nullptr || second == nullptr) {
    return false;
  }

  if (!exprsMap(first, second, forward)) {
    return false;
  }

  TORCH_INTERNAL_ASSERT(
      propagate_exprs_,
      "Asked to propagate expression mappings on a graph that has propagate_exprs_ disabled.");

  auto first_ids = ir_utils::filterByType<IterDomain>(
                       forward ? first->outputs() : first->inputs())
                       .vector();
  auto second_ids = ir_utils::filterByType<IterDomain>(
                        forward ? second->outputs() : second->inputs())
                        .vector();
  TORCH_INTERNAL_ASSERT(
      first_ids.size() == second_ids.size(),
      "This should be unreachable, if transformation expressions match, their number of inputs and outputs should as well.\n However found:\n",
      first->toString(),
      "\nand\n",
      second->toString());
  for (auto out_i : c10::irange(first_ids.size())) {
    mapIds(first_ids[out_i], second_ids[out_i]);
  }

  return true;
}

// TODO: Actually assert if self mapping found. Self mapping test is not correct
// yet.
void IterDomainGraphs::assertNoSelfMapping() {
  if (hasSelfMapping()) {
    TORCH_WARN(
        "IdGraphs thinks there's a self mapping in the problem. It's probably IdGraphs problem, not yours... ",
        std::get<0>(*self_mapping_info_)->toString(),
        ". ",
        std::get<3>(*self_mapping_info_),
        " domains, ",
        std::get<1>(*self_mapping_info_)->toString(),
        " and ",
        std::get<2>(*self_mapping_info_)->toString(),
        ", are mapped with each other.");
  }
}

void IdGraph::mapThroughTrivialExprs() {
  // Grab all expressions
  std::vector<Expr*> exprs;

  for (auto expr_group : disjointExprSets().disjointSets()) {
    for (auto expr : *expr_group) {
      exprs.push_back(expr);
    }
  }

  for (auto expr : exprs) {
    // If not trivial continue
    auto mapped_ids = IdGraph::isTrivialExpr(expr);
    if (mapped_ids.empty()) {
      continue;
    }

    // Map through trivial expressions
    for (auto mapped_id_group : mapped_ids) {
      for (auto id : mapped_id_group) {
        mapIds(mapped_id_group.front(), id);
      }
    }
  }
}

void IdGraph::removeTrivialExprs() {
  ExprGroups trivial_expr_groups;
  // This seems like it shouls just be a copy if.
  for (auto expr_group : disjointExprSets().disjointSets()) {
    if (isTrivialExprGroup(expr_group)) {
      trivial_expr_groups.pushBack(expr_group);
    }
  }

  // Clear out expressions that map inputs and outputs to the same group
  // from definitions and uses. They shouldn't be important in traversal, and
  // will break the terminal input/terminal output logic of traversal. Similar
  // to what's drafted in buildIndexGraph
  for (auto trivial_expr_group : trivial_expr_groups) {
    // Complexity of erase not good as both disjoint set and vector of unique
    // entries require a vector find to erase an entry.
    eraseExprGroup(trivial_expr_group);
  }
}

void IdGraph::mapThroughLoopSwizzles() {
  std::vector<Swizzle2D*> all_swizzles;

  for (auto expr_set : disjointExprSets().disjointSets()) {
    auto swizzles_in_expr_set = ir_utils::filterByType<Swizzle2D>(
        expr_set->vector().begin(), expr_set->vector().end());
    all_swizzles.insert(
        all_swizzles.end(),
        swizzles_in_expr_set.begin(),
        swizzles_in_expr_set.end());
  }

  for (auto swizzle : all_swizzles) {
    if (swizzle->swizzleMode() == SwizzleMode::Loop) {
      mapIds(swizzle->inX(), swizzle->outX());
      mapIds(swizzle->inY(), swizzle->outY());
    }
  }
}

// Complexity here is not great. We might want a better complexity version when
// erasing multiple expr_groups.
void IdGraph::eraseExprGroup(ExprGroup expr_group) {
  // Erase entries that exist in unique_definitions_ and unique_uses_
  for (auto id_group : disjointIdSets().disjointSets()) {
    // Make sure the entries exists
    TORCH_INTERNAL_ASSERT(
        unique_definitions_.find(id_group) != unique_definitions_.end(),
        "Broken definitions, couldn't find entry for id group, ",
        debug::toString(id_group, 0, true));
    TORCH_INTERNAL_ASSERT(
        unique_uses_.find(id_group) != unique_uses_.end(),
        "Broken uses, couldn't find entry for id group, ",
        debug::toString(id_group, 0, true));

    unique_definitions_[id_group].erase(expr_group);
    unique_uses_[id_group].erase(expr_group);
  }

  for (auto expr : *expr_group) {
    disjoint_exprs_.erase(expr);
  }
}

bool IdGraph::isTrivialExprGroup(ExprGroup expr_group) const {
  return !IdGroups(inputGroups(expr_group))
              .intersect(IdGroups(outputGroups(expr_group)))
              .empty();
}

IterDomainGraphs::IterDomainGraphs(
    const std::vector<Expr*>& exprs,
    const std::vector<TensorView*>& additional_tvs,
    bool allow_self_mapping) {
  build(exprs, additional_tvs);

  if (!allow_self_mapping) {
    assertNoSelfMapping();
  }
}

IterDomainGraphs::IterDomainGraphs(
    const std::vector<Expr*>& exprs,
    bool allow_self_mapping)
    : IterDomainGraphs(exprs, {}, allow_self_mapping) {}

IterDomainGraphs::IterDomainGraphs(Fusion* fusion, bool allow_self_mapping) {
  std::vector<TensorView*> inputs_and_outputs;
  {
    auto inp_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    inputs_and_outputs.insert(
        inputs_and_outputs.begin(), inp_tvs.begin(), inp_tvs.end());
  }
  {
    auto out_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
    inputs_and_outputs.insert(
        inputs_and_outputs.begin(), out_tvs.begin(), out_tvs.end());
  }

  build(fusion->exprs(), inputs_and_outputs);

  if (!allow_self_mapping) {
    assertNoSelfMapping();
  }
}

const IdGraph& IterDomainGraphs::idGraph(IdMappingMode mode) const {
  auto graph_it = id_graphs_.find(mode);
  TORCH_INTERNAL_ASSERT(graph_it != id_graphs_.end());
  return graph_it->second;
}

IdGraph& IterDomainGraphs::idGraph(IdMappingMode mode) {
  auto graph_it = id_graphs_.find(mode);
  TORCH_INTERNAL_ASSERT(graph_it != id_graphs_.end());
  return graph_it->second;
}

Expr* IterDomainGraphs::idUse(IterDomain* id) const {
  auto use_it = id_uses_.find(id);
  if (use_it == id_uses_.end()) {
    return nullptr;
  }
  return use_it->second.front();
}

Expr* IterDomainGraphs::idDef(IterDomain* id) const {
  auto def_it = id_definitions_.find(id);
  if (def_it == id_definitions_.end()) {
    return nullptr;
  }
  return def_it->second.front();
}

namespace {

// Returns the first pair of id's in ids detected to match eachother on the
// permissive map of the ID graph. TODO: what this is really looking for is if
// there's any overlapping between the iter domains in the provided set.
//
// i.e. if we have:
// tv0 = arange(6).view({3, 2})
// tv1 = tv0[3, 2].t()
// tv2 = tv0[3, 2].view({2, 3})
// tv3 = tv1 + tv2
//
// Then we can see this overlap in the tv3 expression as:
//
// tv0 = { {0, 1, 2},
//         {3, 4, 5} }
//
// tv1 = { {0, 3},
//         {1, 4},
//         {2, 5} }
//
// tv2 = { {0, 1},
//         {2, 3},
//         {4, 5} }
//
// The elements in tv1 {3, 1, 4, 2}, map respectively to the elements in tv2
// {1, 2, 3, 4}. The reason this is so important is it means that generating
// tv3 is no longer a trivially parallelizable problem (if we include the dag
// all the way to tv0). So tv0's axes cannot be inlined across both the tv0
// and tv1 path. This breaks some assumptions we have today in schedulers that
// will assume tv2 can be trivially inlined/parallelized. Instead we'd need to
// take into consideration the effective communication going on here, so that
// we pull multiple values of tv0 to compute tv3.
c10::optional<std::pair<IterDomain*, IterDomain*>> detectMappablePair(
    const std::vector<IterDomain*>& ids,
    const IterDomainGraphs& id_graph,
    IdMappingMode mode) {
  for (auto id1 : ids) {
    for (auto id2 : ids) {
      if (id1 == id2) {
        continue;
      }
      if (id_graph.idGraph(mode).disjointIdSets().permissiveAreMapped(
              id1, id2)) {
        return std::make_pair(id1, id2);
      }
    }
  }

  return {};
}

// It is assumed that for any tensor represented by a list of domains,
// those domains should never be mapped with each other. It may be
// possible to lift this assumption, but it's unclear if it could
// matter in practice.
c10::optional<std::tuple<TensorView*, IterDomain*, IterDomain*, std::string>>
findFirstSelfMapping(
    const std::vector<TensorView*>& all_tvs,
    const IterDomainGraphs& id_graph) {
  for (auto tv : all_tvs) {
    // For each tensor, make sure root, rfactor and leaf domains
    // should not include domains that are mapped with another domain
    // in the same set of domains. This may be overly conservative,
    // and it maybe enough to check the root domains.

    // Root domains
    auto self_mappped_root_pair =
        detectMappablePair(tv->getRootDomain(), id_graph, IdMappingMode::EXACT);
    if (self_mappped_root_pair.has_value()) {
      return std::make_tuple(
          tv,
          self_mappped_root_pair->first,
          self_mappped_root_pair->second,
          "Root");
    }

    // Rfactor domains
    if (tv->hasRFactor()) {
      auto self_mappped_rf_pair = detectMappablePair(
          tv->getRFactorDomain(), id_graph, IdMappingMode::EXACT);
      if (self_mappped_rf_pair.has_value()) {
        return std::make_tuple(
            tv,
            self_mappped_rf_pair->first,
            self_mappped_rf_pair->second,
            "RFactor");
      }
    }

    // Leaf domains
    auto self_mappped_leaf_pair =
        detectMappablePair(tv->domain()->leaf(), id_graph, IdMappingMode::LOOP);
    if (self_mappped_leaf_pair.has_value()) {
      return std::make_tuple(
          tv,
          self_mappped_leaf_pair->first,
          self_mappped_leaf_pair->second,
          "Leaf");
    }
  }
  return c10::nullopt;
}

} // namespace

void IterDomainGraphs::buildIterDomainDefinitionsAndUses(
    const std::vector<TensorView*>& all_tvs) {
  for (auto tv : all_tvs) {
    VectorOfUniqueEntries<IterDomain*> root_domain_ids{
        tv->getRootDomain().begin(), tv->getRootDomain().end()};

    auto all_ids = ir_utils::allIDsOf(tv);

    // Check is this domain is a consumer of a view-like operation
    bool view_like_domain = tv->domain()->hasViewLikeRFactor();

    for (auto id : all_ids) {
      // Check if this id is a view like rfactor id
      if (view_like_domain && id->isRFactorProduct()) {
        // If the tensor domain is a view like domain, and the iteration
        // domain is marked as an rfactor product and is in the rfactor
        // domain, it's a view like rfactor iteration domain
        const auto& rfactor_domain = tv->domain()->maybeRFactor();
        if (std::find(rfactor_domain.begin(), rfactor_domain.end(), id) !=
            rfactor_domain.end()) {
          view_rfactor_ids_.emplace(id);
        }
      }

      if (id_definitions_.find(id) == id_definitions_.end()) {
        id_definitions_[id] = {};
      }

      if (id_uses_.find(id) == id_uses_.end()) {
        id_uses_[id] = {};
      }

      auto def = id->definition();

      if (def == nullptr || root_domain_ids.has(id)) {
        continue;
      }

      if (id_definitions_.find(id) == id_definitions_.end()) {
        id_definitions_[id] = {};
      }
      id_definitions_.at(id).pushBack(def);

      auto inp_ids = ir_utils::filterByType<IterDomain>(def->inputs());
      for (auto inp_id : inp_ids) {
        if (id_uses_.find(inp_id) == id_uses_.end()) {
          id_uses_[inp_id] = {};
        }
        id_uses_.at(inp_id).pushBack(def);
      }
    }
  }
}

std::string IterDomainGraphs::toString() const {
  // Figure out which graphs are already initialized to make sure we add the new
  // expression to them.
  std::vector<IdMappingMode> initialized_modes;
  for (auto mode : kIdMappingModes) {
    auto graph_it = id_graphs_.find(mode);
    if (graph_it == id_graphs_.end()) {
      continue;
    }

    auto& graph = graph_it->second;
    if (graph.disjointIdSets().disjointSetMap().empty()) {
      continue;
    }

    initialized_modes.push_back(mode);
  }

  std::stringstream ss;
  ss << "IterDomainGraphs { \n";
  for (auto mode : initialized_modes) {
    std::stringstream ss;
    ss << "  IdGraph " << mode << "{ \n";
    ss << "  Disjoint Ids:\n"
       << debug::idGroupsString(idGraph(mode), 2)
       << "\n  Disjoint Expression groups:\n"
       << debug::exprGroupsString(idGraph(mode), 2) << std::endl;
    ss << "   } IdGraph\n" << std::endl;
    return ss.str();
  }
  ss << " } IterDomainGraphs\n" << std::endl;
  return ss.str();
}

// Replay Expr but with the inputs provided.
Expr* IterDomainGraphs::addReplayAs(
    std::vector<IterDomain*> new_inputs,
    Expr* expr) {
  // Figure out which graphs are already initialized to make sure we add the new
  // expression to them.
  std::vector<IdMappingMode> initialized_modes;
  for (auto mode : kIdMappingModes) {
    auto graph_it = id_graphs_.find(mode);
    if (graph_it == id_graphs_.end()) {
      continue;
    }

    auto& graph = graph_it->second;
    if (graph.disjointIdSets().disjointSetMap().empty()) {
      continue;
    }

    initialized_modes.push_back(mode);
  }

  auto orig_inputs = ir_utils::filterByType<IterDomain>(expr->inputs());
  std::vector<IterDomain*> orig_input_ids(
      orig_inputs.begin(), orig_inputs.end());

  if (std::any_of(
          new_inputs.begin(),
          new_inputs.end(),
          [](IterDomain* id) { return id->isReduction(); }) &&
      std::any_of(new_inputs.begin(), new_inputs.end(), [](IterDomain* id) {
        return !id->isReduction();
      })) {
    // Inputs have mismatched type, replace new_inputs
    decltype(new_inputs) tmp_inputs;
    std::swap(tmp_inputs, new_inputs);
    for (auto tmp_input : tmp_inputs) {
      new_inputs.push_back(
          IterDomainBuilder(tmp_input).iter_type(IterType::Iteration).build());
      id_definitions_[new_inputs.back()];
      id_uses_[new_inputs.back()];
      for (auto mode : initialized_modes) {
        idGraph(mode).initializeId(new_inputs.back(), {}, {});
        idGraph(mode).mapIds(new_inputs.back(), tmp_input);
      }
    }
  }

  {
    TORCH_INTERNAL_ASSERT(
        new_inputs.size() == orig_input_ids.size(),
        "Invalid number of inputs: ",
        new_inputs.size(),
        " does not match number of iter domain inputs for ",
        expr->toString());

    VectorOfUniqueEntries<IterDomain*> all_inputs{
        orig_input_ids.begin(), orig_input_ids.end()};

    all_inputs.pushBack(VectorOfUniqueEntries<IterDomain*>{
        new_inputs.begin(), new_inputs.end()});

    for (auto mode : initialized_modes) {
      for (auto inp : all_inputs) {
        TORCH_INTERNAL_ASSERT(
            idGraph(mode).disjointIdSet(inp).second,
            "All inputs for replay need to be initialized in all graphs, ",
            inp->toString(),
            " was not found in mode: ",
            mode);
      }
    }
  }

  // Create the new expression with provided inputs
  auto replay = ReplayTransform::replayAs(new_inputs, expr);

  for (auto out_id : ir_utils::filterByType<IterDomain>(replay->outputs())) {
    id_definitions_[out_id].pushBack(replay);
    id_uses_[out_id];
  }

  // Add the expression to the uses of the inputs
  for (auto inp_id : ir_utils::filterByType<IterDomain>(replay->inputs())) {
    id_definitions_[inp_id];
    id_uses_[inp_id].pushBack(replay);
  }

  // Initialize output iter domains in the graphs
  for (auto mode : initialized_modes) {
    idGraph(mode).disjointExprSets().initializeSet(replay);
    auto replay_group = idGraph(mode).disjointExprSet(replay).first;

    // Initialize output ids in map
    for (auto out_id : ir_utils::filterByType<IterDomain>(replay->outputs())) {
      idGraph(mode).initializeId(out_id, {replay}, {});
    }

    // Update uses of the inputs in the graphs
    for (auto inp_id : ir_utils::filterByType<IterDomain>(replay->inputs())) {
      auto inp_group = idGraph(mode).disjointIdSet(inp_id).first;
      idGraph(mode).uniqueUses().at(inp_group).pushBack(replay_group);
    }

    // Propagate through all the uses of the iter domain groups of the inputs
    // with the new expression.
    auto& graph = idGraph(mode);
    // Gather all use expressions from inputs
    VectorOfUniqueEntries<Expr*> representative_uses;
    for (auto inp : new_inputs) {
      auto uses_pair =
          graph.iterDomainGroupUses(graph.disjointIdSet(inp).first);
      if (uses_pair.second) {
        for (auto use_group : uses_pair.first) {
          representative_uses.pushBack(use_group->front());
        }
      }
    }

    for (auto rep_use : representative_uses) {
      graph.maybeMapThroughExprs(rep_use, replay, true);
    }
  }

  return replay;
}

// Generate a new expr with the IterDomain inputs/outputs replaced based on map.
// Replaced inputs/outputs should almost exact match with provided expr.
Expr* IterDomainGraphs::addExprWithReplacement(
    const std::unordered_map<IterDomain*, IterDomain*>& old_2_new_ids,
    Expr* old_expr) {
  // Figure out which graphs are already initialized to make sure we add the new
  // expression to them.
  std::vector<IdMappingMode> initialized_modes;
  for (auto mode : kIdMappingModes) {
    auto graph_it = id_graphs_.find(mode);
    if (graph_it == id_graphs_.end()) {
      continue;
    }

    auto& graph = graph_it->second;
    if (graph.disjointIdSets().disjointSetMap().empty()) {
      continue;
    }

    initialized_modes.push_back(mode);
  }

  // We will fill this map for every IterDomain in input and output.
  std::unordered_map<IterDomain*, IterDomain*> replacement_map = old_2_new_ids;

  // Validate replacement map. Make sure the keys are an input or output
  for (auto replacement_entry : replacement_map) {
    TORCH_INTERNAL_ASSERT(
        std::find(
            old_expr->inputs().begin(),
            old_expr->inputs().end(),
            replacement_entry.first) != old_expr->inputs().end() ||
            std::find(
                old_expr->outputs().begin(),
                old_expr->outputs().end(),
                replacement_entry.first) != old_expr->outputs().end(),
        "Wanted to replace ",
        replacement_entry.first->toString(),
        " however the is not an input or output of:\n",
        old_expr->toString());
  }

  // If all inputs and or all output were replaced
  bool all_inps_replaced = true;
  bool all_outs_replaced = true;
  {
    for (auto inp_id : ir_utils::filterByType<IterDomain>(old_expr->inputs())) {
      if (replacement_map.find(inp_id) == replacement_map.end()) {
        all_inps_replaced = false;
        replacement_map[inp_id] = inp_id->cloneWithoutRFactor();
      }
    }

    for (auto out_id :
         ir_utils::filterByType<IterDomain>(old_expr->outputs())) {
      if (replacement_map.find(out_id) == replacement_map.end()) {
        all_outs_replaced = false;
        replacement_map[out_id] = out_id->cloneWithoutRFactor();
      }
    }

    TORCH_INTERNAL_ASSERT(
        (all_inps_replaced || all_outs_replaced),
        "Either all the inputs or all the outputs need to be replaced when using this function.");

    for (auto mode : initialized_modes) {
      for (auto inp_or_out_id : all_inps_replaced
               ? ir_utils::filterByType<IterDomain>(old_expr->inputs())
               : ir_utils::filterByType<IterDomain>(old_expr->outputs())) {
        TORCH_INTERNAL_ASSERT(
            idGraph(mode).disjointIdSet(inp_or_out_id).second,
            "Expected ",
            inp_or_out_id->toString(),
            " to be initialized in graph mode: ",
            mode);
      }
    }
  }

  // Create the new expression with provided outputs
  auto replay = ReplacementTransformCloner::clone(replacement_map, old_expr);

  // Add new output iter domains to id_definitions_/id_uses_ of IdGraphs
  for (auto out_id : ir_utils::filterByType<IterDomain>(replay->outputs())) {
    id_definitions_[out_id].pushBack(replay);
    id_uses_[out_id];
  }

  // Add new input iter domains to id_definitions_/id_uses_ of IdGraphs
  for (auto inp_id : ir_utils::filterByType<IterDomain>(replay->inputs())) {
    id_definitions_[inp_id];
    id_uses_[inp_id].pushBack(replay);
  }

  // Update all the initialized graph mappings
  for (auto mode : initialized_modes) {
    auto& graph = idGraph(mode);

    graph.disjointExprSets().initializeSet(replay);
    auto replay_group = graph.disjointExprSet(replay).first;

    // Initialize any non-existant input ids, update existing ones
    for (auto inp_id : ir_utils::filterByType<IterDomain>(replay->inputs())) {
      if (!graph.disjointIdSets().mappingExists(inp_id)) {
        // inp_id is not initialized in the map, initialize it
        graph.initializeId(inp_id, {}, {replay});
      } else {
        // Update unique uses of existing input ids
        auto inp_group = graph.disjointIdSet(inp_id).first;
        graph.uniqueUses()[inp_group].pushBack(replay_group);
      }
    }

    // Initialize any non-existant output ids, update existing ones
    for (auto out_id : ir_utils::filterByType<IterDomain>(replay->outputs())) {
      if (!graph.disjointIdSets().mappingExists(out_id)) {
        // out_id is not initialized in the map, initialize it
        graph.initializeId(out_id, {replay}, {});
      } else {
        // out_id is already initialized, add the replay as a unique definition
        // of its group
        auto out_group = graph.disjointIdSet(out_id).first;
        graph.uniqueDefinitions()[out_group].pushBack(replay_group);
      }
    }

    // If the inputs were replaced we want to map through forward the newly
    // added expression. If the outputs were replaced we want to map through
    // backwards the newly added expression.

    // Forward
    VectorOfUniqueEntries<Expr*> representative_uses;
    for (auto in : ir_utils::filterByType<IterDomain>(replay->inputs())) {
      auto uses_pair = graph.iterDomainGroupUses(graph.disjointIdSet(in).first);
      if (uses_pair.second) {
        for (auto use_group : uses_pair.first) {
          if (use_group == replay_group) {
            continue;
          }
          representative_uses.pushBack(use_group->front());
        }
      }
    }

    for (auto rep_use : representative_uses) {
      graph.maybeMapThroughExprs(rep_use, replay, true);
    }

    // Backwards
    VectorOfUniqueEntries<Expr*> representative_defs;
    for (auto out : ir_utils::filterByType<IterDomain>(replay->outputs())) {
      auto defs_pair =
          graph.iterDomainGroupDefinitions(graph.disjointIdSet(out).first);
      if (defs_pair.second) {
        for (auto def_group : defs_pair.first) {
          if (def_group == replay_group) {
            continue;
          }
          representative_defs.pushBack(def_group->front());
        }
      }
    }

    for (auto rep_def : representative_defs) {
      graph.maybeMapThroughExprs(rep_def, replay, false);
    }
  }
  return replay;
}

// Clone provided iter domain and return the new copy. Map that copy in relevant
// maps.
IterDomain* IterDomainGraphs::cloneIterDomain(IterDomain* id) {
  // Figure out which graphs are already initialized to make sure we add the new
  // expression to them.
  std::vector<IdMappingMode> initialized_modes;
  for (auto mode : kIdMappingModes) {
    auto graph_it = id_graphs_.find(mode);
    if (graph_it == id_graphs_.end()) {
      continue;
    }

    auto& graph = graph_it->second;
    if (graph.disjointIdSets().disjointSetMap().empty()) {
      continue;
    }

    initialized_modes.push_back(mode);
  }

  auto id_copy = id->cloneWithoutRFactor();

  id_uses_[id_copy] = {};
  id_definitions_[id_copy] = {};

  for (auto mode : initialized_modes) {
    idGraph(mode).initializeId(id_copy, {}, {});
    idGraph(mode).mapIds(id, id_copy);
  }

  return id_copy;
}

IdGraph IterDomainGraphs::initializeIdGraph() {
  IdGraph id_graph;

  for (auto definition_entry : id_definitions_) {
    auto id = definition_entry.first;
    auto defs = definition_entry.second;
    auto uses_it = id_uses_.find(id);
    TORCH_INTERNAL_ASSERT(
        uses_it != id_uses_.end(),
        "Failed to initialize id: ",
        id->toString(),
        " as it's missing a definition entry.");
    id_graph.initializeId(id, defs, uses_it->second);
  }

  return id_graph;
}

void IterDomainGraphs::buildExactMap(const std::vector<Expr*>& exprs) {
  for (auto expr : exprs) {
    TensorView* c_tv = ir_utils::getTvOutput(expr);

    auto all_tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());

    // Map siblings, as all other tv output domains must match the first tv
    // outputs domain.
    std::deque<TensorView*> other_tv_outputs(
        all_tv_outputs.begin(), all_tv_outputs.end());
    other_tv_outputs.pop_front();

    for (auto other_tv_output : other_tv_outputs) {
      // Sibling tv's must be exactly mapped with eachother so simply zip
      // their leaf iter domains.

      TORCH_INTERNAL_ASSERT(
          other_tv_output->getRootDomain().size() ==
              c_tv->getRootDomain().size(),
          "Multiple outputs with mismatched TV domains is not supported.");

      for (auto domain_i : c10::irange(c_tv->getRootDomain().size())) {
        auto c_id = c_tv->getRootDomain()[domain_i];
        auto o_id = other_tv_output->getRootDomain()[domain_i];
        idGraph(IdMappingMode::EXACT).mapIds(o_id, c_id);
      }
    }

    // Map producer-consumer relationships based on the root domain map
    auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    for (auto p_tv : tv_inputs) {
      // For exact mapings do not map any broadcast dimensions to
      // non-broadcast dimensions. Prevent any broadcasted axes being mapped
      // to non-broadcasted axes.
      auto exact_c2p_root_map =
          PairwiseRootDomainMap(p_tv, c_tv)
              .mapConsumerToProducer(c_tv->domain(), p_tv->domain());

      for (auto c_id : getSortedKeys(exact_c2p_root_map, Statement::lessThan)) {
        auto p_id = exact_c2p_root_map.at(c_id);
        idGraph(IdMappingMode::EXACT).mapIds(c_id, p_id);
      }
    }

    idGraph(IdMappingMode::EXACT).mapThroughLoopSwizzles();
  }
}

void IterDomainGraphs::buildPermissiveMap(const std::vector<Expr*>& exprs) {
  idGraph(IdMappingMode::PERMISSIVE) = idGraph(IdMappingMode::ALMOSTEXACT);

  for (auto expr : exprs) {
    // Multiple outputs are already mapped, we can ignore all but the first
    // consumer given they have to be replayed in the same exact way
    // Multiple outputs are already mapped, we can ignore all but the first
    // consumer given they have to be replayed in the same exact way
    TensorView* c_tv = ir_utils::getTvOutput(expr);

    auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());

    for (auto p_tv : tv_inputs) {
      auto p_ids_vec = ir_utils::allIDsOf(p_tv);
      auto c_ids_vec = ir_utils::allIDsOf(c_tv);
      std::unordered_set<IterDomain*> p_ids(p_ids_vec.begin(), p_ids_vec.end());
      std::unordered_set<IterDomain*> c_ids(c_ids_vec.begin(), c_ids_vec.end());

      ForwardingInfo permissive_forwarding(p_tv, c_tv);
      for (auto entry : permissive_forwarding.producer_forwarding_map) {
        // std::cout << "Permissive producer forwarding: "
        //           << entry.first->toString() << " -> "
        //           << entry.second->toString() << std::endl;
        idGraph(IdMappingMode::PERMISSIVE).mapIds(entry.first, entry.second);
      }

      // TODO: Should this just get rolled up in the forwarding map now?
      // TODO: Why should IDs be mapped to their compliments? Is this right?
      for (auto entry : permissive_forwarding.producer_compliment_map) {
        for (auto entry_2 : entry.second) {
          // std::cout << "Permissive producer compliment: "
          //           << entry.first->toString() << " -> " <<
          //           entry_2->toString()
          //           << std::endl;
          idGraph(IdMappingMode::PERMISSIVE).mapIds(entry.first, entry_2);
        }
      }

      for (auto entry : permissive_forwarding.consumer_forwarding_map) {
        // std::cout << "Permissive consumer forwarding: "
        //           << entry.first->toString() << " -> "
        //           << entry.second->toString() << std::endl;
        idGraph(IdMappingMode::PERMISSIVE).mapIds(entry.first, entry.second);
      }

      // TODO: Should this just get rolled up in the forwarding map now?
      // TODO: Why should IDs be mapped to their compliments? Is this right?
      for (auto entry : permissive_forwarding.consumer_compliment_map) {
        for (auto entry_2 : entry.second) {
          // std::cout << "Permissive consumer compliment: "
          //           << entry.first->toString() << " -> " <<
          //           entry_2->toString()
          //           << std::endl;
          idGraph(IdMappingMode::PERMISSIVE).mapIds(entry.first, entry_2);
        }
      }

      auto permissive_c2p_root_map = PairwiseRootDomainMap(p_tv, c_tv);

      for (auto entry : permissive_c2p_root_map.mapConsumerToProducer(
               c_tv->domain(), p_tv->domain())) {
        idGraph(IdMappingMode::PERMISSIVE).mapIds(entry.first, entry.second);
      }
    }
  }
  idGraph(IdMappingMode::PERMISSIVE).mapThroughLoopSwizzles();
}

void IterDomainGraphs::buildAlmostExactMap() {
  // Build almost exact map by forwarding through broadcast axes
  idGraph(IdMappingMode::ALMOSTEXACT) = idGraph(IdMappingMode::EXACT);
  idGraph(IdMappingMode::ALMOSTEXACT).mapThroughTrivialExprs();
}

// TODO: Reenable after reenabling parallel propagation.
//        propagateLoopPTypes
void IterDomainGraphs::validatePTypes(
    const std::vector<TensorView*>& all_tvs) const {
  // VectorOfUniqueEntries<IterDomain*> leaf_ids;
  // for (auto tv : all_tvs) {
  //   leaf_ids.pushBack(tv->domain()->leaf());
  // }

  // for (const auto& disjoint_set :
  //      idGraph(IdMappingMode::EXACT).disjointIdSets().disjointSets()) {
  //   for (auto id : disjoint_set->vector()) {
  //     auto id_ptype = id->getParallelType();

  //     TORCH_INTERNAL_ASSERT(
  //         leaf_ids.has(id) || id_ptype == ParallelType::Serial,
  //         "Invalid parallelization of non leaf iter domain: ",
  //         id->toString());
  //   }
  // }
}

void IterDomainGraphs::propagateLoopPTypes() const {
  for (const auto& loop_disjoint_set :
       idGraph(IdMappingMode::LOOP).disjointIdSets().disjointSets()) {
    ParallelType common_ptype = ParallelType::Serial;
    for (auto id : loop_disjoint_set->vector()) {
      auto id_ptype = id->getParallelType();

      TORCH_INTERNAL_ASSERT(
          id_ptype == common_ptype || id_ptype == ParallelType::Serial ||
              common_ptype == ParallelType::Serial,
          "Issue validating parallel type disjoint ptype is, ",
          common_ptype,
          " but found in the set the id: ",
          id->toString());

      common_ptype =
          common_ptype == ParallelType::Serial ? id_ptype : common_ptype;
    }

    for (auto id : loop_disjoint_set->vector()) {
      id->parallelize(common_ptype);
    }
  }
}

namespace {
struct StatefulLoweringInfo {
  // Tracks all p2c mappings in permissive maps even those not inlined between
  // producer and consumer
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      p2c_permissive_maps;

  // All consumer ids in a deterministic order (ignores fusion->inputs())
  VectorOfUniqueEntries<IterDomain*> ordered_c_ids;

  // p2c mappings through the fusion within (including dependencies of) inlined
  // leaf domains.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      p2c_ca_permissive_maps;

  // All producer ids within (including dependencies of) inlined leaf domains,
  // used for deterministic order
  VectorOfUniqueEntries<IterDomain*> ordered_p_ca_ids;

  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      p2c_root_broadcast_resolution_map;
};

// Returns the root producer iteration domains that are resolved by provided
// consumer
std::unordered_map<IterDomain*, IterDomain*> resolvedRootBroadcasts(
    TensorView* producer,
    TensorView* consumer) {
  auto p2c_map =
      PairwiseRootDomainMap(producer, consumer)
          .mapProducerToConsumer(producer->domain(), consumer->domain());

  std::unordered_map<IterDomain*, IterDomain*> resolved_bcast_map;
  for (const auto& kv : p2c_map) {
    auto p_id = kv.first;
    // Ignore non-broadcast dims
    if (!p_id->isBroadcast()) {
      continue;
    }
    auto c_id = kv.second;
    // If the consumer ID is a reduction (i.e., a trivial
    // reduction), do not consider it's concretized.
    if (c_id->isBroadcast() || c_id->isReduction()) {
      continue;
    }

    resolved_bcast_map[p_id] = c_id;
  }
  return resolved_bcast_map;
}

StatefulLoweringInfo buildInfo(
    const std::vector<Expr*>& exprs,
    const IdGraph& exact_graph,
    const IdGraph& permissive_graph) {
  StatefulLoweringInfo info;
  // Grab inlining relationships
  for (auto expr : exprs) {
    for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
      auto producer_root = producer->getMaybeRFactorDomain();
      auto producer_domain = producer->domain()->leaf();

      // Grab all iteration domains in producer that its compute at iter domains
      // depend on.
      VectorOfUniqueEntries<IterDomain*> all_producer_ca_deps;
      {
        auto ca_dep_vals = DependencyCheck::getAllValsBetween(
            {producer_root.begin(), producer_root.end()},
            {producer_domain.begin(),
             producer_domain.begin() + producer->getComputeAtPosition()});
        auto ca_deps_filter = ir_utils::filterByType<IterDomain>(ca_dep_vals);

        all_producer_ca_deps.insert(
            ca_deps_filter.begin(), ca_deps_filter.end());
      }
      // std::cout << "Producer: " << producer->toString() << "\n  "
      //           << all_producer_ca_deps.toString() << std::endl;

      info.ordered_p_ca_ids.pushBack(all_producer_ca_deps);

      for (auto consumer :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        auto resolved_bcast_map = resolvedRootBroadcasts(producer, consumer);
        for (auto entry : resolved_bcast_map) {
          info.p2c_root_broadcast_resolution_map[entry.first].pushBack(
              entry.second);
          for (auto other_exact_bcast :
               *exact_graph.disjointIdSet(entry.first).first) {
            if (all_producer_ca_deps.has(other_exact_bcast)) {
              info.p2c_root_broadcast_resolution_map[other_exact_bcast]
                  .pushBack(entry.second);
            }
          }
        }

        auto all_producer_ids = ir_utils::allIDsOf(producer);
        auto all_consumer_ids = ir_utils::allIDsOf(consumer);
        info.ordered_c_ids.pushBack(all_consumer_ids);

        auto p2c_permissive_map = permissive_graph.buildMapBetween(
            all_producer_ids, all_consumer_ids);

        for (auto entry : p2c_permissive_map) {
          if (entry.second.size() == 0) {
            continue;
          }
          if (all_producer_ca_deps.has(entry.first)) {
            info.p2c_ca_permissive_maps[entry.first].pushBack(entry.second);
          }
          info.p2c_permissive_maps[entry.first].pushBack(entry.second);
        }

        for (auto entry : p2c_permissive_map) {
          if (entry.second.size() == 0) {
            continue;
          }
          info.p2c_permissive_maps[entry.first].pushBack(entry.second);
        }
      }
    }
  }
  return info;
}

} // namespace

void IterDomainGraphs::build(
    const std::vector<Expr*>& exprs,
    const std::vector<TensorView*>& additional_tvs) {
  // Initialize the required sets as if a permissive relationship is never
  // found, then querying an empty permissive map will fail later.
  // Initialize disjoint sets
  for (auto mode : kIdMappingModes) {
    id_graphs_[mode] = IdGraph();
  }

  std::vector<Expr*> tv_exprs;

  std::copy_if(
      exprs.begin(), exprs.end(), std::back_inserter(tv_exprs), [](Expr* expr) {
        TORCH_INTERNAL_ASSERT(expr != nullptr);
        return ir_utils::isTvOp(expr);
      });

  auto all_tvs = ir_utils::allTvsOfExprs(tv_exprs);
  if (additional_tvs.size() > 0) {
    std::unordered_set<TensorView*> all_added_tvs(
        all_tvs.begin(), all_tvs.end());
    for (auto additional_tv : additional_tvs) {
      if (all_added_tvs.find(additional_tv) == all_added_tvs.end()) {
        all_tvs.push_back(additional_tv);
      }
    }
  }

  if (all_tvs.empty()) {
    return;
  }

  FusionGuard fg(all_tvs.front()->fusion());
  // Add uses and definitions to all iter domains.
  buildIterDomainDefinitionsAndUses(all_tvs);

  // Initialize the maps with all the IterDomains used in the provded
  // expressions.
  idGraph(IdMappingMode::EXACT) = initializeIdGraph();

  buildExactMap(tv_exprs);
  buildAlmostExactMap();
  buildPermissiveMap(tv_exprs);

  // Permissive graph needs the trivial exprs from the almost exact graph to
  // build correctly. Once built though we can remove the trivial expressions
  // from the almost exact graph.
  idGraph(IdMappingMode::ALMOSTEXACT).removeTrivialExprs();

  // Only build loop map during lowering
  if (FusionGuard::getCurFusion()->isA<kir::Kernel>()) {
    validatePTypes(all_tvs);

    // FusionGuard::getCurFusion()->print(std::cout, true);

    StatefulLoweringInfo info = buildInfo(
        tv_exprs,
        idGraph(IdMappingMode::EXACT),
        idGraph(IdMappingMode::PERMISSIVE));

    initializeLoopMap(info);
    // std::cout << "Loop groups: "
    //           << debug::idGroupsString(idGraph(IdMappingMode::LOOP))
    //           << std::endl;

    // std::cout << "Promoted groups: "
    //           << debug::idGroupsString(idGraph(IdMappingMode::LOOP))
    //           << std::endl;

    // Initial propagation of parallel types for inlined iter domains. Each time
    // new expressions are replayed this needs to be run. The disjoint sets in
    // the loop graph can only be joined after this point.
    // propagateLoopPTypes();

    auto iel_promotion_map = buildInlinePromotions(info);
    // propagateLoopPTypes();

    // Find loops that need to be promoted because of broadcast resolution,
    // figure out what that resolution should look like, compute IDs for it if
    // necessary.
    iel_promotion_map =
        buildLoopPromotionMap(tv_exprs, info, iel_promotion_map);
    // Loop map potentialy changed changed, as we could have replayed
    // expressions. Re-propagate parallel types.
    // propagateLoopPTypes();

    // This pass still doesn't work, disable for now in case it's disruptive to
    // tests.
    /*
    // Find loops that need to be promoted because of broadcast resolution,
    // figure out what that resolution should look like, compute IDs for it if
    // necessary.
    auto leaf_id_promo_map =
        buildIndexGraph(tv_exprs, all_tvs, info, iel_promotion_map);
    // Make sure we update ptypes onto the index leaf iter domains
    propagateLoopPTypes();
    */
  }

  // Debug, make sure there's no self mapping in TensorView's during lowering
  // that would invalidate lowering assumptions.
  self_mapping_info_ = findFirstSelfMapping(all_tvs, *this);
}

VectorOfUniqueEntries<IterDomain*> IterDomainGraphs::computeTerminalLoopIds(
    const StatefulLoweringInfo info) {
  VectorOfUniqueEntries<IterDomain*> terminal_loop_ids;
  for (auto group :
       idGraph(IdMappingMode::LOOP).disjointIdSets().disjointSets()) {
    if (group->size() == 1) {
      terminal_loop_ids.pushBack(group->front());
    }

    // Don't select producer iter domains
    for (auto loop_id : *group) {
      if (info.p2c_ca_permissive_maps.find(loop_id) !=
          info.p2c_ca_permissive_maps.end()) {
        continue;
      }

      auto uses_it = id_uses_.find(loop_id);
      if (uses_it == id_uses_.end()) {
        terminal_loop_ids.pushBack(loop_id);
        continue;
      }

      // If there's an output group that is not in the same group, then it's id
      // consumer terminal. Also if there's no output groups it's id consumer
      // terminal.
      bool all_outs_in_loop_group = uses_it->second.size() == 0 ? false : true;
      for (auto use : uses_it->second) {
        for (auto out_id : ir_utils::filterByType<IterDomain>(use->outputs())) {
          auto out_loop_set_pair =
              idGraph(IdMappingMode::LOOP).disjointIdSet(out_id);
          TORCH_INTERNAL_ASSERT(out_loop_set_pair.second);
          if (group != out_loop_set_pair.first) {
            all_outs_in_loop_group = false;
          }
        }
      }

      if (!all_outs_in_loop_group) {
        terminal_loop_ids.pushBack(loop_id);
      }
    }
  }
  return terminal_loop_ids;
}

IdGraph IterDomainGraphs::buildIntersection(
    const IdGraph& graph0,
    const IdGraph& graph1,
    bool propagate_exprs) {
  auto intersection = initializeIdGraph();
  if (!propagate_exprs) {
    intersection.disableExprPropagation();
  }
  for (auto exact_group : graph0.disjointIdSets().disjointSets()) {
    auto set_size = exact_group->size();
    for (auto id0_i : c10::irange(set_size)) {
      auto id0 = exact_group->vector()[id0_i];
      for (auto id1_i = id0_i; id1_i < set_size; id1_i++) {
        auto id1 = exact_group->vector()[id1_i];
        // id0 and id1 map in the almost exact map, if they also map in the loop
        // graph, then add the mapping to the inersection
        if (graph1.disjointIdSets().strictAreMapped(id0, id1)) {
          intersection.mapIds(id0, id1);
        }
      }
    }
  }
  return intersection;
}

void IterDomainGraphs::initializeLoopMap(StatefulLoweringInfo& info) {
  idGraph(IdMappingMode::LOOP) = initializeIdGraph();
  // See Indexing20 example for why we shouldn't propagate when generating loop
  // groups
  idGraph(IdMappingMode::LOOP).disableExprPropagation();

  // Make sure this is called in a deterministic order. Build all inlined
  // relationships in loop graph.
  for (auto p_id : info.ordered_p_ca_ids) {
    auto entry_it = info.p2c_ca_permissive_maps.find(p_id);
    if (entry_it != info.p2c_ca_permissive_maps.end()) {
      auto c_ids = entry_it->second;
      for (auto c_id : c_ids) {
        idGraph(IdMappingMode::LOOP).mapIds(p_id, c_id);
      }
    }
  }
}

std::unordered_map<IdGroup, IterDomain*> IterDomainGraphs::
    buildInlinePromotions(StatefulLoweringInfo& info) {
  // Make an intersection of the exact and loop map. This will group together
  // entries in each loop group that are exact with eachother. This provides a
  // better graph to do promotion and replays.

  // It's tempting to use the intersection of the almost exact and loop, but we
  // need to model broadcast promotion, and if we have two tensors like:
  //
  // T1[i0, b1] = T0[i0]
  // T2[i0, b1] = T0[i0]
  //
  // Then resolution of:
  // T4 = T1[i0, b1] + T3[i0, i1]
  // T6 = T2[i0, b1] + T5[i0, i2]
  //
  // The almost exact map will map T1's and T2's b1 together, but they're being
  // resolved to i1 and i2 respectively. So we want to have separate entries so
  // we can have an easy to process promotion map.
  //
  // Loop is a permissive like map, it could have many entries, use the exact
  // map as the one we iterate on to reduce complexity as it hopefully has
  // smaller groups and this algorithm scales with the number of groups *
  // (number of entries in groups ^ 2)

  auto intersection_exact_loop_graph = buildIntersection(
      idGraph(IdMappingMode::EXACT), idGraph(IdMappingMode::LOOP), false);

  // Promotion logic is going to be on the intersection of the exact and loop
  // graph. We will generate a map on the entries of this graph so it's
  // important to not modify this graph moving forward, as that would invalidate
  // the map.
  //
  // iel stands for Intersection of the Exact and Loop graphs.
  std::unordered_map<IdGroup, IterDomain*> iel_promotion_map;

  // This should probably work just on terminating inputs, as we shouldn't be
  // able to modify a broadcast domain between root and rfactor which would be
  // required to resolve a non input broadcast domain. But for now leaving it as
  // traversal on all broadcast groups.
  for (auto iel_group :
       intersection_exact_loop_graph.disjointIdSets().disjointSets()) {
    if (!iel_group->front()->isBroadcast()) {
      continue;
    }

    // Collect all the exact groups of the resolutions of the broadcast id's
    IdGroups resolved_exact_groups;
    for (auto bcast_id : *iel_group) {
      auto p2c_root_broadcast_resolution_map_it =
          info.p2c_root_broadcast_resolution_map.find(bcast_id);

      if (p2c_root_broadcast_resolution_map_it ==
          info.p2c_root_broadcast_resolution_map.end()) {
        continue;
      }

      resolved_exact_groups.pushBack(
          idGraph(IdMappingMode::EXACT)
              .toGroups(p2c_root_broadcast_resolution_map_it->second));
    }

    // Collect all the exact groups in the loop set containing this iel_group
    auto loop_group_pair =
        idGraph(IdMappingMode::LOOP).disjointIdSet(iel_group->front());
    TORCH_INTERNAL_ASSERT(loop_group_pair.second);
    auto loop_group = loop_group_pair.first;
    auto loop_covered_exact_groups =
        idGraph(IdMappingMode::EXACT).toGroups(*loop_group);

    // The intersection of the exact groups that the broadcast domains can be
    // broadcasted to, and those that exist within the same loop groop are is
    // the promotion needed for this iel_group.
    auto loop_exact_resolved_intersection =
        resolved_exact_groups.intersect(loop_covered_exact_groups);

    if (loop_exact_resolved_intersection.empty()) {
      // No resolution
      continue;
    }

    if (loop_exact_resolved_intersection.size() > 1) {
      std::stringstream err_msg;

      err_msg
          << "Invalid multiple broadcast resolution within shared loops detected, group:\n  "
          << iel_group->toString() << "\nIs being broadcasted to:";

      for (auto entry : loop_exact_resolved_intersection) {
        err_msg << "\n  " << entry->toString();
      }
      TORCH_INTERNAL_ASSERT(false, err_msg.str());
    }

    // loop_exact_resolved_intersection.size() must be 1 at this point
    auto exact_resolution_group = loop_exact_resolved_intersection.front();

    VectorOfUniqueEntries<IterDomain*> resolved_ids =
        exact_resolution_group->intersect(*loop_group);
    auto promoted_iel_groups =
        intersection_exact_loop_graph.toGroups(resolved_ids);

    if (promoted_iel_groups.size() == 0) {
      continue;
    }

    if (promoted_iel_groups.size() > 1) {
      std::stringstream err_msg;

      err_msg
          << "Invalid multiple broadcast resolution within shared loops detected, group:\n  "
          << iel_group->toString() << "\nIs being broadcasted to:";

      for (auto entry : promoted_iel_groups) {
        err_msg << "\n  " << entry->toString();
      }
      TORCH_INTERNAL_ASSERT(false, err_msg.str());
    }

    iel_promotion_map[iel_group] = promoted_iel_groups.front()->front();
  }

  // std::cout << "Initial promotion map:" << std::endl;

  for (auto iel_group :
       intersection_exact_loop_graph.disjointIdSets().disjointSets()) {
    auto entry_it = iel_promotion_map.find(iel_group);
    if (entry_it == iel_promotion_map.end()) {
      continue;
    }
    // std::cout << "  " << entry_it->second->toString() << " <- "
    //           << entry_it->first->toString() << std::endl;
  }

  IdGraphStmtSort iel_stmt_sort(intersection_exact_loop_graph);

  // std::cout << "Initial promotion replay:" << std::endl;
  for (auto iel_expr : iel_stmt_sort.exprs()) {
    auto input_groups = intersection_exact_loop_graph.inputGroups(iel_expr);

    // Check if any inputs need promotion indicating this expr group needs to
    // be replayed with promoted inputs
    std::vector<IterDomain*> promoted_inputs;
    bool an_input_was_promoted = false;

    for (auto inp : input_groups) {
      auto inp_promo_it = iel_promotion_map.find(inp);
      if (inp_promo_it == iel_promotion_map.end()) {
        promoted_inputs.push_back(inp->front());
      } else {
        promoted_inputs.push_back(inp_promo_it->second);
        an_input_was_promoted = true;
      }
    }

    if (!an_input_was_promoted) {
      // No inputs need promotion so just continue
      continue;
    }

    Expr* replay = nullptr;

    IdGroups promoted_input_groups;
    for (auto inp_id : promoted_inputs) {
      auto inp_disjoint_set_pair =
          intersection_exact_loop_graph.disjointIdSet(inp_id);
      if (inp_disjoint_set_pair.second) {
        promoted_input_groups.pushBack(inp_disjoint_set_pair.first);
      }
    }

    // Before replaying, check if there's already an expression like this, if so
    // use that for promotion. We would need the iel entries for non-promoted
    // inputs to match exactly to reuse the expression.
    //
    // Unfortunately this doesn't actually seem to save any replays because
    // we're not adding the replayed expression to the iel graph since we're
    // traversing the iel graph.
    //
    // TODO: Can we reduce the number of new expressions generated here?
    ExprGroups non_promoted_input_uses;
    for (auto iel_group : promoted_input_groups.intersect(input_groups)) {
      non_promoted_input_uses.pushBack(
          intersection_exact_loop_graph.uniqueUses(iel_group));
    }

    for (auto iel_use_group : non_promoted_input_uses) {
      if (transformAtributesMatch(iel_expr->front(), iel_use_group->front())) {
        auto use_inps =
            ir_utils::filterByType<IterDomain>(iel_use_group->front()->inputs())
                .vector();
        bool inps_match = true;
        for (auto inp_i : c10::irange(use_inps.size())) {
          inps_match = inps_match &&
              intersection_exact_loop_graph.disjointIdSets().strictAreMapped(
                  use_inps[inp_i], promoted_inputs[inp_i]);
        }
        if (inps_match) {
          replay = iel_use_group->front();
          break;
        }
      }
    }

    bool replayed = replay == nullptr;
    if (replay == nullptr) {
      replay = addReplayAs(promoted_inputs, iel_expr->front());
      // std::cout << "  ***REPLAY***:\n    " << iel_expr->front()
      //           << "    As:" << replay->toString();
    }

    auto out_groups = intersection_exact_loop_graph.outputGroups(iel_expr);

    // Mark outputs as having a promoted iter domain
    auto replay_out_ids =
        ir_utils::filterByType<IterDomain>(replay->outputs()).vector();
    auto ref_out_ids =
        ir_utils::filterByType<IterDomain>(iel_expr->front()->outputs())
            .vector();

    TORCH_INTERNAL_ASSERT(replay_out_ids.size() == out_groups.size());

    for (auto i : c10::irange(replay_out_ids.size())) {
      iel_promotion_map[out_groups[i]] = replay_out_ids[i];
      // Explicitly map loop map since expr propagation doesn't happen
      if (replayed) {
        idGraph(IdMappingMode::LOOP).mapIds(replay_out_ids[i], ref_out_ids[i]);
      }
    }
  }
  return iel_promotion_map;
}

namespace {

std::unordered_map<IdGroup, IterDomain*> updateMap(
    const std::unordered_map<IdGroup, IterDomain*> stale_map,
    IdGraph& new_graph) {
  std::unordered_map<IdGroup, IterDomain*> new_map;
  for (auto stale_entry : stale_map) {
    auto stale_id_group = stale_entry.first;
    auto new_groups = new_graph.toGroups(*stale_id_group);
    TORCH_INTERNAL_ASSERT(
        new_groups.size() == 1,
        "\nUpdate map assumes that new graph is equivalent to old graph plus extra mappings.\n",
        "i.e. all mappings in new_graph should exist in the graph stale_map was produced on.\n",
        "old:",
        debug::toString(stale_id_group),
        "new: ",
        debug::toString(new_groups));
    new_map[new_groups.front()] = stale_entry.second;
  }
  return new_map;
}

// Returns for each IdGroup in provided IdGraph what the input IdGroups are
// traversing on definitions. Ignoring broadcast IdGroups and resetting inputs
// at RFactor IdGroups.
std::unordered_map<IdGroup, IdGroups> computeCoveredGroups(
    const IdGraph& graph,
    std::unordered_set<IterDomain*> view_rfactor_ids) {
  // Map from an exact iter domain group, to all the exact iter domain groups it
  // covers
  std::unordered_map<IdGroup, IdGroups> covered_ids;

  for (auto id_group : graph.disjointIdSets().disjointSets()) {
    // Initialize inputs
    if (graph.uniqueDefinitions(id_group).empty()) {
      covered_ids[id_group] = {id_group};
    }

    // Initialize rfactor groups
    if (std::any_of(id_group->begin(), id_group->end(), [&](IterDomain* id) {
          return view_rfactor_ids.find(id) != view_rfactor_ids.end();
        })) {
      covered_ids[id_group] = {id_group};
    }

    // Initialize broadcast groups to empty
    if (std::any_of(id_group->begin(), id_group->end(), [&](IterDomain* id) {
          return id->isBroadcast();
        })) {
      covered_ids[id_group] = {};
    }
  }

  IdGraphStmtSort exact_stmt_sort(graph);

  for (auto exact_expr : exact_stmt_sort.exprs()) {
    auto input_groups = graph.inputGroups(exact_expr);

    IdGroups covered;
    for (auto inp_group : input_groups) {
      covered.pushBack(covered_ids.at(inp_group));
    }

    for (auto output_group : graph.outputGroups(exact_expr)) {
      covered_ids[output_group] = covered;
    }
  }

  return covered_ids;
}
}; // namespace

std::unordered_map<IdGroup, IterDomain*> IterDomainGraphs::
    buildLoopPromotionMap(
        const std::vector<Expr*>& exprs,
        StatefulLoweringInfo& info,
        std::unordered_map<IdGroup, IterDomain*> stale_promotion_map) {
  // Opportunistically add non-inlined loop relationships where they don't
  // interfere with the loop groups. This should be on all p_ids that are not
  // p_ca_ids.
  for (auto p_id : info.ordered_c_ids.subtract(info.ordered_p_ca_ids)) {
    auto entry_it = info.p2c_permissive_maps.find(p_id);
    if (entry_it == info.p2c_permissive_maps.end()) {
      continue;
    }
    auto c_ids = entry_it->second;
    for (auto c_id : c_ids) {
      if (idGraph(IdMappingMode::LOOP)
              .disjointIdSets()
              .permissiveAreMapped(p_id, c_id)) {
        // Already mapped
        continue;
      }

      // Grab all iter domains already in the loop groups for both iter
      // domains.
      auto loop_groups =
          idGraph(IdMappingMode::LOOP)
              .toGroups(VectorOfUniqueEntries<IterDomain*>{p_id, c_id});

      VectorOfUniqueEntries<IterDomain*> all_ids_in_groups;

      ParallelType common_ptype =
          loop_groups.front()->front()->getParallelType();
      if (std::any_of(
              loop_groups.begin() + 1,
              loop_groups.end(),
              [common_ptype](IdGroup id_group) {
                return id_group->front()->getParallelType() != common_ptype;
              })) {
        // Parallel types don't match, cannot merge non-inlined loop groups.
        continue;
      }

      for (auto loop_group : loop_groups) {
        all_ids_in_groups.pushBack(*loop_group);
      }

      // Ignore new loop mappings from replays, we can still opportunistically
      // merge leaves if they already have a promoted id from replay associated
      // with them.
      all_ids_in_groups = all_ids_in_groups.intersect(info.ordered_c_ids);

      // Grab the almost exact map of all iter domains in those loop groups
      auto ae_groups =
          idGraph(IdMappingMode::ALMOSTEXACT).toGroups(all_ids_in_groups);

      // If there's no broadcast promotion within the loop group then all the
      // iter domains will be almost exact mapped with eachother.
      if (ae_groups.size() == 1) {
        idGraph(IdMappingMode::LOOP).mapIds(p_id, c_id);
      }
    }
  }

  // Need to use the intersection of exact and loop map again, it needs to be
  // recomputed.
  auto intersection_exact_loop_graph = buildIntersection(
      idGraph(IdMappingMode::EXACT), idGraph(IdMappingMode::LOOP), false);

  // Update the promotion map
  auto iel_promotion_map =
      updateMap(stale_promotion_map, intersection_exact_loop_graph);

  // Map from an exact iter domain group, to all the exact iter domain groups it
  // covers; needs to be recomputed.
  std::unordered_map<IdGroup, IdGroups> exact_covered_ids =
      computeCoveredGroups(idGraph(IdMappingMode::EXACT), view_rfactor_ids_);

  // Grab terminal iter domain in the loop groups.
  VectorOfUniqueEntries<IterDomain*> terminal_loop_ids =
      computeTerminalLoopIds(info);

  // Loop promotion map is to prepare for IterDomain replays to resolve
  // non-inlined loop groups. Since these replays will modify the loop map as
  // we're iterating over the loop map, operate on a copy of the loop map, not
  // the original one.
  auto loop_graph_copy = idGraph(IdMappingMode::LOOP);

  // Build a map from loop iter domain group to a promoted iter domain (doesn't
  // have to be in the loop group) that covers all the exact groups
  // representative of the resolved transformations within the loop group. Only
  // the inlined loop groups will be covered here.
  std::unordered_map<IdGroup, IterDomain*> loop_graph_copy_promotion_map;

  // TODO: I'm uncertain if we can simply use the iel_promotion_map. Once this
  // system is in use we should test not recomputing the "concrete ids".

  for (auto loop_group : loop_graph_copy.disjointIdSets().disjointSets()) {
    if (loop_group->size() == 1) {
      loop_graph_copy_promotion_map[loop_group] = loop_group->front();
      continue;
    }

    // Grab all the (potentially promoted) terminal iter domains in this group.
    // Save the exact group and the iter domain in this vector.
    std::vector<std::pair<IdGroup, IterDomain*>> exact_promoted_terminal_ids;
    for (auto loop_id : *loop_group) {
      // If not a terminal id in the group skip
      if (!terminal_loop_ids.has(loop_id)) {
        continue;
      }

      // Grab the iel entry
      auto iel_set_pair = intersection_exact_loop_graph.disjointIdSet(loop_id);
      TORCH_INTERNAL_ASSERT(iel_set_pair.second);
      auto iel_group = iel_set_pair.first;

      auto iel_promo_it = iel_promotion_map.find(iel_group);
      if (iel_promo_it == iel_promotion_map.end()) {
        // If this terminal ID has a promotion, grab the promoted ID.
        auto promo_id_exact_it =
            idGraph(IdMappingMode::EXACT).disjointIdSet(loop_id);
        TORCH_INTERNAL_ASSERT(promo_id_exact_it.second);
        exact_promoted_terminal_ids.push_back(
            std::make_pair(promo_id_exact_it.first, loop_id));
      } else {
        // If this terminal ID doesn't have a promotion associated with it, save
        // the terminal ID.
        auto promo_id_exact_it =
            idGraph(IdMappingMode::EXACT).disjointIdSet(iel_promo_it->second);
        TORCH_INTERNAL_ASSERT(promo_id_exact_it.second);
        exact_promoted_terminal_ids.push_back(
            std::make_pair(promo_id_exact_it.first, iel_promo_it->second));
      }
    }

    // All the exact groups of the iter domains in the loop group
    IdGroups exact_groups = idGraph(IdMappingMode::EXACT).toGroups(*loop_group);

    // All exact groups covered by all iter domains in this loop group
    IdGroups loop_group_covered_ids;
    for (auto exact_group : exact_groups) {
      auto covered_it = exact_covered_ids.find(exact_group);
      TORCH_INTERNAL_ASSERT(covered_it != exact_covered_ids.end());
      loop_group_covered_ids.pushBack(covered_it->second);
    }

    IterDomain* loop_promotion_id = nullptr;

    // Check if any of the candidate Iter Domains we collected cover all the
    // exact groups of loop_group_covered_ids. If so, that's the correct
    // promoted iter domain of this group.
    for (auto entry : exact_promoted_terminal_ids) {
      auto terminal_id_group = entry.first;
      auto terminal_id = entry.second;
      auto covered_it = exact_covered_ids.find(terminal_id_group);
      TORCH_INTERNAL_ASSERT(covered_it != exact_covered_ids.end());
      if (loop_group_covered_ids.subtract(covered_it->second).size() == 0) {
        loop_promotion_id = terminal_id;
        break;
      }
    }

    if (loop_promotion_id == nullptr) {
      std::stringstream err_msg;
      err_msg
          << "\n ERROR Loop promotion map build. Could not find promotion for loop group:\n  ";
      err_msg << debug::toString(loop_group, 0, true);
      err_msg << "\nnone of the terminal iter domains of this group:\n  ";
      for (auto entry : exact_promoted_terminal_ids) {
        auto terminal_id_group = entry.first;
        auto covered_id_groups = exact_covered_ids.at(terminal_id_group);
        err_msg << "  " << debug::toString(terminal_id_group, 0, true)
                << " -(covers)-> " << debug::toString(covered_id_groups)
                << std::endl;
      }
      err_msg << "iter domains in this group cover all id groups:\n";
      for (auto covered_group : loop_group_covered_ids) {
        err_msg << "  " << debug::toString(covered_group, 0, true);
      }
      // TORCH_INTERNAL_ASSERT(false, err_msg.str());
    } else {
      loop_graph_copy_promotion_map[loop_group] = loop_promotion_id;
    }
  }

  // std::cout << "Loop promotion before second replay:" << std::endl;
  for (auto loop_group : loop_graph_copy.disjointIdSets().disjointSets()) {
    if (loop_graph_copy_promotion_map.find(loop_group) !=
        loop_graph_copy_promotion_map.end()) {
      // std::cout << debug::toString(loop_group, 0, true) << " -> "
      //           << loop_graph_copy_promotion_map[loop_group]->toString()
      //           << std::endl;
    }
  }

  // Reset the promotion map for the second pass.
  // TODO: Unclear if we could simply update the iel_promotion_map from
  // buildInlinePromotions, instead of manually building it.
  iel_promotion_map.clear();

  // Need to run a replay for the loop groups that are dependent on inlined loop
  // groups, but themselves are not inlined loop groups.

  for (auto iel_expr : IdGraphStmtSort(intersection_exact_loop_graph).exprs()) {
    auto iel_inp_groups = intersection_exact_loop_graph.inputGroups(iel_expr);

    auto iel_out_groups = intersection_exact_loop_graph.outputGroups(iel_expr);

    // When replaying the transformations we can't blindly apply loop promotion
    // to all iter domains within a loop group as it would replay the
    // transformations within that loop group on the promoted id of that loop
    // group.
    //
    // i.e. if we have the inlined domains from:
    // T2[i0*i1] pa(1) = T0[i0*b1]ca(1) + T1[i0*i1]ca(1)
    // The inlined loop group would be:
    //
    // i0, i1, b1, i0*i1, b0*i1
    // Then if we replayed the iel transformations they would be:
    // merge(i0, i1)
    // merge(i0, b1)
    //
    // So if we replayed them with loop promotion, then i0, i1, b1 would be
    // promoted to i0*i1, and the merges would be replayed.
    //
    // Therefore only promote i0*b1 to i0*i1, or i0*i1 to i0*i1 (i.e. don't
    // promote an input to any transformation within the loop group).
    //
    // So if we have an iel_expr make sure it's inputs and outputs are not in
    // the same loop group.

    IdGroups inp_loop_groups;
    for (auto iel_inp_group : iel_inp_groups) {
      inp_loop_groups.pushBack(loop_graph_copy.toGroup(iel_inp_group->front()));
    }

    IdGroups out_loop_groups;
    for (auto iel_out_group : iel_out_groups) {
      out_loop_groups.pushBack(loop_graph_copy.toGroup(iel_out_group->front()));
    }

    // The inputs should be promoted based on the loop promotion map.
    bool loop_promote_inputs =
        !inp_loop_groups.subtract(out_loop_groups).empty();

    std::vector<IterDomain*> promoted_inputs;

    bool an_input_was_promoted = false;

    // Promote inputs for replay
    for (auto iel_inp_group : iel_inp_groups) {
      // Promote loops based on the loop promotion map. If the loop promotion
      // map should be used and has an entry we should use that promotion. This
      // happen when an iel expression is across a loop group boundary.
      // Signifying and capturing instances when we traverse across an inlined
      // loop group to a non-inlined loop group boundary (think of the iel graph
      // projected onto the loop graph).
      auto loop_copy_group = loop_graph_copy.toGroup(iel_inp_group->front());
      auto inp_loop_promo_it =
          loop_graph_copy_promotion_map.find(loop_copy_group);
      if (loop_promote_inputs &&
          inp_loop_promo_it != loop_graph_copy_promotion_map.end()) {
        promoted_inputs.push_back(inp_loop_promo_it->second);
        an_input_was_promoted = true;
      } else {
        // We still could require an input promotion. We could be traversing
        // across non-inlined groups. Meaning we have inputs that were promoted
        // in an inlined loop group traversing through the non-inlined portions
        // of the iel graph.
        auto inp_promo_it = iel_promotion_map.find(iel_inp_group);
        if (inp_promo_it == iel_promotion_map.end()) {
          promoted_inputs.push_back(iel_inp_group->front());
        } else {
          promoted_inputs.push_back(inp_promo_it->second);
          an_input_was_promoted = true;
        }
      }
    }

    if (!an_input_was_promoted) {
      continue;
    }

    Expr* replay = nullptr;

    // Before replaying, check if there's already an expression like this, if so
    // use that for promotion. We're still only looking for representative iter
    // domains, so if there's already an expression that would produce something
    // representative (matching in the exact graph) of what the new inputs would
    // generate, just promote to that expressions outputs, don't bother
    // generating a new one.
    //
    // Check all uses of the exact map the inputs are in, and look for one that
    // would match. Grab all uses of the promoted inputs' groups in the exact
    // map.
    std::vector<IdGroup> promoted_input_groups;

    ExprGroups promoted_input_uses;
    for (auto inp_id : promoted_inputs) {
      auto inp_exact_group = idGraph(IdMappingMode::EXACT).toGroup(inp_id);
      promoted_input_groups.push_back(inp_exact_group);
      promoted_input_uses.pushBack(
          idGraph(IdMappingMode::EXACT).uniqueUses(inp_exact_group));
    }

    // Check every use to see if it matches
    for (auto exact_use_group : promoted_input_uses) {
      // Check if all the attributes (including type) of the transform match
      if (!transformAtributesMatch(
              iel_expr->front(), exact_use_group->front())) {
        continue;
      }
      // Check if inputs all match
      if (promoted_input_groups !=
          idGraph(IdMappingMode::EXACT).inputGroups(exact_use_group)) {
        continue;
      }
      replay = exact_use_group->front();
      break;
    }

    bool replayed = replay == nullptr;
    if (replay == nullptr) {
      replay = addReplayAs(promoted_inputs, iel_expr->front());
    }
    //   std::cout << "  ***REPLAY2***:\n    " << iel_expr->front()
    //             << "    As:" << replay->toString();
    // } else {
    //   std::cout << "  ***MATCH2***:\n    " << iel_expr->front()
    //             << "    As:" << replay->toString();
    // }

    auto output_groups = intersection_exact_loop_graph.outputGroups(iel_expr);

    // Match or replay, mark promotion for output groups.
    auto replay_out_ids =
        ir_utils::filterByType<IterDomain>(replay->outputs()).vector();
    auto ref_out_ids =
        ir_utils::filterByType<IterDomain>(iel_expr->front()->outputs())
            .vector();

    TORCH_INTERNAL_ASSERT(replay_out_ids.size() == output_groups.size());

    for (auto i : c10::irange(replay_out_ids.size())) {
      if (!idGraph(IdMappingMode::EXACT)
               .disjointIdSets()
               .strictAreMapped(replay_out_ids[i], output_groups[i]->front())) {
        // Promote if necessary, if the output is already in the same exact map
        // it doesn't need a promotion.
        iel_promotion_map[output_groups[i]] = replay_out_ids[i];
        // Explicitly map loop map since expr propagation doesn't happen on the
        // loop map and the replayed outputs are brand new so we can map them
        // without joining disjoint loop groups (other than the new loop groups
        // the outputs of the replay are in)
        if (replayed) {
          // If we built new iter domains because we generated a new expression,
          // link the outputs in the loop graph.
          idGraph(IdMappingMode::LOOP)
              .mapIds(replay_out_ids[i], ref_out_ids[i]);
        }
      }
    }
  }

  // std::cout << "Promotion map from second replay: " << std::endl;
  for (auto group :
       intersection_exact_loop_graph.disjointIdSets().disjointSets()) {
    if (iel_promotion_map.find(group) == iel_promotion_map.end()) {
      continue;
    }
    // std::cout << debug::toString(group, 0, true) << " -> "
    //           << iel_promotion_map.at(group)->toString() << std::endl;
  }

  return iel_promotion_map;
}

std::unordered_map<IterDomain*, IterDomain*> IterDomainGraphs::buildIndexGraph(
    const std::vector<Expr*>& exprs,
    const std::vector<TensorView*>& all_tvs,
    StatefulLoweringInfo& info,
    std::unordered_map<IdGroup, IterDomain*> stale_promotion_map) {
  // Update the iel graph
  auto intersection_exact_loop_graph = buildIntersection(
      idGraph(IdMappingMode::EXACT), idGraph(IdMappingMode::LOOP), false);

  // Update the promotion map
  auto iel_promotion_map =
      updateMap(stale_promotion_map, intersection_exact_loop_graph);

  auto exact_covered_ids =
      computeCoveredGroups(idGraph(IdMappingMode::EXACT), view_rfactor_ids_);

  // Grab terminal iter domain in the loop groups.
  VectorOfUniqueEntries<IterDomain*> terminal_loop_ids =
      computeTerminalLoopIds(info);

  // Loop promotion map is to prepare for IterDomain replays. Since these
  // replays will modify the loop map, we operate on a copy of the loop map,
  // not the original one.
  // Loop promotion map is to prepare for IterDomain replays to resolve
  // non-inlined loop groups. Since these replays will modify the loop map as
  // we're iterating over the loop map, operate on a copy of the loop map, not
  // the original one.
  auto loop_graph_copy = idGraph(IdMappingMode::LOOP);

  // Build a map from loop iter domain group to a promoted iter domain (doesn't
  // have to be in the loop group) that covers all the exact groups
  // representative of the resolved transformations within the loop group. Only
  // the inlined loop groups will be covered here.
  std::unordered_map<IdGroup, IterDomain*> loop_graph_copy_promotion_map;

  // Returns a new promoted domain if one is found in the iel_promotion_map,
  // otherwise returns original id.
  auto get_promoted_id = [&intersection_exact_loop_graph,
                          &iel_promotion_map](IterDomain* id) {
    auto iel_group = intersection_exact_loop_graph.toGroup(id);
    auto iel_promotion_map_it = iel_promotion_map.find(iel_group);
    if (iel_promotion_map_it != iel_promotion_map.end()) {
      return iel_promotion_map_it->second;
    }
    return id;
  };

  // Returns the entry in exact_covered_ids associated with provided IterDomain.
  // Basically calling .at but with a better error.
  auto get_covered_exact_groups = [&](IterDomain* id) {
    auto exact_group = idGraph(IdMappingMode::EXACT).toGroup(id);
    auto covered_it = exact_covered_ids.find(exact_group);
    TORCH_INTERNAL_ASSERT(
        covered_it != exact_covered_ids.end(),
        "Missing map entry in analysis for: ",
        debug::toString(exact_group, 0, true));
    return covered_it->second;
  };

  // Now we need to find the right promoted ID for every loop group, making
  // sure the promoted ID covers every ID of the IDs in the loop group.
  // This ID could be a terminal ID in the group. A promoted ID of the terminal
  // IDs, or an ID that was replayed previously and now part of the loop group.
  //
  // The correct/final promoted ID of the loop group must exist at this point.
  // It just might not be within the loop group we're looking at.
  // std::cout << "Find promoted ids from loop group or promoted iter domains."
  //           << std::endl;
  for (auto loop_group : loop_graph_copy.disjointIdSets().disjointSets()) {
    if (loop_group->size() == 1) {
      auto promoted_id = get_promoted_id(loop_group->front());

      TORCH_INTERNAL_ASSERT(
          get_covered_exact_groups(loop_group->front())
                  .subtract(get_covered_exact_groups(promoted_id))
                  .size() == 0,
          "Promotion failed, promoted id: ",
          promoted_id->toString(),
          " doesn't cover the right domains for ",
          loop_group->front()->toString());
      loop_graph_copy_promotion_map[loop_group] = promoted_id;
      continue;
    }

    // If promotion entry exists for any terminal id the promoted id will be
    // stored here.
    std::vector<IterDomain*> promoted_terminal_ids;

    // If a promotion entry doesn't exist for a terminal id, put it here.
    std::vector<IterDomain*> terminal_ids;

    // All exact groups that the terminal loop id's cover.
    IdGroups all_covered_exact_groups;

    // Populate all three structures above.
    for (auto loop_id : *loop_group) {
      if (!terminal_loop_ids.has(loop_id)) {
        continue;
      }

      all_covered_exact_groups.pushBack(get_covered_exact_groups(loop_id));

      auto promoted_id = get_promoted_id(loop_id);
      if (promoted_id == loop_id) {
        terminal_ids.push_back(loop_id);
      } else {
        promoted_terminal_ids.push_back(promoted_id);
      }
    }

    // If promoted id's exist, those are the candidates to have the right
    // transformations for indexing. Otherwise, use the terminal _ids.
    auto candidate_ids =
        promoted_terminal_ids.empty() ? terminal_ids : promoted_terminal_ids;

    // Find the loop promotion id from the candidates.
    IterDomain* loop_promotion_id = nullptr;
    for (auto candidate_id : candidate_ids) {
      if (all_covered_exact_groups
              .subtract(get_covered_exact_groups(candidate_id))
              .empty()) {
        loop_promotion_id = candidate_id;
        break;
      }
    }

    // If we're still missing the loop_promotion_id, check all replayed IDs in
    // the loop group.
    if (loop_promotion_id == nullptr) {
      candidate_ids = loop_group->subtract(info.ordered_c_ids).vector();
      for (auto candidate_id : candidate_ids) {
        if (all_covered_exact_groups
                .subtract(get_covered_exact_groups(candidate_id))
                .empty()) {
          loop_promotion_id = candidate_id;
        }
      }
    }

    if (loop_promotion_id == nullptr) {
      std::stringstream err_msg;
      err_msg << "\nCould not find promotion for loop group:\n  ";
      err_msg << debug::toString(loop_group, 0, true);
      err_msg << "\nnone of the candidate iter domains of this group:\n  ";
      err_msg << "  "
              << VectorOfUniqueEntries<IterDomain*>(candidate_ids).toString();
      err_msg << "\n cover all id groups that the loop group covers:\n";
      err_msg << " " << debug::toString(all_covered_exact_groups) << std::endl;
      TORCH_INTERNAL_ASSERT(false, err_msg.str());
    }

    loop_graph_copy_promotion_map[loop_group] = loop_promotion_id;
  }

  // std::cout << "Promotion map to build the Index Graph: " << std::endl;
  for (auto group : loop_graph_copy.disjointIdSets().disjointSets()) {
    if (loop_graph_copy_promotion_map.find(group) ==
        loop_graph_copy_promotion_map.end()) {
      continue;
    }
    // std::cout << debug::toString(group, 0, true) << " -> "
    //           << loop_graph_copy_promotion_map.at(group)->toString()
    //           << std::endl;
  }

  // Indexing traversal must start at leaf nodes of TensorViews as that's where
  // the loop indices are defined. For indexing we need to propagate leaves to
  // root domains. We want the indexing graph easy to traverse. Easy to traverse
  // means that we start at terminating outputs of this graph and propagate to
  // terminating inputs. We shouldn't have to worry about which paths each time
  // we traverse the index graph as we may do it many times.

  // The IEL Map cannot be traversed for indexing, because the loop map is
  // really only used to model broadcast promotion. We could have multiple paths
  // from leaf nodes to an intermediate IEL entry. Meaning:

  // T0 root[i0, i1] T0 leaf domain [i0*i1//32, 4, 8]
  // T1 root[i0, i1] T0 leaf domain [i0*i1//32, 8, 4]

  // Even though T0 and T1 are inlined on the outer most dimension, indexing
  // into their roots is different. Yet, their roots would be in the same IEL
  // entries.

  // The index graph should provide a direct model of what indices are reused,
  // i.e. if two ID's in the IndexMap map to eachother, they should use the same
  // index math. Therefore, roughly what we need to do is:

  // - Figure out which leaves share exact indexing and map them together:
  //   (1) Producer-consumer leaf nodes are inlined with eachother (map to the
  //   same promoted id)
  //   (2) Promoted producer-consumer leaf nodes are almost exact, have the same
  //   parallel type, but are not inlined.

  // - Start at the promoted leaf nodes of each tensor view

  // - If those promoted leaf nodes are *ALMOST EXACT* mapped from
  // producer-consumer they can be mapped in the index map

  // - Traversing backward from each tensor view's leaf nodes, we directly reach
  // the root nodes of that tensor view

  // - During the backward traversal, for an expression, if the output iter
  // domains are mapped in the index map, their inputs should be mapped as well.
  //   So as we build the index map, we could also be accumulating mapped iter
  //   domains.

  // Mark all iter domains that share a loop nest and are almost exact mapped.
  // Ignores promotion.

  // Doing the same as above on promoted iter domains is a bit tricky, because
  // there's a promoted IterDomian per IEL group, we need a promoted IterDomain
  // per index group. So let's figure out which leaf domains share a promoted
  // iter domain, so we don't have to build a promoted iter domain for every
  // leaf, then try to rejoin them.

  // TODO: I think we need to validate that for each tensor view leaf domains,
  // no two leaves within a tensor domain map to another leaf in the same tensor
  // domain in the IEL graph. Not sure how this could occur, but I suspect it
  // could.

  // Which non-promoted iter domains, share their promoted iterdomains
  DisjointSets<IterDomain*> shared_promoted_id;

  for (auto expr : exprs) {
    std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
        promo_id_to_producer_ids;
    std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
        promo_id_to_consumer_ids;

    // Copy of all promo ids for determinism
    VectorOfUniqueEntries<IterDomain*> all_promo_ids;

    for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
      for (auto p_id : producer->domain()->leaf()) {
        // Initialize all entries
        shared_promoted_id.initializeSet(p_id);

        auto loop_copy_p_group_pair = loop_graph_copy.disjointIdSet(p_id);
        TORCH_INTERNAL_ASSERT(loop_copy_p_group_pair.second);
        auto loop_copy_p_group = loop_copy_p_group_pair.first;

        auto promo_id_it =
            loop_graph_copy_promotion_map.find(loop_copy_p_group);
        TORCH_INTERNAL_ASSERT(
            promo_id_it != loop_graph_copy_promotion_map.end());

        promo_id_to_producer_ids[promo_id_it->second].pushBack(p_id);
        all_promo_ids.pushBack(promo_id_it->second);
      }
    }

    for (auto consumer : ir_utils::filterByType<TensorView>(expr->outputs())) {
      for (auto c_id : consumer->domain()->leaf()) {
        // Initialize all entries
        shared_promoted_id.initializeSet(c_id);

        auto loop_copy_c_group_pair = loop_graph_copy.disjointIdSet(c_id);
        TORCH_INTERNAL_ASSERT(loop_copy_c_group_pair.second);
        auto loop_copy_c_group = loop_copy_c_group_pair.first;

        auto promo_id_it =
            loop_graph_copy_promotion_map.find(loop_copy_c_group);
        TORCH_INTERNAL_ASSERT(
            promo_id_it != loop_graph_copy_promotion_map.end());

        promo_id_to_consumer_ids[promo_id_it->second].pushBack(c_id);
        all_promo_ids.pushBack(promo_id_it->second);
      }
    }

    for (auto promo_id : all_promo_ids) {
      auto p_ids_it = promo_id_to_producer_ids.find(promo_id);
      if (p_ids_it == promo_id_to_producer_ids.end()) {
        continue;
      }
      auto p_ids = p_ids_it->second;

      auto c_ids_it = promo_id_to_consumer_ids.find(promo_id);
      if (c_ids_it == promo_id_to_consumer_ids.end()) {
        continue;
      }
      auto c_ids = c_ids_it->second;

      if (c_ids.size() && p_ids.size()) {
        for (auto p_id : p_ids) {
          shared_promoted_id.mapEntries(p_ids.front(), p_id);
        }
        for (auto c_id : c_ids) {
          shared_promoted_id.mapEntries(p_ids.front(), c_id);
        }
      }
    }
  }

  auto get_representative_promoted_id = [&](IterDomain* id) {
    auto promo_id_it =
        loop_graph_copy_promotion_map.find(loop_graph_copy.toGroup(id));
    TORCH_INTERNAL_ASSERT(promo_id_it != loop_graph_copy_promotion_map.end());
    return promo_id_it->second;
  };

  // std::cout << "Opportunistic joining of shared promos:" << std::endl;
  // Opportunistically collapse indexing of non-inlined leaf domains if their
  // promoted ids are almost exact mapped and have the same parallel type.
  for (auto expr : exprs) {
    for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
      // std::cout << "  Producer: " << producer->toString() << std::endl;
      auto producer_root = producer->getMaybeRFactorDomain();

      auto non_inline_producer_domain = producer->domain()->leaf();
      non_inline_producer_domain.erase(
          non_inline_producer_domain.begin(),
          non_inline_producer_domain.begin() +
              producer->getComputeAtPosition());

      for (auto consumer :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        // std::cout << "    Consumer: " << consumer->toString() << std::endl;
        auto consumer_domain = consumer->domain()->leaf();

        auto p2c_permissive_map =
            idGraph(IdMappingMode::PERMISSIVE)
                .buildMapBetween(non_inline_producer_domain, consumer_domain);

        for (auto p_id : non_inline_producer_domain) {
          auto p2c_it = p2c_permissive_map.find(p_id);
          if (p2c_it == p2c_permissive_map.end() || p2c_it->second.empty()) {
            continue;
          }

          auto rep_p_id = get_representative_promoted_id(p_id);
          auto c_id = p2c_it->second.front();
          auto rep_c_id = get_representative_promoted_id(c_id);

          // std::cout << "      " << p_id->toString() << " -> "
          //           << rep_p_id->toString() << " :: " << c_id->toString()
          //           << " -> " << rep_c_id->toString() << std::endl;
          if (!idGraph(IdMappingMode::ALMOSTEXACT)
                   .disjointIdSets()
                   .strictAreMapped(rep_p_id, rep_c_id)) {
            continue;
          }
          if (rep_p_id->getParallelType() != rep_c_id->getParallelType()) {
            continue;
          }
          // std::cout << "      Mapped" << std::endl;
          shared_promoted_id.mapEntries(p_id, c_id);
        }
      }
    }
  }

  // std::cout << "Leaf iter domains that share a promoted iter domain."
  //           << std::endl;
  // for (auto disjoint_set : shared_promoted_id.disjointSets()) {
  //   std::cout << disjoint_set->toString() << std::endl;
  // }

  // Map from leaf iter domains to their potentially promoted iter domain used
  // for indexing.
  std::unordered_map<IterDomain*, IterDomain*> leaf_promotion_map;

  // If a promoted iter domain was generated by replays, it won't be connected
  // in the index graph. We can reuse these iter domains directly instead of
  // having to make a clone of them. However, we can only use them once for a
  // group.
  VectorOfUniqueEntries<IterDomain*> used_promo_ids;

  for (auto id_group : shared_promoted_id.disjointSets()) {
    IterDomain* promo_id = get_representative_promoted_id(id_group->front());

    // Promoted id is already part of the group, just use that.
    if (std::find(id_group->begin(), id_group->end(), promo_id) !=
        id_group->end()) {
      for (auto id : *id_group) {
        leaf_promotion_map[id] = promo_id;
      }
      continue;
    }

    // Promo id generated from running replay, we can use it for one of the
    // index groups.
    if (!info.ordered_c_ids.has(promo_id) && !used_promo_ids.has(promo_id)) {
      used_promo_ids.pushBack(promo_id);
      for (auto id : *id_group) {
        leaf_promotion_map[id] = promo_id;
      }
      continue;
    }

    // Need to take a copy of the promo_id as it's already dedicated to an index
    // group.
    promo_id = cloneIterDomain(promo_id);
    for (auto id : *id_group) {
      leaf_promotion_map[id] = promo_id;
    }
  }

  // TODO: This needs to be available as a member function
  auto get_promoted_domain = [&](TensorDomain* td) {
    std::vector<IterDomain*> promoted_leaves;
    for (auto id : td->leaf()) {
      auto promo_it = leaf_promotion_map.find(id);
      TORCH_INTERNAL_ASSERT(promo_it != leaf_promotion_map.end());
      promoted_leaves.push_back(promo_it->second);
    }
    return promoted_leaves;
  };

  // std::cout << "Iter domain group to their promoted iter domain." <<
  // std::endl; for (auto id_group : shared_promoted_id.disjointSets()) {
  //   std::cout << id_group->toString() << "\n  -> "
  //             << leaf_promotion_map.at(id_group->front()) << std::endl;
  // }

  // Track every expression required for indexing
  VectorOfUniqueEntries<Expr*> all_index_exprs;
  // Track every iter domain required for indexing
  VectorOfUniqueEntries<IterDomain*> all_index_ids;

  // std::cout << "\n\nThird and final replay" << std::endl;
  // std::cout << "Building promoted tensor view domains:" << std::endl;
  // Need to "replay" all of the indexing expressions to make sure roots are
  // connected to the promoted leaves, in a way we can index directly on the
  // index graph.
  //
  // Since we're performing replays we need to copy the graph we're iterating
  // on.
  auto ae_graph = idGraph(IdMappingMode::ALMOSTEXACT);

  // Because of how replays work in buildInlinePromotions and
  // buildLoopPromotionMap, we could have multiple uses and definitions of the
  // the same iter domain.
  //
  // However, for the index graph we want to go back to every iter domain having
  // at most one use and definition.
  //
  // We also want to use expressions that exist if we can.
  //
  // If there's multiple paths on the index graph then we would generate
  // conflicting indicies (unless somehow the expressions all end up collapsing
  // by being mapped later). Enforce one defintion and use per iter domain.
  std::unordered_map<IterDomain*, Expr*> id_to_index_use;
  std::unordered_map<IterDomain*, Expr*> id_to_index_def;

  // Initialize index graph using the history of each tensorview. These
  // expressions are not guaranteed to be used, but if it is used, this will
  // prefer those used in a tv's history.
  //
  // This prevents conflicts later where we try to reuse an expression and take
  // an expression in another tensor view's history.
  for (auto tv : all_tvs) {
    auto transforms = StmtSort::getExprsBetween(
        FusionGuard::getCurFusion(),
        {tv->getRootDomain().begin(), tv->getRootDomain().end()},
        {tv->domain()->leaf().begin(), tv->domain()->leaf().end()});
    for (auto transform : transforms) {
      for (auto inp : ir_utils::filterByType<IterDomain>(transform->inputs())) {
        id_to_index_use[inp] = transform;
      }
      for (auto out :
           ir_utils::filterByType<IterDomain>(transform->outputs())) {
        id_to_index_def[out] = transform;
      }
    }
  }

  // Manually initialize the index graph
  for (auto id_group :
       idGraph(IdMappingMode::ALMOSTEXACT).disjointIdSets().disjointSets()) {
    for (auto id : *id_group) {
      VectorOfUniqueEntries<Expr*> defs;
      if (id_to_index_def.find(id) != id_to_index_def.end()) {
        defs.pushBack(id_to_index_def.at(id));
      }

      VectorOfUniqueEntries<Expr*> uses;
      if (id_to_index_use.find(id) != id_to_index_use.end()) {
        uses.pushBack(id_to_index_use.at(id));
      }

      idGraph(IdMappingMode::INDEX).initializeId(id, defs, uses);
    }
  }

  idGraph(IdMappingMode::INDEX).mapThroughTrivialExprs();
  idGraph(IdMappingMode::INDEX).removeTrivialExprs();

  for (auto tv : all_tvs) {
    // We don't have to process inputs at this point as they're already
    // allocated on a global
    if (tv->isFusionInput()) {
      continue;
    }

    auto promoted_domain = get_promoted_domain(tv->domain());
    // replay from root to promoted leaves.
    // std::cout << "\n\n  Processing: TV" << tv->name() << "\n    Root: TV"
    //           << tv->getRootDomain()
    //           << "\n    Domain promoted to: " << promoted_domain <<
    //           std::endl;

    // The promoted leaf iter domains are where indexing starts. We're going to
    // start at those expressions and replay transformations for this tensor
    // view working back to root domains. We want to intercept the history of
    // the transformations local to the tensor view where possible.
    //
    // So effectively what we have to do is map the ae graph to the history of
    // the tensor view as well as the promoted iter domains. We start traversal
    // at the promoted iter domains and will intercept the tensor view history
    // as possible.
    //
    // We must be able to interecept the provided tensor view at the rfactor and
    // root domains, otherwise we wouldn't be able to allocate or index into the
    // buffer at tensor view (rfactor domain) or it's producer (root domain).

    // Grab all the domains and convert them to their ae groups.
    auto all_ids_v = ir_utils::allIDsOf(tv);
    auto all_ids =
        VectorOfUniqueEntries<IterDomain*>(all_ids_v.begin(), all_ids_v.end());

    // Create a map from the ae group to the iter domain as when we replay we'll
    // replace the ae iter domain in the replay with the id in this map.
    std::unordered_map<IdGroup, IterDomain*> ae_group_2_id;

    for (auto tv_id : all_ids) {
      // Use emplace here as it multiple tv_ids could map to the same ae_group.
      // Emplace will simply grab the first one that appears.
      ae_group_2_id.emplace(std::make_pair(ae_graph.toGroup(tv_id), tv_id));
    }

    // Add the promoted domain ids
    for (auto promoted_id : promoted_domain) {
      all_ids.pushBack(promoted_id);
      ae_group_2_id[ae_graph.toGroup(promoted_id)] = promoted_id;
    }

    auto ae_leaf_groups = ae_graph.toGroups(VectorOfUniqueEntries<IterDomain*>{
        promoted_domain.begin(), promoted_domain.end()});

    // Don't support multiple leaf domains promoted to the same ae graph at this
    // point.
    TORCH_INTERNAL_ASSERT(
        ae_leaf_groups.size() == promoted_domain.size(),
        "Multiple leaf domains that map almost exactly is not supported at this point.");

    auto ae_root_groups = ae_graph.toGroups(VectorOfUniqueEntries<IterDomain*>{
        tv->getRootDomain().begin(), tv->getRootDomain().end()});

    // Make a copy of the expressions so we can reverse them
    auto reverse_indexing_transforms =
        ae_graph.getExprsBetween(ae_root_groups, ae_leaf_groups).vector();

    std::reverse(
        reverse_indexing_transforms.begin(), reverse_indexing_transforms.end());

    // Replay indexing transformations start on leaf nodes propagating back to
    // the root domain
    for (ExprGroup ae_expr_group : reverse_indexing_transforms) {
      // Outputs must be promoted with the ae_group_2_id map. Inputs may be
      // promoted when we intercept the history of the TV with the replay.
      //
      // if there isn't an entry in ae_group_2_id, then we have a resolved
      // merged in broadcast, and that resolved iter domain will need to be
      // cloned. Would be nice to see if the dangling input has already been
      // added already through another indexing path that this overlaps with,
      // however having an additional ID and expression per case doesn't seem
      // too bad right now.

      auto ae_output_groups = ae_graph.outputGroups(ae_expr_group);

      std::vector<IterDomain*> promoted_outputs;
      for (auto out_group : ae_output_groups) {
        auto out_promo_it = ae_group_2_id.find(out_group);
        if (out_promo_it == ae_group_2_id.end()) {
          promoted_outputs.push_back(out_group->front());
        } else {
          promoted_outputs.push_back(out_promo_it->second);
        }
      }

      Expr* replay = nullptr;

      // Check if we already have this expression covered in the index graph. If
      // so, don't add another expr, just add mappings for the iter domains
      // necessary.

      // If there isn't already an index expression covering this, check the
      // almost exact map if there's any expression not already in the index
      // graph that we can use, and add in the index graph.

      // Else generate a new index expression from scratch.

      // Before replaying, check if there's already an expression like this, if
      // so use that for promotion.
      ExprGroups promoted_output_defs;
      for (auto out_id : promoted_outputs) {
        auto index_group = idGraph(IdMappingMode::INDEX).toGroup(out_id);
        promoted_output_defs.pushBack(
            idGraph(IdMappingMode::INDEX).uniqueDefinitions(index_group));
      }

      for (auto index_def_group : promoted_output_defs) {
        // This enforces that inputs and outputs are all almost exact mapped
        if (!idGraph(IdMappingMode::ALMOSTEXACT)
                 .disjointExprSets()
                 .strictAreMapped(
                     index_def_group->front(), ae_expr_group->front())) {
          continue;
        }

        // Check that the outputs we need on the replay match in the index map
        // with this expression.
        auto index_def_outputs = ir_utils::filterByType<IterDomain>(
                                     index_def_group->front()->outputs())
                                     .vector();

        bool outs_match = true;
        for (auto out_i : c10::irange(index_def_outputs.size())) {
          outs_match = outs_match &&
              idGraph(IdMappingMode::INDEX)
                  .disjointIdSets()
                  .strictAreMapped(
                      index_def_outputs[out_i], promoted_outputs[out_i]);
        }

        if (!outs_match) {
          continue;
        }

        // Look for an expression in the group we can reuse.
        //
        // See comment on definition of id_to_index_use
        for (auto maybe_match : *index_def_group) {
          VectorOfUniqueEntries<Expr*> input_uses;
          for (auto inp :
               ir_utils::filterByType<IterDomain>(maybe_match->inputs())) {
            auto use_it = id_to_index_use.find(inp);
            if (use_it == id_to_index_use.end()) {
              continue;
            }
            input_uses.pushBack(use_it->second);
          }

          // If there's already a use, make sure it's this use.
          if (input_uses.subtract({maybe_match}).size() > 0) {
            continue;
          }

          VectorOfUniqueEntries<Expr*> output_defs;
          for (auto out :
               ir_utils::filterByType<IterDomain>(maybe_match->outputs())) {
            auto def_it = id_to_index_def.find(out);
            if (def_it == id_to_index_def.end()) {
              continue;
            }
            output_defs.pushBack(def_it->second);
          }

          // If there's already a def, make sure it's this def.
          if (output_defs.subtract({maybe_match}).size() > 0) {
            continue;
          }

          std::vector<IterDomain*> ae_inps =
              ir_utils::filterByType<IterDomain>(
                  ae_expr_group->front()->inputs())
                  .vector();

          auto maybe_match_inputs =
              ir_utils::filterByType<IterDomain>(maybe_match->inputs())
                  .vector();

          // If there are promoted inputs, we need them to match exactly,
          // otherwise we can't reuse this expression. So although replay is not
          // nullptr, we may set it back and keep looking.
          bool promo_inps_match = true;
          for (auto inp_i : c10::irange(maybe_match_inputs.size())) {
            auto ae_group_pair = ae_graph.disjointIdSet(ae_inps[inp_i]);
            if (ae_group_pair.second &&
                ae_group_2_id.find(ae_group_pair.first) !=
                    ae_group_2_id.end()) {
              auto promo_inp = ae_group_2_id.at(ae_group_pair.first);
              if (promo_inp != maybe_match_inputs[inp_i]) {
                promo_inps_match = false;
              }
            }
          }

          if (!promo_inps_match) {
            continue;
          }

          replay = maybe_match;

          for (auto inp :
               ir_utils::filterByType<IterDomain>(replay->inputs())) {
            id_to_index_use[inp] = replay;
          }

          for (auto out :
               ir_utils::filterByType<IterDomain>(replay->outputs())) {
            id_to_index_def[out] = replay;
          }
          break;
        }

        // No expression we could use found, keep trying.
        if (replay == nullptr) {
          continue;
        }

        std::vector<IterDomain*> ae_inps =
            ir_utils::filterByType<IterDomain>(ae_expr_group->front()->inputs())
                .vector();

        auto replay_inputs =
            ir_utils::filterByType<IterDomain>(replay->inputs()).vector();

        for (auto inp_i : c10::irange(replay_inputs.size())) {
          auto ae_group_pair = ae_graph.disjointIdSet(ae_inps[inp_i]);
          if (!(ae_group_pair.second &&
                ae_group_2_id.find(ae_group_pair.first) !=
                    ae_group_2_id.end())) {
            continue;
          }
          idGraph(IdMappingMode::INDEX)
              .mapIds(
                  replay_inputs[inp_i], ae_group_2_id.at(ae_group_pair.first));
        }
      }

      // No existing expression could be reused.
      if (replay == nullptr) {
        std::vector<IterDomain*> ae_inps_outs =
            ir_utils::filterByType<IterDomain>(ae_expr_group->front()->inputs())
                .vector();
        auto outs = ir_utils::filterByType<IterDomain>(
            ae_expr_group->front()->outputs());
        ae_inps_outs.insert(ae_inps_outs.end(), outs.begin(), outs.end());

        std::unordered_map<IterDomain*, IterDomain*> replacement_map;
        for (auto id : ae_inps_outs) {
          auto ae_group = ae_graph.toGroup(id);
          auto promoted_it = ae_group_2_id.find(ae_group);
          if (promoted_it == ae_group_2_id.end()) {
            replacement_map[id] = id->cloneWithoutRFactor();
          } else {
            replacement_map[id] = promoted_it->second;
          }
        }

        replay =
            addExprWithReplacement(replacement_map, ae_expr_group->front());
        // std::cout << "      ***REPLAY3***:\n        "
        //           << ae_expr_group->front()->toString()
        //           << "        As:" << replay->toString();

      } else {
        // std::cout << "      ***MATCH3***:\n        "
        //           << "        " << replay->toString();
      }

      all_index_exprs.pushBack(replay);
      {
        auto in_ids = ir_utils::filterByType<IterDomain>(replay->inputs());
        all_index_ids.insert(in_ids.begin(), in_ids.end());

        auto out_ids = ir_utils::filterByType<IterDomain>(replay->outputs());
        all_index_ids.insert(out_ids.begin(), out_ids.end());
      }

      std::vector<IterDomain*> ae_inps =
          ir_utils::filterByType<IterDomain>(ae_expr_group->front()->inputs())
              .vector();
      std::vector<IterDomain*> replay_inps =
          ir_utils::filterByType<IterDomain>(replay->inputs()).vector();
      TORCH_INTERNAL_ASSERT(ae_inps.size() == replay_inps.size());

      for (auto inp_i : c10::irange(ae_inps.size())) {
        auto ae_group = ae_graph.toGroup(ae_inps[inp_i]);
        // Only replace if entry does not exist.
        ae_group_2_id.emplace(std::make_pair(ae_group, replay_inps[inp_i]));
      }
    }
  }

  // std::cout << "All indexing expressions (on the index graph): " <<
  // std::endl;
  auto index_expr_groups =
      idGraph(IdMappingMode::INDEX).toGroups(all_index_exprs);

  ExprGroups extraneous_expr_groups =
      ExprGroups(
          idGraph(IdMappingMode::INDEX).disjointExprSets().disjointSets())
          .subtract(index_expr_groups);
  for (auto group : extraneous_expr_groups) {
    idGraph(IdMappingMode::INDEX).eraseExprGroup(group);
  }

  // std::cout << "All index graph exprs: " << std::endl;
  // std::cout << debug::exprGroupsString(idGraph(IdMappingMode::INDEX))
  //           << std::endl;

  return {};
}

} // namespace nvfuser
