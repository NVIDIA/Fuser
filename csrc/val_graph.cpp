// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/to_string.h>
#include <ir/utils.h>
#include <val_graph.h>

namespace nvfuser {

namespace {
using UnorderedSetOfExprGroup = std::unordered_set<ExprGroup>;
using DequeOfExprGroup = std::deque<ExprGroup>;
} // namespace

ValGraph::ValGraph(const ValGraph& other)
    : disjoint_vals_(other.disjoint_vals_),
      disjoint_exprs_(other.disjoint_exprs_),
      unique_definitions_(),
      unique_uses_() {
  for (const auto& [orig_val_group, orig_expr_groups] :
       other.unique_definitions_) {
    auto new_val_group = toGroup(orig_val_group->front());

    ExprGroups new_expr_groups;
    for (const ExprGroup& orig_expr_group : orig_expr_groups) {
      new_expr_groups.pushBack(toGroup(orig_expr_group->front()));
    }

    unique_definitions_[new_val_group] = new_expr_groups;
  }

  for (const auto& [orig_val_group, orig_expr_groups] : other.unique_uses_) {
    auto new_val_group = toGroup(orig_val_group->front());

    ExprGroups new_expr_groups;
    for (const ExprGroup& orig_expr_group : orig_expr_groups) {
      new_expr_groups.pushBack(toGroup(orig_expr_group->front()));
    }

    unique_uses_[new_val_group] = new_expr_groups;
  }
}

ValGraph& ValGraph::operator=(const ValGraph& other) {
  disjoint_vals_.clear();
  disjoint_exprs_.clear();
  unique_definitions_.clear();
  unique_uses_.clear();
  ValGraph copy(other);
  std::swap(*this, copy);
  return *this;
}

// Return if there's a group entry in the graph for this expr
bool ValGraph::hasGroup(Expr* expr) const {
  return disjoint_exprs_.mappingExists(expr);
}

// Return if there's a group entry in the graph for this val
bool ValGraph::hasGroup(Val* val) const {
  return disjoint_vals_.mappingExists(val);
}

const ExprGroup& ValGraph::toGroup(Expr* expr) const {
  auto disjoint_set_it = disjoint_exprs_.disjointSetMap().find(expr);
  NVF_ERROR(
      disjoint_set_it != disjoint_exprs_.disjointSetMap().end(),
      "\nExpr group could not be found in graph associated with: ",
      expr->toString());
  return disjoint_set_it->second;
}

const ValGroup& ValGraph::toGroup(Val* val) const {
  auto disjoint_set_it = disjoint_vals_.disjointSetMap().find(val);
  NVF_ERROR(
      disjoint_set_it != disjoint_vals_.disjointSetMap().end(),
      "\nId group could not be found in graph associated with: ",
      val->toString(),
      "\n");
  return disjoint_set_it->second;
}

ExprGroups ValGraph::toGroups(const VectorOfUniqueEntries<Expr*>& exprs) const {
  ExprGroups expr_groups;
  for (auto expr : exprs) {
    expr_groups.pushBack(toGroup(expr));
  }
  return expr_groups;
}

ValGroups ValGraph::toGroups(const VectorOfUniqueEntries<Val*>& vals) const {
  ValGroups val_groups;
  for (auto val : vals) {
    val_groups.pushBack(toGroup(val));
  }
  return val_groups;
}

std::vector<ValGroup> ValGraph::outputGroups(const ExprGroup& expr) const {
  std::vector<ValGroup> output_groups;
  for (auto id_output : expr->front()->outputs()) {
    output_groups.push_back(toGroup(id_output));
  }
  return output_groups;
}

std::vector<ValGroup> ValGraph::inputGroups(const ExprGroup& expr) const {
  std::vector<ValGroup> input_groups;
  for (auto id_input : expr->front()->inputs()) {
    input_groups.push_back(toGroup(id_input));
  }
  return input_groups;
}

ExprGroups ValGraph::allUsesOf(const ValGroups& of) const {
  DequeOfExprGroup to_visit;
  for (const ValGroup& of_val_group : of) {
    if (const ExprGroups* group_uses = getUses(of_val_group);
        group_uses != nullptr) {
      to_visit.insert(to_visit.end(), group_uses->begin(), group_uses->end());
    }
  }

  UnorderedSetOfExprGroup visited;
  while (!to_visit.empty()) {
    ExprGroup current_expr = to_visit.front();
    to_visit.pop_front();
    visited.emplace(current_expr);
    for (const ValGroup& output_id : outputGroups(current_expr)) {
      if (const ExprGroups* group_uses = getUses(output_id);
          group_uses != nullptr) {
        for (const ExprGroup& group_use : *group_uses) {
          if (visited.count(group_use)) {
            continue;
          }
          to_visit.push_back(group_use);
        }
      }
    }
  }

  return visited;
}

ExprGroups ValGraph::allDefinitionsOf(const ValGroups& of) const {
  DequeOfExprGroup to_visit;
  for (const ValGroup& of_val_group : of) {
    if (const ExprGroups* group_defs = getDefinitions(of_val_group);
        group_defs != nullptr) {
      to_visit.insert(to_visit.end(), group_defs->begin(), group_defs->end());
    }
  }

  UnorderedSetOfExprGroup visited;
  while (!to_visit.empty()) {
    ExprGroup current_expr = to_visit.front();
    to_visit.pop_front();
    visited.emplace(current_expr);
    for (const ValGroup& input_id : inputGroups(current_expr)) {
      if (const ExprGroups* group_defs = getDefinitions(input_id);
          group_defs != nullptr) {
        for (const ExprGroup& group_def : *group_defs) {
          if (visited.count(group_def)) {
            continue;
          }
          to_visit.push_back(group_def);
        }
      }
    }
  }

  return visited;
}

std::unordered_map<Val*, VectorOfUniqueEntries<Val*>> ValGraph::buildMapBetween(
    const std::vector<Val*>& from,
    const std::vector<Val*>& to) const {
  std::unordered_map<Val*, ValGroup> from_ids2set;

  for (auto from_id : from) {
    if (!hasGroup(from_id)) {
      continue;
    }
    from_ids2set[from_id] = toGroup(from_id);
  }

  // Map from the sets associated with the IterDomains in to, to those iter
  // domains
  std::unordered_map<ValGroup, VectorOfUniqueEntries<Val*>> set2to_ids;

  for (auto to_id : to) {
    if (!hasGroup(to_id)) {
      continue;
    }
    auto to_set = toGroup(to_id);
    auto set2to_ids_it = set2to_ids.find(to_set);

    if (set2to_ids_it == set2to_ids.end()) {
      set2to_ids[to_set] = {to_id};
    } else {
      set2to_ids[to_set].pushBack(to_id);
    }
  }

  std::unordered_map<Val*, VectorOfUniqueEntries<Val*>> from_ids2to_ids;
  for (auto from_id : from) {
    from_ids2to_ids[from_id] = VectorOfUniqueEntries<Val*>();

    auto from_it = from_ids2set.find(from_id);
    NVF_ERROR(from_it != from_ids2set.end());

    auto from_set = from_it->second;
    auto to_entry_it = set2to_ids.find(from_set);
    if (to_entry_it == set2to_ids.end()) {
      continue;
    }
    from_ids2to_ids[from_id] = to_entry_it->second;
  }
  return from_ids2to_ids;
}

std::unordered_map<Val*, VectorOfUniqueEntries<Val*>> ValGraph::buildMapBetween(
    const VectorOfUniqueEntries<Val*>& from,
    const VectorOfUniqueEntries<Val*>& to) const {
  return buildMapBetween(from.vector(), to.vector());
}

bool ValGraph::hasDefinitions(const ValGroup& val_group) const {
  NVF_ERROR(val_group);
  return unique_definitions_.find(val_group) != unique_definitions_.end();
}

bool ValGraph::hasUses(const ValGroup& val_group) const {
  NVF_ERROR(val_group);
  return unique_uses_.find(val_group) != unique_uses_.end();
}

std::string ValGraph::toString() const {
  std::stringstream ss;
  ss << "IdGraph { \n";
  ss << "Disjoint Ids:\n"
     << idGroupsString(*this, 1) << "\n\nDisjoint Expression groups:\n"
     << exprGroupsString(*this, 1) << std::endl;
  ss << " } IdGraph\n" << std::endl;
  return ss.str();
}

void ValGraph::initializeVal(
    Val* val,
    const VectorOfUniqueEntries<Expr*>& definitions,
    const VectorOfUniqueEntries<Expr*>& uses) {
  const ValGroup& val_disjoint_set =
      disjointValSets().initializeSet(val).first->second;

  ExprGroups def_groups;
  for (auto def : definitions) {
    const ExprGroup& expr_set =
        disjointExprSets().initializeSet(def).first->second;
    def_groups.pushBack(expr_set);
  }
  // TODO-NM: def_groups can be empty. Should it be still mapped?
  NVF_ERROR(
      unique_definitions_.emplace(val_disjoint_set, def_groups).second,
      "Multiple defining groups for ",
      nvfuser::toString(val_disjoint_set));

  ExprGroups use_groups;
  for (auto use : uses) {
    const ExprGroup& expr_set =
        disjointExprSets().initializeSet(use).first->second;
    use_groups.pushBack(expr_set);
  }
  // TODO-NM: use_groups can be empty. Should it be still mapped?
  NVF_ERROR(
      unique_uses_.emplace(val_disjoint_set, use_groups).second,
      "Multiple use groups for ",
      nvfuser::toString(val_disjoint_set));
}

void ValGraph::initializeVal(Val* val) {
  VectorOfUniqueEntries<Expr*> defs;
  if (val->definition()) {
    defs.pushBack(val->definition());
  }
  VectorOfUniqueEntries<Expr*> uses;
  for (Expr* use : val->uses()) {
    uses.pushBack(use);
  }
  initializeVal(val, defs, uses);
}

bool ValGraph::exprsMap(Expr* first, Expr* second, bool forward) const {
  NVF_ERROR(first);
  NVF_ERROR(second);

  if (!first->sameOp(second)) {
    return false;
  }

  std::vector<Val*> first_vals = forward ? first->inputs() : first->outputs();
  std::vector<Val*> second_vals =
      forward ? second->inputs() : second->outputs();

  NVF_ERROR(
      first_vals.size() == second_vals.size(),
      "Expected number of ",
      (forward ? "inputs" : "outputs"),
      " to match for\n",
      first->toString(),
      second->toString());

  for (const auto i : c10::irange(first_vals.size())) {
    if (!disjointValSets().permissiveAreMapped(
            first_vals.at(i), second_vals.at(i))) {
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

    auto extent_o_match = extent_0o->sameAs(extent_1o) ||
        (extent_0o->isConstInt() && extent_1o->isConstInt() &&
         extent_0o->evaluate() == extent_1o->evaluate()) ||
        disjointValSets().permissiveAreMapped(merge0->outer(), merge1->outer());

    auto extent_i_match = extent_0i->sameAs(extent_1i) ||
        (extent_0i->isConstInt() && extent_1i->isConstInt() &&
         extent_0i->evaluate() == extent_1i->evaluate()) ||
        disjointValSets().permissiveAreMapped(merge0->inner(), merge1->inner());

    if (!(extent_o_match || extent_i_match)) {
      return false;
    }
  }

  // TODO: For now we're using same as, however we could know what val's are
  // exactly the same given the exact map. We might want to pipe that
  // information through to here.

  return true;
}

const ExprGroups* ValGraph::getDefinitions(const ValGroup& val_group) const {
  NVF_ERROR(val_group, "Nullptr not allowed");
  if (auto it = unique_definitions_.find(val_group);
      it != unique_definitions_.end()) {
    return &(it->second);
  } else {
    return nullptr;
  }
}

const ExprGroups* ValGraph::getUses(const ValGroup& val_group) const {
  NVF_ERROR(val_group, "Nullptr not allowed");
  if (auto it = unique_uses_.find(val_group); it != unique_uses_.end()) {
    return &(it->second);
  } else {
    return nullptr;
  }
}

void ValGraph::mapVals(Val* val0, Val* val1) {
  if (val0 == val1) {
    return;
  }

  if (disjointValSets().strictAreMapped(val0, val1)) {
    return;
  }
  // Definitions and uses are based on the groups of id0 and id1, don't merge
  // them into a single group until we grab all definitions and uses for later
  // processing.
  ValGroup orig_val_group0 = toGroup(val0);
  ValGroup orig_val_group1 = toGroup(val1);
  const ExprGroups* orig_defs0 = getDefinitions(orig_val_group0);
  NVF_ERROR(orig_defs0);
  const ExprGroups* orig_defs1 = getDefinitions(orig_val_group1);
  NVF_ERROR(orig_defs1);
  const ExprGroups* orig_uses0 = getUses(orig_val_group0);
  NVF_ERROR(orig_uses0);
  const ExprGroups* orig_uses1 = getUses(orig_val_group1);
  NVF_ERROR(orig_uses1);

  // Map the iter domains together before we traverse across definitions and
  // uses. Traversing definitions and uses could use the new property of id0 and
  // id1 being mapped.
  disjointValSets().mapEntries(val0, val1);
  auto new_val_group = toGroup(val0);

  unique_definitions_[new_val_group] = orig_defs0->computeUnion(*orig_defs1);
  unique_uses_[new_val_group] = orig_uses0->computeUnion(*orig_uses1);

  // Propagate on uses
  if (!orig_uses0->empty() && !orig_uses1->empty()) {
    for (const ExprGroup& use_group_1 : *orig_uses1) {
      for (const ExprGroup& use_group_0 : *orig_uses0) {
        if (use_group_0 == use_group_1) {
          continue;
        }
        Expr* use0 = use_group_0->front();
        Expr* use1 = use_group_1->front();
        maybeMapThroughExprs(use0, use1, true);
      }
    }
  }

  // Propagate on definitions
  if (!orig_defs0->empty() && !orig_defs1->empty()) {
    for (const ExprGroup& def_group_1 : *orig_defs1) {
      for (const ExprGroup& def_group_0 : *orig_defs0) {
        if (def_group_0 == def_group_1) {
          continue;
        }
        auto def0 = def_group_0->front();
        auto def1 = def_group_1->front();
        maybeMapThroughExprs(def0, def1, false);
      }
    }
  }

  unique_definitions_.erase(orig_val_group0);
  unique_definitions_.erase(orig_val_group1);
  unique_uses_.erase(orig_val_group0);
  unique_uses_.erase(orig_val_group1);
}

void ValGraph::maybeMapThroughExprs(Expr* expr0, Expr* expr1, bool forward) {
  if (!exprsMap(expr0, expr1, forward)) {
    return;
  }

  // Expr inputs are mapped. If propagate_exprs_ is true, map the
  // exprs and outputs
  if (propagate_through_exprs_) {
    mapExprs(expr0, expr1);
    mapThroughExpr(expr0, expr1, forward);
  } else if (
      inputGroups(toGroup(expr0)) == inputGroups(toGroup(expr1)) &&
      outputGroups(toGroup(expr0)) == outputGroups(toGroup(expr1))) {
    mapExprs(expr0, expr1);
  }
}

void ValGraph::mapExprs(Expr* expr0, Expr* expr1) {
  if (expr0 == expr1) {
    return;
  }

  if (disjointExprSets().strictAreMapped(expr0, expr1)) {
    return;
  }

  const ExprGroup& expr0_orig_group = toGroup(expr0);
  const ExprGroup& expr1_orig_group = toGroup(expr1);

  disjointExprSets().mapEntries(expr0, expr1);

  const ExprGroup& expr_new_group = toGroup(expr0);

  // Update unique uses of producers
  ValGroups producers;
  for (auto expr : std::vector<Expr*>{expr0, expr1}) {
    for (auto input : expr->inputs()) {
      producers.pushBack(toGroup(input));
    }
  }

  for (const ValGroup& producer_group : producers) {
    unique_uses_.at(producer_group).erase(expr0_orig_group);
    unique_uses_.at(producer_group).erase(expr1_orig_group);
    unique_uses_.at(producer_group).pushBack(expr_new_group);
  }

  // Update unique definitinos of consumers
  ValGroups consumers;
  for (auto expr : std::vector<Expr*>{expr0, expr1}) {
    for (auto output : expr->outputs()) {
      consumers.pushBack(toGroup(output));
    }
  }

  for (const ValGroup& consumer_group : consumers) {
    unique_definitions_.at(consumer_group).erase(expr0_orig_group);
    unique_definitions_.at(consumer_group).erase(expr1_orig_group);
    unique_definitions_.at(consumer_group).pushBack(expr_new_group);
  }
}

bool ValGraph::mapThroughExpr(Expr* first, Expr* second, bool forward) {
  if (first == nullptr || second == nullptr) {
    return false;
  }

  if (!exprsMap(first, second, forward)) {
    return false;
  }

  NVF_ERROR(
      propagate_through_exprs_,
      "Asked to propagate expression mappings on a graph that has propagate_exprs_ disabled.");

  const auto& first_ids = forward ? first->outputs() : first->inputs();
  const auto& second_ids = forward ? second->outputs() : second->inputs();

  NVF_ERROR(
      first_ids.size() == second_ids.size(),
      "This should be unreachable, if transformation expressions match, their number of inputs and outputs should as well.\n However found:\n",
      first->toString(),
      "\nand\n",
      second->toString());
  for (auto out_i : c10::irange(first_ids.size())) {
    mapVals(first_ids[out_i], second_ids[out_i]);
  }

  return true;
}

void ValGraph::removeTrivialExprs() {
  ExprGroups trivial_expr_groups;
  // This seems like it shouls just be a copy if.
  for (const ExprGroup& expr_group : disjointExprSets().disjointSets()) {
    if (isTrivialExprGroup(expr_group)) {
      trivial_expr_groups.pushBack(expr_group);
    }
  }

  // Clear out expressions that map inputs and outputs to the same group
  // from definitions and uses. They shouldn't be important in traversal, and
  // will break the terminal input/terminal output logic of traversal. Similar
  // to what's drafted in buildIndexGraph
  for (const ExprGroup& trivial_expr_group : trivial_expr_groups) {
    // Complexity of erase not good as both disjoint set and vector of unique
    // entries require a vector find to erase an entry.
    eraseExprGroup(trivial_expr_group);
  }
}

// Complexity here is not great. We might want a better complexity version when
// erasing multiple expr_groups.
void ValGraph::eraseExprGroup(const ExprGroup& expr_group) {
  // Erase entries that exist in unique_definitions_ and unique_uses_
  for (const ValGroup& id_group : disjointValSets().disjointSets()) {
    // Make sure the entries exists
    NVF_ERROR(
        unique_definitions_.find(id_group) != unique_definitions_.end(),
        "Broken definitions, couldn't find entry for id group, ",
        nvfuser::toString(id_group, 0, true));
    NVF_ERROR(
        unique_uses_.find(id_group) != unique_uses_.end(),
        "Broken uses, couldn't find entry for id group, ",
        nvfuser::toString(id_group, 0, true));

    unique_definitions_[id_group].erase(expr_group);
    unique_uses_[id_group].erase(expr_group);
  }

  for (auto expr : *expr_group) {
    disjoint_exprs_.erase(expr);
  }
}

bool ValGraph::isTrivialExprGroup(const ExprGroup& expr_group) const {
  return !ValGroups(inputGroups(expr_group))
              .intersect(ValGroups(outputGroups(expr_group)))
              .empty();
}

} // namespace nvfuser
