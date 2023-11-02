// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <disjoint_set.h>
#include <ir/all_nodes.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace nvfuser {

using ValGroup = std::shared_ptr<VectorOfUniqueEntries<Val*>>;
using ValGroups = VectorOfUniqueEntries<ValGroup>;
using ExprGroup = std::shared_ptr<VectorOfUniqueEntries<Expr*>>;
using ExprGroups = VectorOfUniqueEntries<ExprGroup>;

class ValGraph {
 public:
  ValGraph() = default;

  ValGraph(const ValGraph& other);
  ValGraph(ValGraph&& other) = default;

  ValGraph& operator=(const ValGraph& other);
  ValGraph& operator=(ValGraph&& other) = default;

  ValGraph(bool propagate_through_exprs)
      : propagate_through_exprs_(propagate_through_exprs) {}

  // Returns the disjoint val set.
  const DisjointSets<Val*>& disjointValSets() const {
    return disjoint_vals_;
  }

  DisjointSets<Val*>& disjointValSets() {
    return disjoint_vals_;
  }

  // Returns the disjoint Expr set.
  const DisjointSets<Expr*>& disjointExprSets() const {
    return disjoint_exprs_;
  }

  DisjointSets<Expr*>& disjointExprSets() {
    return disjoint_exprs_;
  }

  // Return if there's a group entry in the graph for this expr
  bool hasGroup(Expr* expr) const;

  // Return if there's a group entry in the graph for this val
  bool hasGroup(Val* val) const;

  // Convert expr to its exprGroup, assert that it exists.
  const ExprGroup& toGroup(Expr* expr) const;

  // Convert iter domain to its IdGroup, assert that it exists.
  const ValGroup& toGroup(Val* val) const;

  // Return output/input iter domain groups of provided expr
  // Note that the same IdGroup can show up multiple times, so the
  // output type cannot be VectorOfUniqueEntries
  std::vector<ValGroup> outputGroups(const ExprGroup& expr) const;
  std::vector<ValGroup> inputGroups(const ExprGroup& expr) const;

  // Recursively traverses uses of the IdGroups in 'of' and returns all
  // ExprGroups that have a use in their definition of provided of IdGroups.
  ExprGroups allUsesOf(const ValGroups& of) const;

  // Recursively traverses definitions of the IdGroups in 'of' and returns all
  // ExprGroups used in this history of defining the 'of' IdGroups.
  ExprGroups allDefinitionsOf(const ValGroups& of) const;

  //! Returns the pointer to expressions associated with the
  //! definitions of the provided IterDomain group in the provided
  //! mapping mode (if it exists). Nullptr is returned otherwise.
  //!
  //! The returned pointer is to a vector of vector of expressions. The
  //! inner vector is proven to be equivalent based on the provided mode. The
  //! outer vector are expression groups that are not equivalent based on the
  //! provided mode, but produce one of the IterDomains within the same disjoint
  //! Iter Domain set based on the provided mode.
  const ExprGroups* getDefinitions(const ValGroup& id_group) const;

  //! Same as iterDomainGroupDefinitions but for uses instead of
  //! definitions
  const ExprGroups* getUses(const ValGroup& id_group) const;

  bool hasDefinitions(const ValGroup& id_group) const;

  bool hasUses(const ValGroup& id_group) const;

  std::string toString() const;

  // Checks if the expression is a trivial operation where an input is simply an
  // output of the transformation. Returns the mapped iter domains if found.
  static std::vector<std::vector<Val*>> isTrivialExpr(Expr* expr);

  // Returns if all atributes of the exprs first and second are the same
  static bool exprAttributesMatch(Expr* first, Expr* second);

  // Initializes entries for the provided IterDomain in the IterDomainGraphs
  void initializeVal(
      Val* id,
      const VectorOfUniqueEntries<Expr*>& definitions,
      const VectorOfUniqueEntries<Expr*>& uses);

  // Returns if first and second are expressions through which the provided
  // id_map have matching inputs (if forward), or outputs (if not forward).
  // Returning true means the expressions are "the same", in terms they modify
  // matching original extents, by the same amount.
  bool exprsMap(Expr* first, Expr* second, bool forward) const;

 public:
  void addUniqueUses(const ValGroup& id_group, const ExprGroup& uses) {
    unique_uses_.at(id_group).pushBack(uses);
  }

  void addUniqueDefinitions(const ValGroup& id_group, const ExprGroup& defs) {
    unique_definitions_.at(id_group).pushBack(defs);
  }

  // Set val0 and val1 to mapped in disjointValsSet[mode], attempt to propagate
  // new mapping through val0/val1 definitions/uses.
  void mapVals(Val* val0, Val* val1);

  // Checks if expr0 and expr1 should map together, maps them together, and if
  // expression propagation is on, propagates mapping through them. This should
  // be the only call in IdGraph to mapThroughExpr
  void maybeMapThroughExprs(Expr* expr0, Expr* expr1, bool forward);

  // Map through loop swizzles, as input/output IterDomains are exact, only the
  // order they're traversed differs.
  void mapThroughLoopSwizzles();

  void setPropagateThroughExprs(bool b) {
    propagate_through_exprs_ = b;
  }

 private:
  // Map expr0 and expr1 with eachother, update unique_definitions_ unique_uses_
  // TODO: Make this variant hidden?
  void mapExprs(Expr* expr0, Expr* expr1);

  // Checks if expr's are considered "the same" where sameness inputs and
  // outputs in the same position across expressions map with  provided
  // MappingMode. If the expressions are determined the same then
  // if forward
  //   will map outputs
  // else
  //   will map inputs
  // in the provided mode.
  // Returns if expressions were mapped through.
  //
  bool mapThroughExpr(Expr* first, Expr* second, bool forward);

 private:
  // If propagate_through_exprs_ = false, then mapThroughExpr will not be called
  // as a consequence of calling mapIds. As well as mapThroughExpr will not be
  // called (again) as a result of calling mapThroughExpr.
  //
  // Note: For the second sentence of above... mapThroughExpr can call mapIds
  // which could in return call mapThoughExpr again, but propagate_exprs_ as
  // mentioned above prevents that from happening.
  bool propagate_through_exprs_ = true;

  // Keeps a disjoint set entry for all IterDomain for all mapping mode types.
  //
  // Using an array here might be nice, but it seems hard to use an enum as an
  // array key
  // https://stackoverflow.com/questions/2102582/how-can-i-count-the-items-in-an-enum
  DisjointSets<Val*> disjoint_vals_;

  // Keeps a disjoint set entry for all Expressions for all mapping mode types.
  DisjointSets<Expr*> disjoint_exprs_;

  // Definitions of IdGroup. There can be multiple definitions due to
  // replays.
  std::unordered_map<ValGroup, ExprGroups> unique_definitions_;

  std::unordered_map<ValGroup, ExprGroups> unique_uses_;
};

} // namespace nvfuser
