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

using IdGroup = std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>;
using IdGroups = VectorOfUniqueEntries<IdGroup>;
using ExprGroup = std::shared_ptr<VectorOfUniqueEntries<Expr*>>;
using ExprGroups = VectorOfUniqueEntries<ExprGroup>;

class TORCH_CUDA_CU_API IdGraph {
 public:
  IdGraph() = default;

  IdGraph(const IdGraph& other);
  IdGraph(IdGraph&& other) = default;

  IdGraph& operator=(const IdGraph& other);
  IdGraph& operator=(IdGraph&& other) = default;

  // Returns the disjoint IterDomain set.
  const DisjointSets<IterDomain*>& disjointIdSets() const {
    return disjoint_ids_;
  }

  DisjointSets<IterDomain*>& disjointIdSets() {
    return disjoint_ids_;
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

  // Return if there's a group entry in the graph for this id
  bool hasGroup(IterDomain* id) const;

  // Convert expr to its exprGroup, assert that it exists.
  const ExprGroup& toGroup(Expr* expr) const;

  // Convert iter domain to its IdGroup, assert that it exists.
  const IdGroup& toGroup(IterDomain* id) const;

  // Convert unique vector of expressions to unique vector of its groups
  ExprGroups toGroups(const VectorOfUniqueEntries<Expr*>& exprs) const;

  // Convert unique vector of IterDomain to unique vector of its groups
  IdGroups toGroups(const VectorOfUniqueEntries<IterDomain*>& ids) const;

  // Return output/input iter domain groups of provided expr
  // Note that the same IdGroup can show up multiple times, so the
  // output type cannot be VectorOfUniqueEntries
  std::vector<IdGroup> outputGroups(const ExprGroup& expr) const;
  std::vector<IdGroup> inputGroups(const ExprGroup& expr) const;

  // Recursively traverses uses of the IdGroups in 'of' and returns all
  // ExprGroups that have a use in their definition of provided of IdGroups.
  ExprGroups allUsesOf(const IdGroups& of) const;

  // Recursively traverses definitions of the IdGroups in 'of' and returns all
  // ExprGroups used in this history of defining the 'of' IdGroups.
  ExprGroups allDefinitionsOf(const IdGroups& of) const;

  // Return sorted expressions to go from the provided IterDomains in from to
  // the provided IterDomains in to with provided mode. Minimal expressions to
  // get from 'from' to 'to' returned.
  ExprGroups getExprsBetween(const IdGroups& from, const IdGroups& to) const;

  // Supports one to many mappings, uses the disjoint sets of the provided mode
  // to produce mappings between from and to. If multiple IterDomains in to map
  // to a single iter domain in from, the order of the IterDomains in value of
  // the map is preserved to be the order provided in to.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
  buildMapBetween(
      const std::vector<IterDomain*>& from,
      const std::vector<IterDomain*>& to) const;

  // Alias of the above on unique vector entries
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
  buildMapBetween(
      const VectorOfUniqueEntries<IterDomain*>& from,
      const VectorOfUniqueEntries<IterDomain*>& to) const;

  //! Returns
  //!   (1) The expressions associated with the definitions of the provided
  //!     IterDomain group in the provided mapping mode (if it exists).
  //!   (2) If there is a definitions entry of the provided IterDomain group in
  //!     the provided mapping mode.
  //! First entry in the returned pair is a vector of vector of expressions. The
  //! inner vector is proven to be equivalent based on the provided mode. The
  //! outer vector are expression groups that are not equivalent based on the
  //! provided mode, but produce one of the IterDomains within the same disjoint
  //! Iter Domain set based on the provided mode.
  //!
  //! TODO-NM: ExprGroups is a real container. Consider returning a reference
  std::pair<ExprGroups, bool> getDefinitions(const IdGroup& id_group) const;

  //! Same as iterDomainGroupDefinitions but for uses instead of
  //! definitions
  //!
  //! TODO-NM: ExprGroups is a real container. Consider returning a reference
  std::pair<ExprGroups, bool> getUses(const IdGroup& id_group) const;

  std::string toString() const;

  // Checks if the expression is a trivial operation where an input is simply an
  // output of the transformation. Returns the mapped iter domains if found.
  static std::vector<std::vector<IterDomain*>> isTrivialExpr(Expr* expr);

  // Returns if all atributes of the ID transforms first and second are the same
  static bool transformAtributesMatch(Expr* first, Expr* second);

  // Initializes entries for the provided IterDomain in the IterDomainGraphs
  void initializeId(
      IterDomain* id,
      const VectorOfUniqueEntries<Expr*>& definitions,
      const VectorOfUniqueEntries<Expr*>& uses);

  // Returns if first and second are expressions through which the provided
  // id_map have matching inputs (if forward), or outputs (if not forward).
  // Returning true means the expressions are "the same", in terms they modify
  // matching original extents, by the same amount.
  bool exprsMap(
      Expr* first,
      Expr* second,
      bool forward
      // , std::vector<IterDomain*> second_input_or_output_override
  ) const;

  // Returns entry in unique_definitions_ for provided group in provided mode,
  // otherwise errors if no entry is found.
  const ExprGroups& getUniqueDefinitions(const IdGroup& group) const;

  // Returns entry in unique_uses_ for provided group in provided mode,
  // otherwise errors if no entry is found.
  const ExprGroups& getUniqueUses(const IdGroup& group) const;

 public:
  void addUniqueUses(const IdGroup& id_group, const ExprGroup& uses) {
    unique_uses_.at(id_group).pushBack(uses);
  }

  void addUniqueDefinitions(const IdGroup& id_group, const ExprGroup& defs) {
    unique_definitions_.at(id_group).pushBack(defs);
  }

  // Set id0 and id1 to mapped in disjointIdsSet[mode], attempt to propagate
  // new mapping through id0/id1 definitions/uses.
  void mapIds(IterDomain* id0, IterDomain* id1);

  // Checks if expr0 and expr1 should map together, maps them together, and if
  // expression propagation is on, propagates mapping through them. This should
  // be the only call in IdGraph to mapThroughExpr
  void maybeMapThroughExprs(Expr* expr0, Expr* expr1, bool forward);

  // Map through loop swizzles, as input/output IterDomains are exact, only the
  // order they're traversed differs.
  void mapThroughLoopSwizzles();

  // Maps iter domain pairs returned by calling that return mappings from
  // IdGraph::isTrivialExpr on every expression in the graph.
  void mapThroughTrivialExprs();

  // Removes expressions from unique_definitions_ and unique_uses_ that return
  // mappings from IdGraph::isTrivialExpr
  void removeTrivialExprs();

  // See comment on propagate_expr_ member bool for description
  // Once disabled this can't be reenabled on a graph. If it's reenabled it's
  // hard to predict how mappings will propagate, which will be triggered on the
  // next mapping. To support changing this flag, we should likely run through
  // all expressions currently registered and propagate through all of them on
  // switch. Then once enabled it couldn't be redisabled because we don't record
  // the history of mapId calls.
  void disableExprPropagation() {
    propagate_exprs_ = false;
  }

  // Removes the provided expression group from unique_definitions_ and
  // unique_uses_ breaking traversal through them.
  void eraseExprGroup(const ExprGroup& expr_group);

  // Returns if the expression group has an input id group that matches an
  // output id group. This means traversing on this expression doesn't actually
  // do anything.
  bool isTrivialExprGroup(const ExprGroup& expr_group) const;

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
  // If propagate_exprs_ = false, then mapThroughExpr will not be called as a
  // consequence of calling mapIds. As well as mapThroughExpr will not be called
  // (again) as a result of calling mapThroughExpr.
  //
  // Note: For the second sentence of above... mapThroughExpr can call mapIds
  // which could in return call mapThoughExpr again, but propagate_exprs_ as
  // mentioned above prevents that from happening.
  bool propagate_exprs_ = true;

  // Keeps a disjoint set entry for all IterDomain for all mapping mode types.
  //
  // Using an array here might be nice, but it seems hard to use an enum as an
  // array key
  // https://stackoverflow.com/questions/2102582/how-can-i-count-the-items-in-an-enum
  DisjointSets<IterDomain*> disjoint_ids_;

  // Keeps a disjoint set entry for all Expressions for all mapping mode types.
  DisjointSets<Expr*> disjoint_exprs_;

  // Definitions of IdGroup. There can be multiple definitions due to
  // replays.
  // TODO-NM: IdGroup by a new definition ExprGroup would not be used
  // by existing uses. Does it make sense to represent uses and defs
  // this way? In other words, there is a traversal path from a
  // definition ExprGroup to an IdGroup and its use ExprGroup, but
  // that does't guarantee the path actually exist
  std::unordered_map<IdGroup, ExprGroups> unique_definitions_;

  std::unordered_map<IdGroup, ExprGroups> unique_uses_;
};

} // namespace nvfuser
