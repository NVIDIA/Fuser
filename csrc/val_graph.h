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

#include <iostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace nvfuser {

// ValGraph is a DAG of Vals and Exprs connected by their input and
// output dependencies. Each graph node is a collection of
// either Vals or Exprs that are grouped together through mapVals and
// mapExprs, respectively.
//
// The primary use case of ValGraph is for representing groupings and
// dependencies of iteration domains. For example, given a fusion as
// shown below:
//
// T1 = set(T0);
// T2 = set(T1);
//
// T0: root [I0, I1], loop [I0, I1]
// T1: root [I2, I3], loop [I2*I3/4, 4]
// T2: root [I4, I5], loop [I4*I5/4, 4]
//
// The Exact ValGraph consists of ValGroups of:
//
// - {I0, I2, I4}
// - {I1, I3, I5}
// - {I2*I3, I4*I5}
// - {I2*I3/4, I4*I5/4}
// - {4, 4}
//
// and ExprGroups of:
//
// - {merge of I2 and I3, merge of I4 and I5}
// - {split of I2*I3, split of I4*I5}
//
// ValGraph can be used with any Val types, however, it's currenty
// only tested with IterDomain. Some of the routines might need to be
// extended for other Val types.

using ValGroup = std::shared_ptr<VectorOfUniqueEntries<Val*>>;
using ValGroups = VectorOfUniqueEntries<ValGroup>;
using ExprGroup = std::shared_ptr<VectorOfUniqueEntries<Expr*>>;
using ExprGroups = VectorOfUniqueEntries<ExprGroup>;

class NVF_API ValGraph {
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

  // Returns the disjoint Expr set.
  const DisjointSets<Expr*>& disjointExprSets() const {
    return disjoint_exprs_;
  }

  // Return if there's a group entry in the graph for this expr
  bool hasGroup(Expr* expr) const;

  // Return if there's a group entry in the graph for this val
  bool hasGroup(Val* val) const;

  // Convert expr to its exprGroup, assert that it exists.
  const ExprGroup& toGroup(Expr* expr) const;

  // Convert Val to its ValGroup, assert that it exists.
  const ValGroup& toGroup(Val* val) const;

  // Convert a vector-like container of Val* or Expr* to their
  // ValGroups or ExprGroups. The vector-like container type must
  // define the element type as value_type
  template <
      typename ContainerType,
      typename ElementType = typename std::remove_pointer<
          typename ContainerType::value_type>::type,
      typename RetType = typename std::conditional<
          std::is_base_of<Val, ElementType>::value,
          ValGroups,
          ExprGroups>::type,
      typename = std::enable_if_t<
          std::is_base_of<Val, ElementType>::value ||
          std::is_base_of<Expr, ElementType>::value>>
  RetType toGroups(const ContainerType& entries) const {
    RetType groups;
    for (auto entry : entries) {
      groups.pushBack(toGroup(entry));
    }
    return groups;
  }

  // Return output/input Val groups of provided expr
  // Note that the same ValGroup can show up multiple times, so the
  // output type cannot be VectorOfUniqueEntries
  std::vector<ValGroup> outputGroups(const ExprGroup& expr) const;
  std::vector<ValGroup> inputGroups(const ExprGroup& expr) const;

  // Return Val groups that have no definition.
  ValGroups getTerminatingInputs() const;

  // Recursively traverses uses of the IdGroups in 'of' and returns all
  // ExprGroups that have a use in their definition of provided of IdGroups.
  ExprGroups allUsesOf(const ValGroups& of) const;

  // Recursively traverses definitions of the IdGroups in 'of' and returns all
  // ExprGroups used in this history of defining the 'of' IdGroups.
  ExprGroups allDefinitionsOf(const ValGroups& of) const;

  //! Returns the expressions associated with the
  //! definitions of the provided ValGroup.
  //!
  //! Each ExprGroup of the returned ExprGroup vector is proven to be
  //! equivalent. The ExprGroup vector holds expression groups that are not
  //! equivalent, but produce one of the ValGroups within the same disjoint Val
  //! set.
  const ExprGroups& getDefinitions(const ValGroup& val_group) const;

  //! Same as getDefinitions but for uses instead of
  //! definitions
  const ExprGroups& getUses(const ValGroup& val_group) const;

  bool hasDefinitions(const ValGroup& val_group) const;

  bool hasUses(const ValGroup& val_group) const;

  // Uses the Valgraph to produce mappings between from and to.
  // Supports one to many mappings. If a single Val in from maps to
  // multiple Vals in to, the order of the Vals in value of
  // the map is preserved to be the order provided in to.
  //
  // Example:
  //  tv0: [i0, b1]
  //  tv1: [i2, i3]
  //  tv2: [i4, i5]
  //  tv2 = tv0 + tv1
  //
  //  tv0: [i0*b1] CA(1)
  //  tv1: [i2*i3] CA(1)
  //  tv2: [i4*i5] CA(1)
  //
  // Between tv0 and tv2, the Permissive graph would map:
  //   {i0, i4}
  //   {b1, i5}
  //   {i0*b1, i4*i5}
  //
  // Here, buildMapBetween with:
  //   from: {i0, b1, i0*b1}
  //   to: {i4, i5, i4*i5}
  // will return a map of:
  //   i0: {i4}
  //   b1: {i5}
  //   i0*b1: {i4*i5}
  std::unordered_map<Val*, VectorOfUniqueEntries<Val*>> buildMapBetween(
      const std::vector<Val*>& from,
      const std::vector<Val*>& to) const;

  // Alias of the above on unique vector entries
  std::unordered_map<Val*, VectorOfUniqueEntries<Val*>> buildMapBetween(
      const VectorOfUniqueEntries<Val*>& from,
      const VectorOfUniqueEntries<Val*>& to) const;

  std::string toString() const;

  std::string toGraphvizDotGraph() const;

  void dumpGraphvizDotGraph(const std::string& file_path) const;

  // Initializes entries for the provided Val with its definitions and
  // uses. The provided Val will have its own new ValGroup, each item in the
  // definitions and uses will become a new ExprGroup, and these new ExprGroups
  // will be the definitions and uses of the new ValGroup.
  void initializeVal(
      Val* val,
      const VectorOfUniqueEntries<Expr*>& definitions,
      const VectorOfUniqueEntries<Expr*>& uses);

  // Same as the above exept val->definition() and val->uses() are
  // used
  void initializeVal(Val* val);

  // Initializes entries for the provided Val. The provided Val will be added to
  // the provided existing ValGroup. There will be no changes on the definitions
  // and uses of the provided ValGroup.
  void initializeVal(Val* v, ValGroup vg) {
    disjoint_vals_.appendToSet(v, vg);
  }

  // Add expr to the disjoint sets as a sole group. Used for
  // registering replayed domains and exprs. Error if the expr is
  // already registered.
  void registerExpr(Expr* expr);

  // Returns true if first and second are expressions through which
  // this ValGraph has matching inputs (if forward), or outputs (if not
  // forward). Returning true means the expressions are "the same", in terms
  // they modify matching original inputs by the same amount.
  bool exprsMap(Expr* first, Expr* second, bool forward) const;

  // Check basic consistencies of val and expr groups and their
  // mappings.
  void validateConsistency() const;

  void addUniqueUses(const ValGroup& id_group, const ExprGroup& uses) {
    unique_uses_.at(id_group).pushBack(uses);
  }

  void addUniqueDefinitions(const ValGroup& id_group, const ExprGroup& defs) {
    unique_definitions_.at(id_group).pushBack(defs);
  }

  // Set val0 and val1 to mapped in this graph, attempt to propagate
  // new mapping through val0/val1 definitions/uses.
  void mapVals(Val* val0, Val* val1);

  // Checks if expr0 and expr1 should map together, maps them together, and if
  // expression propagation is on, propagates mapping through
  // them. The forward parameter determines the direction of the
  // propagation. The expressions are mapped if the inputs are mapped
  // when the forward parameter is true. This should
  // be the only call in ValGraph to mapThroughExpr.
  void maybeMapThroughExprs(Expr* expr0, Expr* expr1, bool forward);

  // Can't back prop through merge without making sure one input actually
  // matches. This can be done on a map or extent basis.
  // TODO: Move this to val_graph.cpp once validation_utils.cpp is
  // retired.
  template <typename T>
  static bool shouldMapMergeBackward(
      Merge* merge0,
      Merge* merge1,
      const DisjointSets<T*>& id_sets) {
    auto extent_match = [](IterDomain* id0, IterDomain* id1) -> bool {
      return id0->extent()->sameAs(id1->extent()) ||
          (id0->extent()->isConstInt() && id1->extent()->isConstInt() &&
           id0->extent()->evaluate().as<int64_t>() ==
               id1->extent()->evaluate().as<int64_t>());
    };

    // If one pair of the domains are mapped in the given graph, the
    // backward merge is considered mapped
    if (id_sets.permissiveAreMapped(merge0->outer(), merge1->outer()) ||
        id_sets.permissiveAreMapped(merge0->inner(), merge1->inner())) {
      return true;
    }

    // Considered mapped if the extents are equal
    if (extent_match(merge0->outer(), merge1->outer()) ||
        extent_match(merge0->inner(), merge1->inner())) {
      return true;
    }

    // The mapped ID group may have different extents depending on the
    // mapping conditions. For example, the Permissive graph may have a
    // symbolic extent as well as an extent of 1 for broadcast
    // domains. Those other mapped domains need to be checked as well.

    // First, the outer groups
    auto outer0_group = id_sets.mappingExists(merge0->outer())
        ? id_sets.disjointSetMap().at(merge0->outer())
        : std::make_shared<VectorOfUniqueEntries<T*>>(
              VectorOfUniqueEntries<T*>{merge0->outer()});
    auto outer1_group = id_sets.mappingExists(merge1->outer())
        ? id_sets.disjointSetMap().at(merge1->outer())
        : std::make_shared<VectorOfUniqueEntries<T*>>(
              VectorOfUniqueEntries<T*>{merge1->outer()});

    for (T* outer0 : *outer0_group) {
      for (T* outer1 : *outer1_group) {
        if (extent_match(
                outer0->template as<IterDomain>(),
                outer1->template as<IterDomain>())) {
          return true;
        }
      }
    }

    // Check the inner groups as well if not already matched
    auto inner0_group = id_sets.mappingExists(merge0->inner())
        ? id_sets.disjointSetMap().at(merge0->inner())
        : std::make_shared<VectorOfUniqueEntries<T*>>(
              VectorOfUniqueEntries<T*>{merge0->inner()});
    auto inner1_group = id_sets.mappingExists(merge1->inner())
        ? id_sets.disjointSetMap().at(merge1->inner())
        : std::make_shared<VectorOfUniqueEntries<T*>>(
              VectorOfUniqueEntries<T*>{merge1->inner()});

    for (T* inner0 : *inner0_group) {
      for (T* inner1 : *inner1_group) {
        if (extent_match(
                inner0->template as<IterDomain>(),
                inner1->template as<IterDomain>())) {
          return true;
        }
      }
    }

    return false;
  }

  // Mark val0 and val1 should not be mapped
  void setUnmappable(Val* val0, Val* val1);

  // Mark any of Vals of a given list of Vals should not be mapped
  void setUnmappable(const std::vector<Val*>& vals);

 private:
  // Map expr0 and expr1 with each other, update unique_definitions_
  // unique_uses_
  // TODO: Make this variant hidden?
  void mapExprs(Expr* expr0, Expr* expr1);

  // Checks if expr's are considered "the same" where sameness is
  // defined as inputs and outputs in the same position across
  // expressions are mapped. If the expressions are determined the
  // same then
  //
  // if forward
  //   will map outputs
  // else
  //   will map inputs
  //
  // Returns true if expressions were mapped through.
  bool mapThroughExpr(Expr* first, Expr* second, bool forward);

  // Check if val0 and val1 are marked as unmappable
  bool areUnmappable(Val* val0, Val* val1) const;

 private:
  // If propagate_through_exprs_ = false, then mapThroughExpr will not be called
  // as a consequence of calling mapVals. As well as mapThroughExpr will not be
  // called (again) as a result of calling mapThroughExpr.
  //
  // Note: For the second sentence of above... mapThroughExpr can call mapVals
  // which could in return call mapThoughExpr again, but
  // propagate_through_exprs_ as mentioned above prevents that from happening.
  bool propagate_through_exprs_ = true;

  // Keeps a disjoint set entry for all Vals.
  DisjointSets<Val*> disjoint_vals_;

  // Keeps a disjoint set entry for all Exprs.
  DisjointSets<Expr*> disjoint_exprs_;

  // Definitions of ValGroup. There can be multiple definitions due to
  // replays.
  std::unordered_map<ValGroup, ExprGroups> unique_definitions_;

  std::unordered_map<ValGroup, ExprGroups> unique_uses_;

  // Mapping of a Val to a set of Vals that should be mapped
  std::unordered_map<Val*, std::unordered_set<Val*>> unmappable_vals_;
};

struct ValGroupAndItsGraph {
  ValGroup group;
  ValGraph* graph;
  bool operator==(const ValGroupAndItsGraph& other) const {
    return group == other.group && graph == other.graph;
  }
  bool operator!=(const ValGroupAndItsGraph& other) const {
    return !operator==(other);
  }
  operator const ValGroup&() const {
    return group;
  }
};

inline std::ostream& operator<<(
    std::ostream& os,
    const ValGroupAndItsGraph& g) {
  return os << g.group;
}

// Returns the first pair of id's in ids detected to match each other on the
// given ID graph. TODO: what this is really looking for is if
// there's any overlapping between the iter domains in the provided set.
//
// i.e. if we have:
// tv0 = arange(6).reshape({3, 2})
// tv1 = tv0[3, 2].t()
// tv2 = tv0[3, 2].reshape({2, 3})
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
//
// Note, however, that the above example is not detectable at this
// moment as the self mapping is partial through reshape. The analysis
// below would need to be extended to consider producer and consumers
// of domains as well rather than just root, logical and loop domains.
std::optional<std::pair<IterDomain*, IterDomain*>> detectSelfMapping(
    const std::vector<IterDomain*>& ids,
    const ValGraph& id_graph);

struct SelfMapping {
  IterDomain* id1;
  IterDomain* id2;
  // For debugging, records which domain `id1` and `id2` belong to. This value
  // is either "Root", "Logical", or "Leaf". Consider making it an enum.
  std::string where;
};

// Returns if a self mapping was detected that would invalidate assumptions of
// the overall lowering system.
//
// It is assumed that for any tensor represented by a list of domains,
// those domains should never be mapped with each other. It may be
// possible to lift this assumption, but it's unclear if it could
// matter in practice.
//
// TODO: Can we make this more of an alias analysis?
// Ref: https://github.com/csarofeen/pytorch/pull/1954#discussion_r961940498
std::optional<SelfMapping> hasSelfMapping(
    const TensorView* tv,
    const ValGraph& id_graph);

} // namespace nvfuser
