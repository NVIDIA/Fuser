// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <val_graph.h>

namespace nvfuser {

class IdModel;
struct StatefulInliningInfo;

struct CoveredGroup;

using CoveredGroups = std::unordered_set<CoveredGroup>;

std::string toString(const CoveredGroups& covered_groups);

// Represents an input (or split output) ID group that an exact group
// depends on (i.e., covers). If an input ID group is split, split_in_
// refers to the covered groups of the input ID group, and group_
// refers to either inner or outer output group.
struct CoveredGroup {
  CoveredGroup() = default;
  CoveredGroup(
      ValGroup group,
      std::shared_ptr<CoveredGroups> split_in = nullptr,
      bool is_inner = false)
      : group_(std::move(group)),
        split_in_(std::move(split_in)),
        is_inner_(is_inner) {}

  const ValGroup& group() const {
    return group_;
  }

  const std::shared_ptr<CoveredGroups>& splitIn() const {
    return split_in_;
  }

  bool isInner() const {
    return is_inner_;
  }

  // Note that the equality of this information is only determined by
  // group_ and that split_in_ does not matter.
  bool operator==(const CoveredGroup& other) const {
    return group_ == other.group_;
  }

  bool operator!=(const CoveredGroup& other) const {
    return !(group_ == other.group_);
  }

  // Check if this CoveredGroup is equal to or covers a given other
  // CoveredGroup
  bool isEqualToOrSuperSetOf(const CoveredGroup& other) const;

  std::string toString() const;

 private:
  // Covered group
  ValGroup group_;
  // If this group is an output of a split, keep track of the covered
  // groups of the split input group.
  std::shared_ptr<CoveredGroups> split_in_;
  // Indicates if the split is inner or not. Not relevant if split_in_
  // is nullptr.
  bool is_inner_ = false;
};

// Returns true if covered_groups_x is equal to or a superset of
// covered_groups_y, that is, for all of CoveredGroup of
// covered_groups_y, if there's a CoveredGroup in covered_groups_x
// that is equal or a superset.
bool isEqualToOrSuperSetOf(
    const CoveredGroups& covered_groups_x,
    const CoveredGroups& covered_groups_y);

} // namespace nvfuser

namespace std {
template <>
struct hash<nvfuser::CoveredGroup> {
  size_t operator()(const nvfuser::CoveredGroup& x) const {
    return std::hash<nvfuser::ValGroup>()(x.group());
  }
};
} // namespace std

namespace nvfuser {

// Callback interface for LoopPromotionMapBuilder. Allow exposing the
// temporary maps for testing and debugging
class LoopPromotionMapBuilderCallback {
 public:
  virtual ~LoopPromotionMapBuilderCallback() = default;

  // Called after Step 1 with the root resolution map and the
  // corresponding IEL graph
  virtual void postStep1(
      const std::unordered_map<ValGroup, IterDomain*>& iel_root_resolution_map,
      const ValGraph& iel_graph) {}
  // Called after Step 2 with the IEL promotion map and the
  // corresponding IEL graph
  virtual void postStep2(
      const std::unordered_map<ValGroup, IterDomain*>& iel_promotion_map,
      const ValGraph& iel_graph) {}
  // Called after Step 3 with the loop promotion map
  virtual void postStep3(
      const std::unordered_map<ValGroup, IterDomain*>& loop_promotion_map) {}
  // Called after Step 4 with the IEL promotion map and the
  // corresponding IEL graph
  virtual void postStep4(
      const std::unordered_map<ValGroup, IterDomain*>& iel_promotion_map,
      const ValGraph& iel_graph) {}
  // Called after Step 3 with the final loop promotion map
  virtual void postStep5(
      const std::unordered_map<ValGroup, IterDomain*>& loop_promotion_map) {}
};

class LoopPromotionMapBuilder {
 public:
  // Build a map of loop groups to IterDomains that represent actual
  // loops. The map is built based on the broadcast resolution with
  // root domains between inlined producer and consumer tensors.
  //
  // (For debugging only) When force_full_loop_promotion_analysis is
  // true, it always performs the full loop promotion analysis even
  // when it's possible to take a quicker shortcut.
  static std::unordered_map<ValGroup, IterDomain*> get(
      IdModel& id_model,
      const StatefulInliningInfo& inlining_info,
      LoopPromotionMapBuilderCallback* callback = nullptr,
      bool force_full_loop_promotion_analysis = false);

  // Computes coverage info of each exact group. Coverage is
  // represented as a set of CoveredGroup, which is either an exact
  // group of input IDs or an output group of split.
  static std::unordered_map<ValGroup, std::shared_ptr<CoveredGroups>>
  computeCoveredGroups(const ValGraph& exact_graph, const IdModel& id_model);

 private:
  LoopPromotionMapBuilder(
      IdModel& id_model,
      const StatefulInliningInfo& inlining_info,
      LoopPromotionMapBuilderCallback* callback = nullptr,
      bool force_full_loop_promotion_analysis = false);

  // Given an Exact graph, get val groups that should be used as
  // starting groups when propagating promotion info. For non-cyclic
  // graphs, this should be equivalent to what ValGraph::getTerminatingInputs()
  // returns. For cyclic graphs, there may be no terminating inputs
  // due to a cyclic dependency, so getTerminatingInputs() may return
  // just nothing.
  //
  // Instead, we first find input iter domains, which are (maybe) root
  // iter domains that have no corresponding producer iter domains as
  // defined by PairwiseLogicalDomainMap. Any exact groups that
  // include any of the input iter domains are considered input
  // groups.
  //
  // For example, given a graph like shown below:
  //
  //  i0 -> i1 -> i2 -> i3
  //   ^          |
  //   +----------+
  //
  // Here, i0 represents a Val group that contains IDs of fusion input
  // tensors.
  //
  // ValGraph::getTerminatingInputs would return nothing as there's no
  // terminating input. However, when this is used in
  // computeCoveredGroups, what we need to do is to propagate the
  // informatiom of the IDs of the fusion inputs, i.e., i0, so the
  // propagation should start from i0, then i1, i2 and i3, ignoring
  // the back edge to i0.
  static ValGroups getInputGroupsOfExactGraph(
      const ValGraph& exact_graph,
      const IdModel& id_model);

  // Similar to getInputGroupsOfExactGraph but for an IEL graph.
  // We first get the inputs of the Exact graph. For the
  // IEL propagation, any IEL group that has an ID that is included
  // in any of the input groups of the exact graph is used as an input.
  static ValGroups getInputGroupsOfIELGraph(
      const ValGraph& iel_graph,
      const IdModel& id_model);

  std::unordered_map<ValGroup, IterDomain*> build();

  // Shortcut to build a map of promotion IDs without doing the full
  // loop promotion analysis. Can only be used when the full analysis
  // is not requierd.
  std::unordered_map<ValGroup, IterDomain*> buildWithNoBroadcast();

  ValGraph& idGraph(IdMappingMode mode);
  const ValGraph& idGraph(IdMappingMode mode) const;

  std::unordered_map<ValGroup, IterDomain*> buildInlineRootResolutionMap(
      const ValGraph& iel_graph,
      const StatefulInliningInfo& info) const;

  // Helper function for building loop promotion map.
  //
  // Propagate promotion mappings from root IEL groups to intermediate
  // and loop IEL groups by traversing IEL exprs. For each expr, if an
  // input is promoted, the output needs to be promoted too. If
  // there's already an equivalent expr that uses the promoted inputs,
  // create a mapping from the outputs of the IEL expr to the outputs
  // of the equivalent expr. We only consider exprs that are mapped
  // in the loop graph as we are looking for domains that represent
  // the actual loops of the input and output domains of the IEL
  // expr. If no such expr is found, the IEL expr is replayed with the
  // promoted inputs.
  //
  // This is used twice when building the promotion map. The first time
  // it is used there's no loop graph promotion yet, so only the IEL
  // promotions are propagated. In that case, loop_graph_promotion_map
  // should be just empty.
  //
  // Propagation uses iel_promotion_map and
  // loop_graph_promotion_map. If both are available for an IEL group,
  // the former has the precedence. This is because when this function
  // is used for step 4, the given iel_promotion_map starts as an
  // empty map and gets populated during this propagation, so any
  // mapping in the map is guaranteed to be the correct final mapping,
  // whereas the loop graph may have invalid mappings for partially
  // inlined domains.
  void propagatePromotionsInIELGraph(
      const ValGraph& iel_graph,
      std::unordered_map<ValGroup, IterDomain*>& iel_promotion_map,
      const ValGraph& loop_graph,
      const std::unordered_map<ValGroup, IterDomain*>& loop_promotion_map);

  // Same as the other propagatePromotionsInIELGraph but without loop
  // graph map. This is used for step 2, where there's no loop
  // graph map yet.
  void propagatePromotionsInIELGraph(
      const ValGraph& iel_graph,
      std::unordered_map<ValGroup, IterDomain*>& iel_promotion_map);

  // Given an IEL promotion map, identify the mapping of each loop
  // group. The promotion must represent all the domains in each loop
  // group. If a valid representative promotion is not found for a
  // loop group, no mapping is added for the group.
  std::unordered_map<ValGroup, IterDomain*> projectIELPromotionToLoopGraph(
      const ValGraph& iel_graph,
      const std::unordered_map<ValGroup, IterDomain*>& iel_promotion_map,
      const ValGraph& loop_graph,
      const StatefulInliningInfo& inlining_info) const;

  // Find a promoted iter domain of a given loop group that covers all
  // the exact groups representative of the resolved transformations
  // within the loop group. Specifically, we examine each IEL group of
  // the loop group, and if an IEL group has a promotion, we consider it as a
  // candidate of the promotion of this loop group. If not, we include a
  // domain of the IEL group as a candidate too. Once all candidates are
  // obtained, we pick one that covers all the exact domains (cf. concrete
  // domains in ComputeAtMap)
  IterDomain* findPromotionOfLoopGroup(
      const ValGroup& loop_group,
      const ValGraph& iel_graph,
      const std::unordered_map<ValGroup, IterDomain*>& iel_promotion_map,
      const std::unordered_map<ValGroup, std::shared_ptr<CoveredGroups>>&
          exact_covered_ids,
      const VectorOfUniqueEntries<IterDomain*>& terminal_loop_ids) const;

  // Terminal loop ids are iteration domains in each loop group that:
  // 1) Don't have an entry in p2c_ca_permissive_maps, which would mean a
  //    consumer TV's iter domain maps to this domain in a way that that domain
  //    is also in the same loop group
  // 2) Don't have a direct IterDomain consumer within the group
  VectorOfUniqueEntries<IterDomain*> computeTerminalLoopIds(
      const StatefulInliningInfo& info) const;

  // Given the Step-3 promotion results, returns only promotions of
  // groups that are producers to partially inlined groups. Those
  // partially inlined groups may not have correct promotions as of
  // Step 3 and need another propagation pass.
  std::unordered_map<ValGroup, IterDomain*>
  getProducerPromotionsOfPartiallyInlinedGroups(
      const std::unordered_map<ValGroup, IterDomain*>&
          initial_loop_promotion_map,
      const ValGraph& loop_graph) const;

  // Basic consistency check of the given loop promotion map
  void sanityCheckLoopPromotionMap(
      const std::unordered_map<ValGroup, IterDomain*>& loop_promotion_map)
      const;

  // Revert unnecessary promotions to non-broadcast IDs
  void revertBroadcastOnlyLoopGroups(
      std::unordered_map<ValGroup, IterDomain*>& loop_promotion_map) const;

 private:
  IdModel& id_model_;
  const StatefulInliningInfo& inlining_info_;
  LoopPromotionMapBuilderCallback* callback_ = nullptr;
  // Keep track of IDs of broadcast only loop groups
  std::unordered_set<Val*> broadcast_only_loop_group_ids_;
  // IDs of between logical (root) and loop domains. Loop promotion
  // only matters for these IDs.
  std::unordered_set<Val*> logical_to_loop_ids_;

  // (For debugging only) When force_full_loop_promotion_analysis_ is
  // true, it always performs the full loop promotion analysis even
  // when it's possible to take a quicker shortcut.
  bool force_full_loop_promotion_analysis_ = false;
};

} // namespace nvfuser
