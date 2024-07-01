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
  static std::unordered_map<ValGroup, IterDomain*> get(
      IdModel& id_model,
      const StatefulInliningInfo& inlining_info,
      LoopPromotionMapBuilderCallback* callback = nullptr);

 private:
  LoopPromotionMapBuilder(
      IdModel& id_model,
      const StatefulInliningInfo& inlining_info,
      LoopPromotionMapBuilderCallback* callback = nullptr);

  std::unordered_map<ValGroup, IterDomain*> build();

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
      const std::unordered_map<ValGroup, ValGroups>& exact_covered_ids,
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

 private:
  IdModel& id_model_;
  const StatefulInliningInfo& inlining_info_;
  LoopPromotionMapBuilderCallback* callback_ = nullptr;
};

} // namespace nvfuser
