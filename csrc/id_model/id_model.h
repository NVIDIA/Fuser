// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <disjoint_set.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <val_graph.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nvfuser {

class ValGraph;

struct StatefulInliningInfo {
  // All producer ids within (including dependencies of) inlined leaf domains,
  // used for deterministic order
  VectorOfUniqueEntries<IterDomain*> ordered_p_ca_ids;

  // p2c mappings through the fusion within (including dependencies of) inlined
  // leaf domains.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<Val*>>
      p2c_ca_permissive_maps;

  // Broadcast resolution map for root domains, including non-inlined
  // root domains
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      p2c_root_broadcast_resolution_map;
};

StatefulInliningInfo buildStatefulInliningInfo(
    const std::vector<Expr*>& exprs,
    const ValGraph& exact_graph,
    const ValGraph& permissive_graph);

// A collection of ValGraphs that are built from a fusion or series of
// expressions. These graphs are related, but have some distinct features based
// on the IdMappingMode.
//
// EXACT/PERMISSIVE mode:
//
// consumer[i0, b1] = producer[i0]
// consumer->merge(0) (consumer will now be [i0 * b1])
//
// When producer is replayed as consumer (the direction we use for mapping)
// with forwarding from ForwardingInfo the producer to consumer map will have
// both a mapping of consumer(i0) to producer(i0) as well as consumer(i0*b1) to
// producer(i0). This latter mapping is important for loop nest mappings as the
// consumer will generate a loop based on i0*b1 and the producer may be
// computeAt inside this loop nest. However, for indexing we do not want these
// two iter domains mapped as producer may be indexed as i0*i1 depending on the
// loop nest structure and how it was built.
//
// Exact mode is if the iter domain relationships from producer to consumer are
// considered the exact same size operating on matching dimensions from the root
// domain mapping.
//
// IdMappingMode::EXACT
//   Don't map any broadcast axes to non-broadcast axes
//   Do not forward through any broadcast IDs
// IdMappingMode::PERMISSIVE
//   Forward broadcast axes in replay
//   Map all iteration domains
//   Always contain root mappings (otherwise they could have been forwarded in
//   broadcast)
// IdMappingMode::AlmostExact
//   Forward through broadcast axes, but not through to a non-broadcast axis
//     i.e. id{b1*i0}, id{i0} are mapped
//          id{i1*i0}, id{i0} are not mapped (this part is the difference from
//          PERMISSIVE)
//   Forward through split one axes, i.e. id{ceilDiv(i0, 1)}, id{i0} are mapped
// IdMappingMode::LOOP
//   Subgraph of the permissive graph. Maps only CA and their
//   dependent domains
//
class IdModel : public PolymorphicBase {
 public:
  // Sometimes fusion inputs or outputs are disconnected from expressions, in
  // those cases we still may want to send in some additional tensor views from
  // the Fusion that don't have expressions associated with them.
  //
  // All graphs are built by default. It can be disabled with
  // build_graphs=false.
  IdModel(
      const std::vector<Expr*>& exprs,
      const std::vector<TensorView*>& additional_tvs = {},
      bool build_graphs = true,
      bool allow_self_mapping = false);

  // Same as the above constructor with fusion->exprs() excpet fusion may have
  // some dangling inputs/outputs that are expected to have IterDomain entries
  // even though there's no possible connections from them.
  //
  // The validate parameter is a temporary option during the
  // transition from the current ComputeAtMap.
  IdModel(
      Fusion* fusion,
      bool build_graphs = true,
      bool allow_self_mapping = false,
      bool validate = true);

  // Returns iter domain graph of provided mode. The graph must have
  // been already built.
  const ValGraph& idGraph(IdMappingMode mode) const;
  ValGraph& idGraph(IdMappingMode mode);

  // TODO: Seems a bit unfortunate that this isn't IterDomain local information.
  const std::unordered_set<IterDomain*>& viewRfactorIds() const {
    return view_rfactor_ids_;
  }

  std::string toString() const;

  // Build all graphs, i.e., Exact, AlmostExact, Permissive and
  // LOOP. This is by default called from the constructor
  void buildAllGraphs();

  // Fills disjoint_ids_[IdMappingMode::EXACT] for relationships between inputs
  // and first output of expr
  void buildExactGraph();

  // Fills disjoint_ids_[IdMappingMode::ALMOSTEXACT]. Initialize AlmostExact as
  // Exact entries, then map anything that's either merged with a size-1 or
  // split by a size-1 dimension.
  void buildAlmostExactGraph();

  // Fills disjoint_ids_[IdMappingMode::PERMISSIVE]. Initialize it as
  // Exact entries, then map through broadcasts. Build the Exact graph
  // as well if not yet done.
  void buildPermissiveGraph();

  // Fills disjoint_ids_[IdMappingMode::LOOP]. Map only inlined
  // domains that are mapped in the permissive graph. Build the Exact
  // and Permissive graphs as well if not yet done.
  void buildLoopGraph();

  // Build a graph. Dependent graphs are also built if not yet done.
  void buildGraph(IdMappingMode mode);

  // Build a graph if not already built
  void maybeBuildGraph(IdMappingMode mode);

  // Iterates over all IterDomains in id_definitions_ and calls initializeVal on
  // a new ValGraph and returns it.
  ValGraph initializeIdGraph(bool propagate_through_exprs = true);

  // Returns an IdGraph with all Id's mapped that are mapped both in graph0 and
  // graph1.
  ValGraph buildIntersection(
      const ValGraph& graph0,
      const ValGraph& graph1,
      bool propagate_exprs = true);

  const std::unordered_map<ValGroup, IterDomain*>& loopPromotionMap() const {
    return loop_promotion_map_;
  }

 protected:
  // Fills id_uses_ and id_definitions_ for all IterDomains active in the
  // fusion.
  void buildIterDomainDefinitionsAndUses();

  /// Start loop map by grouping inlined iter domains
  void initializeLoopGraph(const StatefulInliningInfo& info);

  // Build a map of loop groups to IterDomains that represent actual
  // loops. The map is built based on the broadcast resolution with
  // root domains between inlined producer and consumer tensors.
  std::unordered_map<ValGroup, IterDomain*> buildLoopPromotionMap(
      const StatefulInliningInfo& info);

  // Helper function for buildLoopPromotionMap. Returns a map of
  // root broadcast ValGroups in the IEL graph to a representative
  // IterDomain picked from its IEL group.
  std::unordered_map<ValGroup, IterDomain*> buildInlineRootResolutionMap(
      const ValGraph& iel_graph,
      const StatefulInliningInfo& info);

  // Helper function for building loop promotion map.
  //
  // Propagate promotion mappings from root IEL groups to intermediate
  // and leaf IEL groups by traversing IEL exprs. For each expr, if an
  // input is promoted, the output needs to be promoted too. If
  // there's already an equivalent expr that uses the promoted inputs,
  // create a mapping from the outputs of the IEL expr to the outputs
  // of the equivalent expr.
  void propagatePromotionsInIELGraph(
      const ValGraph& iel_graph,
      std::unordered_map<ValGroup, IterDomain*>& iel_promotion_map);

  // Errors if self mapping occurs
  void assertNoSelfMapping();

  // Replay Expr but with the inputs provided. ValGraphs will be updated
  // for all maps that have entries, adding the output iter domains of the
  // replayed expression and adding potential mappings through the expression.
  Expr* addReplayAs(std::vector<IterDomain*> new_inputs, Expr* expr);

 protected:
  // All tensor expressions that this model analyzes
  std::vector<Expr*> tv_exprs_;

  // All tensors that this model analyzes
  std::vector<TensorView*> tvs_;

  // Tensors should not have domains that are mapped with another
  // domains of the same tensor. This flag disables the check
  bool allow_self_mapping_ = false;

  // If true, validate graphs by comparing them with ComputeAtMap
  bool validate_ = false;

  // By default, the permissive graph should map compliment domains as
  // well. See the design doc for more details
  bool permissive_graph_map_compliment_ids_ = true;

  // Keeps ValGraphs containing all IterDomains for all mapping mode types.
  //
  // Using an array here might be nice, but it seems hard to use an enum as an
  // array key
  // https://stackoverflow.com/questions/2102582/how-can-i-count-the-items-in-an-enum
  std::unordered_map<IdMappingMode, ValGraph> id_graphs_;

  // If multiple transformations occur IterDomains could have multiple uses,
  // however only one should be active in the given Fusion. When we resolve loop
  // promotions during lowering, we can generate new iter domains from existing
  // ones, so there can be multiple uses generated. Tracks all the active iter
  // domain uses.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<Expr*>> id_uses_;

  // Make sure we don't blindly use definitions as we don't want to grab
  // transformations before a tensor view's root domain. There can be
  // multiple definitions due to replays.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<Expr*>> id_definitions_;

  std::unordered_set<IterDomain*> view_rfactor_ids_;

  // Promotion domain for each loop group
  std::unordered_map<ValGroup, IterDomain*> loop_promotion_map_;
};

} // namespace nvfuser
