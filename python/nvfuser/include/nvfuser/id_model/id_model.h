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
class LoopPromotionMapBuilderCallback;

struct StatefulInliningInfo {
  // All producer ids within (including dependencies of) inlined loop domains,
  // used for deterministic order
  VectorOfUniqueEntries<IterDomain*> ordered_p_ca_ids;

  // p2c mappings through the fusion within (including dependencies of) inlined
  // loop domains.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<Val*>>
      p2c_ca_permissive_maps;

  // Broadcast resolution map for root domains, including non-inlined
  // root domains
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      p2c_root_broadcast_resolution_map;

  // All IDs of all first siblings
  VectorOfUniqueEntries<IterDomain*> ordered_sibling_ids;

  // Mappings to other sibling IDs from ordered_sibling_ids
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<Val*>> sibling_maps;
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
// LOOP mode is important to resolve inlined broadcassts. If we have something
// like: consumer[i0o, threadIdx.x{i0i}] = producer[i0o,
// threadIdx.y{i0i}](computeAt = 1) which can easily happen when using shared
// memory. Loop is actually defined for all iteration domains, and resembles
// groups of iter domains that are effectively inlined with each other.
// Therefore iter domain's that are a common dependency of inlined loop domains
// may be loop mapped together.
//
// Loop promotion is a mechanism by which to capture inlined resolved
// broadcasts. If a consumer resolves a broadcast of a producer, and the
// producer's broadcast is inlined (in total or partially). Then the producer's
// iter domain will be "promoted" to the size of the consumers iter domain.
//
// IdMappingMode::EXACT
//   Don't map any broadcast axes to non-broadcast axes
//   Do not forward through any broadcast IDs
// IdMappingMode::BROADCAST
//   Map any broadcast axes to non-broadcast axes
//   Do not forward through any broadcast IDs
// IdMappingMode::PERMISSIVE
//   Forward broadcast axes in replay
//   Map all iteration domains
//   Always contain root mappings (otherwise they could have been forwarded in
//   broadcast)
// IdMappingMode::ALMOSTEXACT
//   Forward through broadcast axes, but not through to a non-broadcast axis
//     i.e. id{b1*i0}, id{i0} are mapped
//          id{i1*i0}, id{i0} are not mapped (this part is the difference from
//          PERMISSIVE)
//   Forward through split one axes, i.e. id{ceilDiv(i0, 1)}, id{i0} are mapped
// IdMappingMode::LOOP
//   Subgraph of the permissive graph. Maps only CA and their
//   dependent domains.
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
      bool build_graphs = false,
      bool allow_self_mapping = true,
      LoopPromotionMapBuilderCallback* loop_promotion_map_builder_callback =
          nullptr);

  // Same as the above constructor with fusion->exprs() excpet fusion may have
  // some dangling inputs/outputs that are expected to have IterDomain entries
  // even though there's no possible connections from them.
  //
  // The validate parameter is a temporary option during the
  // transition from the current ComputeAtMap.
  IdModel(
      Fusion* fusion,
      bool build_graphs = false,
      bool allow_self_mapping = true,
      bool validate = false,
      LoopPromotionMapBuilderCallback* loop_promotion_map_builder_callback =
          nullptr);

  bool hasIdGraph(IdMappingMode mode) const {
    return id_graphs_.find(mode) != id_graphs_.end();
  }

  // Returns iter domain graph of provided mode. The graph must have
  // been already built.
  const ValGraph& idGraph(IdMappingMode mode) const;
  ValGraph& idGraph(IdMappingMode mode);

  const std::unordered_map<IterDomain*, VectorOfUniqueEntries<Expr*>>& idUses()
      const {
    return id_uses_;
  }

  const std::unordered_map<IterDomain*, VectorOfUniqueEntries<Expr*>>&
  idDefinitions() const {
    return id_definitions_;
  }

  // TODO: Seems a bit unfortunate that this isn't IterDomain local information.
  const std::unordered_set<IterDomain*>& viewRfactorIds() const {
    return view_rfactor_ids_;
  }

  std::string toString() const;

  bool empty() const {
    return tvs_.empty();
  }

  const std::vector<TensorView*>& tvs() const {
    return tvs_;
  }

  const std::vector<Expr*>& tvExprs() const {
    return tv_exprs_;
  }

  Fusion* fusion() const {
    return fusion_;
  }

  // Build all graphs, i.e., Exact, AlmostExact, Permissive and
  // LOOP. This is by default called from the constructor
  void buildAllGraphs();

  // Fills disjoint_ids_[IdMappingMode::EXACT] for relationships between inputs
  // and first output of expr
  ValGraph& buildExactGraph();

  // Fills disjoint_ids_[IdMappingMode::ALMOSTEXACT]. Initialize AlmostExact as
  // Exact entries, then map anything that's either merged with a size-1 or
  // split by a size-1 dimension.
  ValGraph& buildAlmostExactGraph();

  // Fills disjoint_ids_[IdMappingMode::BROADCAST]. Initialize it as
  // Exact entries, then map through broadcasts. Build the Exact graph
  // as well if not yet done.
  ValGraph& buildBroadcastGraph();

  // Fills disjoint_ids_[IdMappingMode::PERMISSIVE]. Initialize it as
  // BROADCAST entries, then map through forwarded domains. Build the
  // BROADCAST graph as well if not yet done.
  ValGraph& buildPermissiveGraph();

  // Fills disjoint_ids_[IdMappingMode::LOOP]. Map only inlined
  // domains that are mapped in the permissive graph. Build the Exact
  // and Permissive graphs as well if not yet done.
  //
  // (For debugging only) When force_full_loop_promotion_analysis is
  // true, it always performs the full loop promotion analysis even
  // when it's possible to take a quicker shortcut.
  ValGraph& buildLoopGraph(bool force_full_loop_promotion_analysis = false);

  // Build a graph. Dependent graphs are also built if not yet done.
  ValGraph& buildGraph(IdMappingMode mode);

  // Build a graph if not already built
  ValGraph& maybeBuildGraph(IdMappingMode mode);

  // Remove a graph if already built
  void removeGraph(IdMappingMode mode);

  // Iterates over all IterDomains in id_definitions_ and calls initializeVal on
  // a new ValGraph and returns it.
  ValGraph initializeIdGraph(bool propagate_through_exprs = true) const;

  // Returns an IdGraph with all Id's mapped that are mapped both in graph0 and
  // graph1.
  ValGraph buildIntersection(
      const ValGraph& graph0,
      const ValGraph& graph1,
      bool propagate_exprs = true) const;

  const std::unordered_map<ValGroup, IterDomain*>& loopPromotionMap() const {
    return loop_promotion_map_;
  }

  // Replay Expr but with the inputs provided. ValGraphs will be updated
  // for all maps that have entries, adding the output iter domains of the
  // replayed expression and adding potential mappings through the expression.
  Expr* addReplayAs(std::vector<IterDomain*> new_inputs, Expr* expr);

  //! Run through disjoint sets in the LOOP graph, make sure there's only one
  //! non-serial parallel type in each disjoint set, set the parallel type of
  //! all IterDomains in the disjoint set to that PType.
  void validateAndPropagatePType();

  //! (Copied from ComputeAtMap::allocateIndexVariables)
  //!  Run through disjoint sets in the LOOP map and allocate the index
  //!  variable for the associated for loop that will be generated
  //!  for each disjoint sets in the loop map. This pre-allocation makes
  //!  2 key assumptions about computeAt map that would very likely be
  //!  long term invariant:
  //!    1. All kir::forloop created in the lowering pass should belong
  //!  to one of the disjoint sets in loop map.
  //!    2. The lowering pass will *never* create a loop nest with 2
  //!  different nesting levels mapped together, i.e. the case below
  //!  never occurs:
  //!   for i in IterDomain1
  //!    for j in IterDomain2
  //!     ...
  //!   With loop_map.areMapped(IterDomain1, IterDomain2) == true.
  //! Under this condition, we can pre-allocate all required index
  //!  variable integers before creating any kir::forloop, and this
  //!  would help optimizing the generated integer math for indexing.
  void allocateLoopIndexVariables();

  // Get the index variable assigned for a given loop ID
  Val* getLoopIndexVariable(
      IterDomain* id,
      CircularBufferLoopStage circular_buffer_loop_stage =
          CircularBufferLoopStage::NotApplicable) const;

  // Get the index variable assigned for a given loop group
  Val* getLoopIndexVariable(
      const ValGroup& loop_group,
      CircularBufferLoopStage circular_buffer_loop_stage =
          CircularBufferLoopStage::NotApplicable) const;

 protected:
  // Fills id_uses_ and id_definitions_ for all IterDomains active in the
  // fusion.
  void buildIterDomainDefinitionsAndUses();

  // Start loop map by grouping inlined iter domains
  void initializeLoopGraph(const StatefulInliningInfo& info);

  // Build a map of loop groups to IterDomains that represent actual
  // loops. The map is built based on the broadcast resolution with
  // root domains between inlined producer and consumer tensors.
  std::unordered_map<ValGroup, IterDomain*> buildLoopPromotionMap(
      const StatefulInliningInfo& info);

  // Errors if self mapping occurs
  void assertNoSelfMapping(const ValGraph& graph) const;

  // Loop graph represents the loop structure of the given fusion, so
  // there must not be any mapping between the loop domains of each
  // tensor.
  void validateLoopGraphHasNoSelfMappedLeafDomains() const;

 protected:
  // Fusion where iter domains belong
  Fusion* fusion_ = nullptr;

  // All tensor expressions that this model analyzes
  std::vector<Expr*> tv_exprs_;

  // All tensors that this model analyzes
  std::vector<TensorView*> tvs_;

  // Tensors should not have domains that are mapped with another
  // domains of the same tensor. This flag disables the check
  bool allow_self_mapping_ = false;

  // If true, validate graphs by comparing them with ComputeAtMap
  bool validate_ = false;

  // Optional callback for the loop promotion map builder for
  // debugging and testing
  LoopPromotionMapBuilderCallback* loop_promotion_map_builder_callback_ =
      nullptr;

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

  // Allocated Loop index variable through the LOOP graph
  std::unordered_map<ValGroup, Val*> loop_index_variable_map_;

  // Allocated loop indices for circular buffer loops
  std::unordered_map<
      ValGroup,
      std::unique_ptr<std::unordered_map<CircularBufferLoopStage, Val*>>>
      circular_buffered_loop_index_variable_map_;
};

// A utility function to update a map of ValGroups to ID from an old
// Valgraph to a new ValGraph. The new graph must be a superset of the
// old graph.
std::unordered_map<ValGroup, IterDomain*> updateValGroupIdMap(
    const std::unordered_map<ValGroup, IterDomain*>& stale_map,
    ValGraph& new_graph);

} // namespace nvfuser
