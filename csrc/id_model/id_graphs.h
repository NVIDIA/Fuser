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
#include <id_model/id_graph.h>
#include <ir/all_nodes.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nvfuser {

class IdGraph;

namespace {
// Convenience to store some intermediate data across a few lowering build
// passes.
struct StatefulLoweringInfo;
} // namespace

// A collection of IterDomainGraphs that are built from a fusion or series of
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
// groups of iter domains that are effectively inlined with eachother. Therefore
// iter domain's that are a common dependency of inlined leaf domains may be
// loop mapped together. This map is developed in lowering from
// bulidInlinePromotions and buildLoopPromotionMap.
//
// Loop promotion is a mechanism by which to capture inlined resolved
// broadcasts. If a consumer resolves a broadcast of a producer, and the
// producer's broadcast is inlined (in total or partially). Then the producer's
// iter domain will be "promoted" to the size of the consumers iter domain.
//
// IdMappingMode::LOOP
//   Forward broadcast axes in replay
//   Denotes groups of IterDomains that are considered promoted to a common iter
//   domain size
// IdMappingMode::PERMISSIVE
//   Forward broadcast axes in replay
//   Map all iteration domains
//   Always contain root mappings (otherwise they could have been forwarded in
//   broadcast)
// IdMappingMode::EXACT
//   Don't map any broadcast axes to non-broadcast axes
//   Do not forward through any broadcast IDs
// IdMappingMode::AlmostExact
//   Forward through broadcast axes, but not through to a non-broadcast axis
//     i.e. id{b1*i0}, id{i0} are mapped
//          id{i1*i0}, id{i0} are not mapped (this part is the difference from
//          PERMISSIVE)
//   Forward through split one axes, i.e. id{ceilDiv(i0, 1)}, id{i0} are mapped
//
class IterDomainGraphs : public PolymorphicBase {
 public:
  IterDomainGraphs(
      const std::vector<Expr*>& exprs,
      const std::vector<TensorView*>& additional_tvs,
      bool allow_self_mapping = false);

  IterDomainGraphs(
      const std::vector<Expr*>& exprs,
      bool allow_self_mapping = false);

  // Same as the above constructor with fusion->exprs() excpet fusion may have
  // some dangling inputs/outputs that are expected to have IterDomain entries
  // even though there's no possible connections from them.
  IterDomainGraphs(Fusion* fusion, bool allow_self_mapping = false);

  // Returns iter domain graph of provided mode.
  const IdGraph& idGraph(IdMappingMode mode) const;
  IdGraph& idGraph(IdMappingMode mode);

  // IterDomains from the original fusion are only allowed to be used once in
  // the IterDomain graph, id->uses() are not directly used as there's no bounds
  // check that would prevent a use from being defined that's not part of the
  // actual fusion definition.
  //
  // Note, any iter domains used during something like loop or concrete id
  // resolution could actually have multiple Expr* uses, and uses on disjoint id
  // sets should be used, not this.
  //
  // TODO: Refactor or remove?
  Expr* idUse(IterDomain* id) const;
  Expr* idDef(IterDomain* id) const;

  // TODO: Seems a bit unfortunate that this isn't IterDomain local information.
  const std::unordered_set<IterDomain*>& viewRfactorIds() const {
    return view_rfactor_ids_;
  }

  // Returns if a self mapping was detected that would invalidate assumptions of
  // the overall lowering system.
  //
  // TODO: Can we make this more of an alias analysis?
  // Ref: https://github.com/csarofeen/pytorch/pull/1954#discussion_r961940498
  bool hasSelfMapping() const {
    return self_mapping_info_.has_value();
  }

  // Update the LOOP ID disjoint sets with resolved computeWith
  void updateComputeWith(TensorView* compute_with_tv);

  std::string toString() const;

  // Replay Expr but with the inputs provided. IterDomainGraphss will be updated
  // for all maps that have entries, adding the output iter domains of the
  // replayed expression and adding potential mappings through the expression.
  Expr* addReplayAs(std::vector<IterDomain*> new_inputs, Expr* expr);

  // Similar to addReplayAs, but clones the expr exactly instead of replaying it
  // forward. It's up to the calling code to make sure the replacements are
  // valid for the provided expr. It's generally recommended that the
  // IterDomains exactly match those in the expr.
  //
  // "forward" dictates the same argument for mapThroughExpr. If forward the
  // function will apply mapThroughExpr forward if inputs map in each
  // initialized map. Else does the same but backwards through the expression
  // from outputs.
  //
  // TODO-NM: Unused?
  Expr* addExprWithReplacement(
      const std::unordered_map<IterDomain*, IterDomain*>& old_2_new_ids,
      Expr* old_expr);

  // Make a new expr matching that provided but using the outputs provided.
  // IterDomainGraphss will be updated for all maps that have entries. Adding
  // the input iter domains of the replayed expression and adding potential
  // mappings through the expressions. Input domains will match exactly in all
  // properties as those in expr. This is unlike addReplayAs which will produce
  // new outputs using transformations directly.
  //
  // TODO-NM: Not implemented?
  Expr* addBackwardsReplayAs(
      const std::vector<IterDomain*>& new_outputs,
      Expr* expr);

  // Make an exact copy of provided IterDomain (without rfactor set), and map
  // the copy to the original in all registered IdGraphs. IterDomain copy will
  // not have any registered uses or definitions.
  //
  // TODO-NM: Unused?
  IterDomain* cloneIterDomain(IterDomain* id);

  const std::unordered_map<IdGroup, IterDomain*> loopPromotionMap() const {
    return loop_promotion_map_;
  }

  // TODO: Should this not be private?
 protected:
  // Sometimes fusion inputs or outputs are disconnected from expressions, in
  // those cases we still may want to send in some additional tensor views from
  // the Fusion that don't have expressions associated with them.
  void build(
      const std::vector<Expr*>& exprs,
      const std::vector<TensorView*>& additional_tvs);

  // ======= START Iteration domain build process in order called =======

  // Fills id_uses_ and id_definitions_ for all IterDomains active in the
  // fusion.
  void buildIterDomainDefinitionsAndUses(
      const std::vector<TensorView*>& all_tvs);

  // Iterates over all IterDomains in id_definitions_ and calls initializeID on
  // a new IdGraph and returns it.
  IdGraph initializeIdGraph(bool propagate_through_exprs = true);

  // Fills disjoint_ids_[IdMappingMode::EXACT] for relationships between inputs
  // and first output of expr
  void buildExactMap(const std::vector<Expr*>& exprs);

  // Fills disjoint_ids_[IdMappingMode::ALMOSTEXACT]. Initialize AlmostExact as
  // Exact entries, then map anything that's either merged with a size-1 or
  // split by a size-1 dimension.
  void buildAlmostExactMap();

  // Fills disjoint_ids_[IdMappingMode::PERMISSIVE]. Initialize PermissiveMap as
  // AlmostExact entries, then map through broadcasts
  void buildPermissiveMap(const std::vector<Expr*>& exprs);

  // Make sure only leaf nodes of tensor views are parallelized
  void validatePTypes(const std::vector<TensorView*>& all_tvs) const;

  //! Run through disjoint sets in the LOOP map, make sure there's only one
  //! non-serial parallel type in each disjoint set, set the parallel type of
  //! all IterDomains in the disjoint set to that PType.
  //
  // TODO-NM: Unused
  void propagateLoopPTypes() const;

  // !! START Helper functions to build loop promotion and index map!!

  // Terminal loop ids are iteration domains in each loop group that:
  // 1) Don't have an entry in p2c_ca_permissive_maps, which would mean a
  //    consumer TV's iter domain maps to this domain in a way that that domain
  //    is also in the same loop group
  // 2) Don't have a direct IterDomain consumer within the group
  VectorOfUniqueEntries<IterDomain*> computeTerminalLoopIds(
      const StatefulLoweringInfo info);

  // Returns an IdGraph with all Id's mapped that are mapped both in graph0 and
  // graph1.
  IdGraph buildIntersection(
      const IdGraph& graph0,
      const IdGraph& graph1,
      bool propagate_exprs = true);

  // !! END Helper functions to build loop promotion and index map!!

  // Start loop map by grouping inlined iter domains
  void initializeLoopMap(StatefulLoweringInfo& info);

  // Returns map of IdGroups in the loop map to a representative IterDomain that
  // contains all resolved transformations that the terminal IterDomains should
  // be promoted to. The returned promotions are valid only for inlined iter
  // domains.
  std::unordered_map<IdGroup, IterDomain*> buildInlinePromotions(
      StatefulLoweringInfo& info);

  // Returns a similar thing to buildInlinePromotions but also includes iter
  // domains that are not inlined.
  std::unordered_map<IdGroup, IterDomain*> buildLoopPromotionMap(
      const std::vector<Expr*>& exprs,
      StatefulLoweringInfo& info,
      const std::unordered_map<IdGroup, IterDomain*>& stale_promotion_map);

  // Builds idGraph(IdMappingMode::INDEX) and returns the iter domain promotion
  // map to go from leaf domains of each (consumer only?) tensor to their
  // corresponding leaf domain in the index graph.
  std::unordered_map<IterDomain*, IterDomain*> buildIndexGraph(
      const std::vector<Expr*>& exprs,
      const std::vector<TensorView*>& all_tvs,
      StatefulLoweringInfo& info,
      std::unordered_map<IdGroup, IterDomain*> stale_promotion_map);

  // Returns the terminal rfactor or input iter domains each group in the almost
  // exact map covers (in the almost exact map). This effectively returns all
  // the input almost exact iter domain groups for each almost exact iter domain
  // group. RFactor axes are considered an "input" as all broadcast dimensions
  // have to be resolved by or before the rfactor iter domain.
  std::unordered_map<IdGroup, IdGroups> buildCoveredAlmostExact();

  // ======= END Iteration domain build process in order called =======

  // Errors if self mapping occurs
  void assertNoSelfMapping();

  // Keeps a disjoint set entry for all IterDomain for all mapping mode types.
  //
  // Using an array here might be nice, but it seems hard to use an enum as an
  // array key
  // https://stackoverflow.com/questions/2102582/how-can-i-count-the-items-in-an-enum
  std::unordered_map<IdMappingMode, IdGraph> id_graphs_;

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

  // Debug information to hold if a self mapping in a TensorView is found.
  c10::optional<std::tuple<TensorView*, IterDomain*, IterDomain*, std::string>>
      self_mapping_info_ = c10::nullopt;

  // Promotion domain for each loop group
  std::unordered_map<IdGroup, IterDomain*> loop_promotion_map_;

  std::unordered_set<IterDomain*> view_rfactor_ids_;
};

} // namespace nvfuser
