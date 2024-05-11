// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/id_model.h>
#include <id_model/loop_promotion.h>
#include <id_model/to_string.h>
#include <id_model/utils.h>
#include <ir/utils.h>
#include <val_graph_visitor.h>

namespace nvfuser {

LoopPromotionMapBuilder::LoopPromotionMapBuilder(
    IdModel& id_model,
    const StatefulInliningInfo& inlining_info,
    LoopPromotionMapBuilderCallback* callback)
    : id_model_(id_model), inlining_info_(inlining_info), callback_(callback) {}

ValGraph& LoopPromotionMapBuilder::idGraph(IdMappingMode mode) {
  return id_model_.idGraph(mode);
}

const ValGraph& LoopPromotionMapBuilder::idGraph(IdMappingMode mode) const {
  return id_model_.idGraph(mode);
}

std::unordered_map<ValGroup, IterDomain*> LoopPromotionMapBuilder::build() {
  // Make an intersection of the exact and loop map. This will group together
  // entries in each loop group that are exact with each other. This provides a
  // better graph to do promotion and replays.
  //
  // It's tempting to use the intersection of the almost exact and loop, but we
  // need to model broadcast promotion, and if we have two tensors like:
  //
  // T1[i0, b1] = T0[i0]
  // T2[i0, b2] = T0[i0]
  // Then resolution of:
  // T4 = T1[i0, b1] + T3[i0, i1]
  // T6 = T2[i0, b2] + T5[i0, i2]
  //
  // Then merge(0, 1) with all tensors except for T0
  //
  // The almost exact map will map i0, i0*b1, and i0*b2 together, but b1 and b2
  // are being resolved to i1 and i2 respectively. So we want to have separate
  // entries so we can have an easy to process promotion map.
  //
  // Loop is a permissive like map, it could have many entries, use the exact
  // map as the one we iterate on to reduce complexity as it hopefully has
  // smaller groups and this algorithm scales with the number of groups *
  // (number of entries in groups ^ 2)
  //
  // iel stands for Intersection of the Exact and Loop graphs.
  const ValGraph iel_graph = id_model_.buildIntersection(
      idGraph(IdMappingMode::EXACT), idGraph(IdMappingMode::LOOP), false);

  // We'll create mappings from a static loop graph since
  // idGraph(IdMappingMode::LOOP) will change with replayed
  // domains.
  const auto loop_graph = idGraph(IdMappingMode::LOOP);

  // Step 1: Build a map of the IEL groups of root broadcast domains
  // to resolving domains.
  std::unordered_map<ValGroup, IterDomain*> iel_promotion_map =
      buildInlineRootResolutionMap(iel_graph, inlining_info_);

  if (callback_) {
    callback_->postStep1(iel_promotion_map, iel_graph);
  }

  // Step 2: Propagate the root promotions to intermediate and leaf groups.
  // At this point, the promotion may not be final as the analysis is
  // localized to IEL groups. The map is used in the next step to
  // build mappings of the loop groups.
  propagatePromotionsInIELGraph(iel_graph, iel_promotion_map);

  if (callback_) {
    callback_->postStep2(iel_promotion_map, iel_graph);
  }

  // Step 3: Determine the promotion of each loop graph based on the
  // IEL promotion map. For each loop group, examine all the IEL
  // promotions and find the most representative one that captures all
  // the dependent input domains of the loop group
  const std::unordered_map<ValGroup, IterDomain*> initial_loop_promotion_map =
      projectIELPromotionToLoopGraph(
          iel_graph, iel_promotion_map, loop_graph, inlining_info_);

  if (callback_) {
    callback_->postStep3(initial_loop_promotion_map);
  }

  // At this point, most of loop groups should have correct promoted
  // IDs. However, non-inlined loop groups may miss promotion that
  // should be propagated from parent ID groups, e.g., iS50 of T2 in
  // Indexing19. Its parent ID loop group is promoted, but the loop
  // group of iS50 is not found yet.

  // Step 4: In order to fully propagate the loop graph promotions, first
  // propagate them to the IEL groups, which are then used to
  // propagate back to the loop groups in Step 5. Unlike Step 2, the
  // initial IEL promotion map is empty and is populated with the loop
  // promotion map as we traverse down the IEL graph.
  std::unordered_map<ValGroup, IterDomain*> final_iel_promotion_map;
  propagatePromotionsInIELGraph(
      iel_graph,
      final_iel_promotion_map,
      loop_graph,
      initial_loop_promotion_map);

  if (callback_) {
    callback_->postStep4(final_iel_promotion_map, iel_graph);
  }

  // Step 5: Find the final promotion of each loop group based on the
  // final IEL promotion map
  auto final_loop_promotion_map = projectIELPromotionToLoopGraph(
      iel_graph, final_iel_promotion_map, loop_graph, inlining_info_);

  // The promotion map produced in Step 5 only includes those are
  // further propagated at Step 4, so the correct mappings produced at
  // Step 3 may not be included in the Step-5 results. Any Step-3 mappings
  // that are not found in the Step-5 results are already valid
  // results, so merge them into the Step-5 results.
  //
  // For example, in the below case, nothing will be propated at Step
  // 4.
  //
  // t0: [i0]
  // t1: [i1, i2]
  // t2 = broadcast(t0, {true, false})
  // t3 = t2 + t1
  //
  // t2: [b3, i4]
  // t3: [i5, i6]
  //
  // t3->merge(0)
  // propagate-and-inline-most
  //
  // t0: [i0] ca_pos(1)
  // t1: [i1*i2] ca_pos(1)
  // t2: [b3*i4] ca_pos(1)
  // t3: [i5*i6]
  //
  // In this case, all domains will be grouped together and there will
  // be just a single group in the Loop graph:
  //
  // - {i0, i1, i2, b3, i4, i5, i6, i1*i2, b3*i4, i5*i6}
  //
  // Step 3 will identify i5*i6 is the promotion domain. Since all
  // domains are promoted to i5*i6, there will be no propagation in
  // Step 4 (i.e., loop_promote_inputs will be false). Since the
  // result of Step 4 is empty, the Step 5 result will also be empty,
  // but that just means there's no change is necessary from the Step
  // 3 results.

  // Insert the updated Step-3 results into the Step-5 resutls. Note
  // that this insertion does not overwrite the existing mappings.
  final_loop_promotion_map.insert(
      initial_loop_promotion_map.begin(), initial_loop_promotion_map.end());

  // The map is currently for the stale loop graph. Update for the
  // latest loop graph.
  final_loop_promotion_map = updateValGroupIdMap(
      final_loop_promotion_map, idGraph(IdMappingMode::LOOP));

  sanityCheckLoopPromotionMap(final_loop_promotion_map);

  if (callback_) {
    callback_->postStep5(final_loop_promotion_map);
  }

  return final_loop_promotion_map;
}

std::unordered_map<ValGroup, IterDomain*> LoopPromotionMapBuilder::
    buildInlineRootResolutionMap(
        const ValGraph& iel_graph,
        const StatefulInliningInfo& info) const {
  std::unordered_map<ValGroup, IterDomain*> iel_promotion_map;

  // This should probably work just on terminating inputs, as we shouldn't be
  // able to modify a broadcast domain between root and rfactor which would be
  // required to resolve a non input broadcast domain. But for now leaving it as
  // traversal on all broadcast groups.
  //

  // We first visit all broadcast root domains. If a broadcast is
  // resovled, see if it's promoted. Note that a domain be resolved to
  // a domain that may not be loop mapped, yet it can still be
  // promoted. In other words, there can be a domain that is exactly
  // mapped with the resolving domain *and* is mapped with the
  // broadcast domain by the loop map. The algorihm here is:
  //
  // 1. For a broadcast domain, find the domain that the broadcast is
  //    resolved to.
  // 2. If the resolving domain is also loop-mapped with the
  //    broadcast, that is the promotion domain, but the resolving
  //    domain may not be loop mapped as mentioned above. Instead,
  //    find all loop-mapped domains with the broadcast domain and
  //    pick one that is exactly mapped with the resolving domain
  //
  // Note again this process is only done for root domains. Once we
  // find promotion relationships for root domains, we propagate the
  // mappings to derived domains
  for (const ValGroup& iel_group : iel_graph.disjointValSets().disjointSets()) {
    NVF_ERROR(!iel_group->empty());

    IterDomain* iel_group_id = iel_group->front()->as<IterDomain>();

    if (!iel_group_id->isBroadcast()) {
      continue;
    }

    // Collect all the exact groups of the resolutions of the broadcast id's
    ValGroups resolved_exact_groups;
    for (Val* bcast_id : *iel_group) {
      if (auto p2c_root_broadcast_resolution_map_it =
              info.p2c_root_broadcast_resolution_map.find(
                  bcast_id->as<IterDomain>());
          p2c_root_broadcast_resolution_map_it !=
          info.p2c_root_broadcast_resolution_map.end()) {
        resolved_exact_groups.pushBack(
            idGraph(IdMappingMode::EXACT)
                .toGroups(p2c_root_broadcast_resolution_map_it->second));
      }
    }

    if (resolved_exact_groups.empty()) {
      // No resolution
      continue;
    }

    // resolved_exact_groups is a list of IDs that resolves the
    // broadcast. We only care those that are also in the same loop
    // group, and there must be just one or none. Otherwise, the
    // resolution is ambiguous.

    // Collect all the exact groups in the loop set containing this iel_group
    const ValGroup& loop_group =
        idGraph(IdMappingMode::LOOP).toGroup(iel_group_id);
    ValGroups loop_covered_exact_groups =
        idGraph(IdMappingMode::EXACT).toGroups(*loop_group);

    // The intersection of the exact groups that the broadcast domains can be
    // broadcasted to, and those that exist within the same loop groop are is
    // the promotion needed for this iel_group. The promotion should
    // be none or unique.
    ValGroups loop_exact_resolved_intersection =
        resolved_exact_groups.computeIntersect(loop_covered_exact_groups);

    if (loop_exact_resolved_intersection.empty()) {
      // No promotion
      continue;
    }

    if (loop_exact_resolved_intersection.size() > 1) {
      // Ambiguous promotion. This should not happen.
      std::stringstream err_msg;
      err_msg
          << "Invalid multiple broadcast resolution within shared loops detected, group:\n  "
          << iel_group->toString() << "\nIs being broadcasted to:";
      for (const ValGroup& entry : loop_exact_resolved_intersection) {
        err_msg << "\n  " << entry->toString();
      }
      NVF_ERROR(false, err_msg.str());
    }

    const ValGroup& exact_resolution_group =
        loop_exact_resolved_intersection.front();

    // Within the loop group, find the IDs that the broadcast IDs are
    // resolved to
    VectorOfUniqueEntries<Val*> resolved_ids =
        exact_resolution_group->computeIntersect(*loop_group);

    NVF_ERROR(!resolved_ids.empty());

    // All the IDs in resolved_ids are mapped with both of the exact
    // and loop graphs, so any of them can be used as an IEL promotion
    // ID. Just to make it extra clear, look for corresponding
    // groups in the IEL graph and make sure there's only one such group.
    ValGroups promoted_iel_groups = iel_graph.toGroups(resolved_ids);

    NVF_ERROR(!promoted_iel_groups.empty());

    if (promoted_iel_groups.size() > 1) {
      std::stringstream err_msg;
      err_msg
          << "Invalid multiple broadcast resolution within shared loops detected, group:\n  "
          << iel_group->toString() << "\nIs being broadcasted to:";
      for (const ValGroup& entry : promoted_iel_groups) {
        err_msg << "\n  " << entry->toString();
      }
      NVF_ERROR(false, err_msg.str());
    }

    iel_promotion_map[iel_group] =
        promoted_iel_groups.front()->front()->as<IterDomain>();
  }

  return iel_promotion_map;
}

namespace {

// Check if there's an equivalent expression as iel_expr that uses
// maybe_promoted_inputs. This is used to avoid redundantly replaying
// expressions.
// NOTE: This is currently overly conservative and some
// opportunities for reuse are lost, althought it doesn't affect
// the correctness of the analysis.
Expr* findMatchingExpr(
    const ExprGroup& iel_expr,
    const ValGraph& iel_graph,
    const std::vector<IterDomain*>& maybe_promoted_inputs,
    const ValGraph& loop_graph) {
  // If any of domains in maybe_promoted_inputs is not found in
  // iel_graph, it means the domain is just replayed and by definition
  // has no mapping with any existing domain, which means there's no
  // matching expr.
  if (std::any_of(
          maybe_promoted_inputs.begin(),
          maybe_promoted_inputs.end(),
          [&](IterDomain* maybe_promoted_input) -> bool {
            return !iel_graph.hasGroup(maybe_promoted_input);
          })) {
    return nullptr;
  }

  // Grab all eligible uses of the promoted inputs.
  // Note that any eligible matching expr should be a use of all
  // inputs in maybe_promoted_input_uses, no matter it's promoted or
  // not. So it isn't necessary to look at all of
  // maybe_promoted_input_uses but just need to grab one.
  NVF_ERROR(!maybe_promoted_inputs.empty());
  ExprGroups maybe_promoted_input_uses =
      iel_graph.getUses(iel_graph.toGroup(maybe_promoted_inputs.front()));

  if (maybe_promoted_input_uses.empty()) {
    return nullptr;
  }

  // Look for exprs that have inputs that are mapped in the IEL
  // graph with the (promoted) inputs of iel_expr.
  for (const ExprGroup& maybe_promoted_input_use_group :
       maybe_promoted_input_uses) {
    NVF_ERROR(!maybe_promoted_input_use_group->empty());
    // maybe_promoted_inputs may include non-promoted inputs as
    // well, so maybe_promoted_input_uses may include the original
    // iel_expr itself. Since there must at least be a promoted input,
    // iel_expr itself should not be an expr group we are looking for.
    if (iel_expr == maybe_promoted_input_use_group) {
      continue;
    }
    Expr* maybe_promoted_input_use = maybe_promoted_input_use_group->front();
    if (!iel_expr->front()->sameOp(maybe_promoted_input_use)) {
      continue;
    }
    // Check if all inputs are mapped
    NVF_ERROR(
        maybe_promoted_inputs.size() ==
        maybe_promoted_input_use->inputs().size());
    bool all_inputs_match = true;
    for (const auto inp_i : c10::irange(maybe_promoted_inputs.size())) {
      all_inputs_match = all_inputs_match &&
          iel_graph.disjointValSets().strictAreMapped(
              maybe_promoted_inputs[inp_i],
              maybe_promoted_input_use->inputs().at(inp_i));
    }
    if (!all_inputs_match) {
      continue;
    }

    // We always want to find promotions within the same loop
    // groups since we are looking for domains that represent actual
    // loops. Note that that's guaranteed when a new domain is
    // replayed instead of reusing an existing domain.
    if (!loop_graph.disjointExprSets().permissiveAreMapped(
            iel_expr->front(), maybe_promoted_input_use_group->front())) {
      continue;
    }
    // This is just an extra sanity check. Make sure all exprs in
    // the use group are mapped
    NVF_ERROR(
        std::all_of(
            maybe_promoted_input_use_group->vector().begin(),
            maybe_promoted_input_use_group->vector().end(),
            [&](Expr* iel_use) {
              return loop_graph.disjointExprSets().permissiveAreMapped(
                  iel_expr->front(), iel_use);
            }),
        "Not all mapped: ",
        nvfuser::toString(iel_expr),
        "\n",
        nvfuser::toString(maybe_promoted_input_use_group));

    return maybe_promoted_input_use;
  }

  return nullptr;
}

// When propagating loop promotions from inputs to outputs of an IEL
// expr, we can't blindly apply loop promotion when all of the input
// domains are loop mapped with the outputs.
//
// i.e. if we have the inlined domains from:
// Inputs:
//   T0[i0]
//   T1[i0, i1]
//
// T2[i0, b2] = broadcast(T0)
// T3[i0, i1] = T2 + T1
//
// {T1, T2, T3}->merge(0, 1)
// inlineMost
//
// The inlined loop group would consist of:
//
// {i0, i1, b2, i0*b2, i0*i1}
//
// Note that all these domains would have promotion to i0*i1 at the
// end of Step 3. When the IEL expression of merge(i0, i1) is visited by
// propagatePromotionsInIELGraph again, the promotion to i0*i1 of both
// inputs would be propagated to its output, resulting in promotion of
// i0*i1 to (i0*i1)*(i0*i1), which is not the correct propagation.
//
// Therefore only promote i0*b1 to i0*i1, or i0*i1 to i0*i1 (i.e. don't
// promote an input to any transformation within the loop group).
//
// So if we have an iel_expr make sure its inputs and outputs are not in
// the same loop group.
bool hasUniqueInputLoopGroups(
    const ExprGroup& iel_expr,
    const ValGraph& iel_graph,
    const ValGraph& loop_graph) {
  const std::vector<ValGroup> iel_inp_groups = iel_graph.inputGroups(iel_expr);

  const std::vector<ValGroup> iel_out_groups = iel_graph.outputGroups(iel_expr);

  ValGroups inp_loop_groups;
  for (const ValGroup& iel_inp_group : iel_inp_groups) {
    inp_loop_groups.pushBack(loop_graph.toGroup(iel_inp_group->front()));
  }
  ValGroups out_loop_groups;
  for (const ValGroup& iel_out_group : iel_out_groups) {
    out_loop_groups.pushBack(loop_graph.toGroup(iel_out_group->front()));
  }

  // Check if input groups that are not included in the output group set
  return !inp_loop_groups.computeSubtract(out_loop_groups).empty();
}

} // namespace

void LoopPromotionMapBuilder::propagatePromotionsInIELGraph(
    const ValGraph& iel_graph,
    std::unordered_map<ValGroup, IterDomain*>& iel_promotion_map,
    const ValGraph& loop_graph,
    const std::unordered_map<ValGroup, IterDomain*>& loop_graph_promotion_map) {
  // In order to make this traversal work, the traversal order must be
  // topologically sorted.
  ValGraphStmtSort iel_stmt_sort(iel_graph);

  for (const ExprGroup& iel_expr : iel_stmt_sort.exprs()) {
    NVF_ERROR(!iel_expr->empty());
    const std::vector<ValGroup> iel_inp_groups =
        iel_graph.inputGroups(iel_expr);

    // Check if any inputs need promotion indicating this expr group needs to
    // be replayed with promoted inputs
    bool an_input_was_promoted = false;
    std::vector<IterDomain*> maybe_promoted_inputs;
    maybe_promoted_inputs.reserve(iel_inp_groups.size());

    // Propagate loop graph promotion only when the inputs and outputs are
    // not in the same loop group.
    const bool loop_promote_inputs = !loop_graph_promotion_map.empty() &&
        hasUniqueInputLoopGroups(iel_expr, iel_graph, loop_graph);

    // FusionReshapePersistentShmoo with
    // std::vector<int64_t> x{3, 17, 2 * 4 * 10, 1};
    // std::vector<int64_t> y{3 * 17, 1, 2, 4, -1};
    // Incorrect loop promotion at the second segment. Need to use the
    // step 3 result when propagating through IEL expr of merge 104
    // and 197.
    bool war = iel_expr->front()->isA<Split>() &&
        iel_expr->front()->input(0)->definition() &&
        iel_expr->front()->input(0)->definition()->isA<Merge>();

    for (const ValGroup& iel_inp_group : iel_inp_groups) {
      // Assumed all inputs are IterDomains
      NVF_ERROR(iel_inp_group->front()->isA<IterDomain>());

      if (war) {
        // This is a copy of the loop_promote_inputs block below. When
        // war is true, this block should be prioritized.

        // Promote loops based on the loop promotion map. If the loop promotion
        // map should be used and has an entry we should use that promotion.
        if (loop_promote_inputs) {
          const ValGroup& loop_copy_group =
              loop_graph.toGroup(iel_inp_group->front());
          auto inp_loop_promo_it =
              loop_graph_promotion_map.find(loop_copy_group);
          if (inp_loop_promo_it != loop_graph_promotion_map.end()) {
            maybe_promoted_inputs.push_back(inp_loop_promo_it->second);
            an_input_was_promoted = true;
            VERBOSE() << "Promoted input by loop promotion: "
                      << nvfuser::toString(iel_inp_group) << " -> "
                      << inp_loop_promo_it->second->name() << std::endl;
            continue;
          }
        }
      }

      // Propagate IEL promotions when available.
      if (auto inp_promo_it = iel_promotion_map.find(iel_inp_group);
          inp_promo_it != iel_promotion_map.end()) {
        maybe_promoted_inputs.push_back(inp_promo_it->second);
        an_input_was_promoted = true;
        continue;
      }

      // Promote loops based on the loop promotion map. If the loop promotion
      // map should be used and has an entry we should use that promotion.
      if (loop_promote_inputs) {
        const ValGroup& loop_copy_group =
            loop_graph.toGroup(iel_inp_group->front());
        auto inp_loop_promo_it = loop_graph_promotion_map.find(loop_copy_group);
        if (inp_loop_promo_it != loop_graph_promotion_map.end()) {
          maybe_promoted_inputs.push_back(inp_loop_promo_it->second);
          an_input_was_promoted = true;
          continue;
        }
      }

      // No promotion found. Just use the non-promoted domain
      maybe_promoted_inputs.push_back(iel_inp_group->front()->as<IterDomain>());
    }

    if (!an_input_was_promoted) {
      // No inputs need promotion so just continue
      continue;
    }

    Expr* promoted_expr = findMatchingExpr(
        iel_expr,
        iel_graph,
        maybe_promoted_inputs,
        idGraph(IdMappingMode::LOOP));

    bool replayed = false;

    if (!promoted_expr) {
      promoted_expr =
          id_model_.addReplayAs(maybe_promoted_inputs, iel_expr->front());
      replayed = true;
    }

    // Mark outputs as having a promoted iter domain
    std::vector<ValGroup> out_groups = iel_graph.outputGroups(iel_expr);
    NVF_ERROR(promoted_expr->outputs().size() == out_groups.size());
    NVF_ERROR(
        ir_utils::filterByType<IterDomain>(promoted_expr->outputs()).size() ==
            out_groups.size(),
        "Unexpected non IterDomain outputs found: ",
        promoted_expr->toString());

    for (const auto i : c10::irange(out_groups.size())) {
      // Promote if necessary, if the output is already in the same exact map
      // it doesn't need a promotion.
      if (idGraph(IdMappingMode::EXACT)
              .disjointValSets()
              .strictAreMapped(
                  promoted_expr->output(i), out_groups[i]->front())) {
        continue;
      }
      iel_promotion_map[out_groups[i]] =
          promoted_expr->output(i)->as<IterDomain>();
      // Explicitly map loop map since expr propagation doesn't happen
      if (replayed) {
        idGraph(IdMappingMode::LOOP)
            .mapVals(iel_expr->front()->output(i), promoted_expr->output(i));
      }
    }
  }
}

void LoopPromotionMapBuilder::propagatePromotionsInIELGraph(
    const ValGraph& iel_graph,
    std::unordered_map<ValGroup, IterDomain*>& iel_promotion_map) {
  propagatePromotionsInIELGraph(
      iel_graph, iel_promotion_map, idGraph(IdMappingMode::LOOP), {});
}

namespace {

// Returns for each ValGroup in provided IdGraph what the input ValGroups are
// traversing on definitions. Ignoring broadcast ValGroups and resetting inputs
// at RFactor ValGroups.
std::unordered_map<ValGroup, ValGroups> computeCoveredGroups(
    const ValGraph& graph,
    const std::unordered_set<IterDomain*>& view_rfactor_ids) {
  // Map from an exact iter domain group, to all the exact iter domain groups it
  // covers
  std::unordered_map<ValGroup, ValGroups> covered_ids;

  for (const ValGroup& id_group : graph.disjointValSets().disjointSets()) {
    // Initialize inputs
    const ExprGroups& id_group_defs = graph.getDefinitions(id_group);
    if (id_group_defs.empty()) {
      covered_ids[id_group] = {id_group};
    }

    // Initialize rfactor groups
    if (std::any_of(id_group->begin(), id_group->end(), [&](Val* id) {
          return view_rfactor_ids.find(id->as<IterDomain>()) !=
              view_rfactor_ids.end();
        })) {
      covered_ids[id_group] = {id_group};
    }

    // Initialize broadcast groups to empty since broadcast domains
    // don't matter for indexing
    if (std::any_of(id_group->begin(), id_group->end(), [&](Val* id) {
          return id->as<IterDomain>()->isBroadcast();
        })) {
      covered_ids[id_group] = {};
    }
  }

  ValGraphStmtSort exact_stmt_sort(graph);

  for (const ExprGroup& exact_expr : exact_stmt_sort.exprs()) {
    std::vector<ValGroup> input_groups = graph.inputGroups(exact_expr);

    ValGroups covered;
    for (const ValGroup& inp_group : input_groups) {
      covered.pushBack(covered_ids.at(inp_group));
    }

    for (const ValGroup& output_group : graph.outputGroups(exact_expr)) {
      // Don't overwrite initialized cases due to rfactor markings.
      if (covered_ids.find(output_group) == covered_ids.end()) {
        covered_ids[output_group] = covered;
      } else {
        if (!getenv("DISABLE_COMBINE")) {
          VERBOSE() << "Combining coverage of "
                    << nvfuser::toString(output_group) << ". "
                    << nvfuser::toString(covered_ids[output_group]);
          covered_ids[output_group].pushBack(covered);
        } else {
          VERBOSE() << "Avoid overwriting covered group of "
                    << nvfuser::toString(output_group) << ". Existing: "
                    << nvfuser::toString(covered_ids[output_group])
                    << ". New: " << nvfuser::toString(covered)
                    << ", expr: " << exact_expr->front()->toString()
                    << std::endl;
        }
      }
    }
  }

  return covered_ids;
}

}; // namespace

std::unordered_map<ValGroup, IterDomain*> LoopPromotionMapBuilder::
    projectIELPromotionToLoopGraph(
        const ValGraph& iel_graph,
        const std::unordered_map<ValGroup, IterDomain*>& iel_promotion_map,
        const ValGraph& loop_graph,
        const StatefulInliningInfo& inlining_info) const {
  const std::unordered_map<ValGroup, ValGroups> exact_covered_ids =
      computeCoveredGroups(
          idGraph(IdMappingMode::EXACT), id_model_.viewRfactorIds());

  // Grab terminal iter domain in the loop groups.
  const VectorOfUniqueEntries<IterDomain*> terminal_loop_ids =
      computeTerminalLoopIds(inlining_info);

  std::unordered_map<ValGroup, IterDomain*> loop_promotion_map;

  for (const ValGroup& loop_group :
       loop_graph.disjointValSets().disjointSets()) {
    IterDomain* promotion_id = findPromotionOfLoopGroup(
        loop_group,
        iel_graph,
        iel_promotion_map,
        exact_covered_ids,
        terminal_loop_ids);
    if (promotion_id) {
      loop_promotion_map[loop_group] = promotion_id;
    }
  }

  return loop_promotion_map;
}

IterDomain* LoopPromotionMapBuilder::findPromotionOfLoopGroup(
    const ValGroup& loop_group,
    const ValGraph& iel_graph,
    const std::unordered_map<ValGroup, IterDomain*>& iel_promotion_map,
    const std::unordered_map<ValGroup, ValGroups>& exact_covered_ids,
    const VectorOfUniqueEntries<IterDomain*>& terminal_loop_ids) const {
  const ValGraph& exact_graph = idGraph(IdMappingMode::EXACT);

  // Grab all the (potentially promoted) terminal iter domains in this group.
  // Save the exact group and the iter domain in this vector.
  std::vector<std::pair<ValGroup, IterDomain*>> exact_promoted_terminal_ids;
  for (auto loop_group_val : *loop_group) {
    auto loop_id = loop_group_val->as<IterDomain>();

    // If not a terminal id in the group skip
    if (!terminal_loop_ids.has(loop_id)) {
      continue;
    }

    // Grab the iel entry. There can be iter domains that were added
    // after the IEL graph was built. All the promotion information is
    // associated with the domains that exist in the original graph,
    // so the new domains can be simply ignored.
    if (!iel_graph.hasGroup(loop_id)) {
      continue;
    }

    // TODO: Terminal rfactor domains are definitely a promotion domain
    if (id_model_.viewRfactorIds().find(loop_id) !=
        id_model_.viewRfactorIds().end()) {
      VERBOSE() << "Terminal rfactor id found: " << loop_id->toString()
                << std::endl;
      return loop_id;
    }

    const ValGroup& iel_group = iel_graph.toGroup(loop_id);

    // Does it still need iel_promotion_map? The loop group already has
    // the replayed domains, so we should be able to find it.
    auto iel_promo_it = iel_promotion_map.find(iel_group);
    if (iel_promo_it == iel_promotion_map.end()) {
      // If this terminal ID doesn't have a promotion associated with it, save
      // the terminal ID.
      exact_promoted_terminal_ids.emplace_back(
          exact_graph.toGroup(loop_id), loop_id->as<IterDomain>());
    } else {
      // If this terminal ID has a promotion, grab the promoted ID.
      exact_promoted_terminal_ids.emplace_back(
          exact_graph.toGroup(iel_promo_it->second), iel_promo_it->second);
    }
  }

  // All the exact groups of the iter domains in the loop group
  ValGroups exact_groups = exact_graph.toGroups(*loop_group);

  // All exact groups covered by all iter domains in this loop group
  ValGroups loop_group_covered_ids;
  for (const ValGroup& exact_group : exact_groups) {
    auto covered_it = exact_covered_ids.find(exact_group);
    NVF_ERROR(covered_it != exact_covered_ids.end());
    loop_group_covered_ids.pushBack(covered_it->second);
  }

  // Check if any of the candidate Iter Domains we collected cover all the
  // exact groups of loop_group_covered_ids. If so, that's the correct
  // promoted iter domain of this group.
  for (const auto& entry : exact_promoted_terminal_ids) {
    const ValGroup& terminal_id_group = entry.first;
    IterDomain* terminal_id = entry.second;
    auto covered_it = exact_covered_ids.find(terminal_id_group);
    NVF_ERROR(covered_it != exact_covered_ids.end());
    if (loop_group_covered_ids.computeSubtract(covered_it->second).empty()) {
      return terminal_id;
    }
  }

  return nullptr;
}

VectorOfUniqueEntries<IterDomain*> LoopPromotionMapBuilder::
    computeTerminalLoopIds(const StatefulInliningInfo& info) const {
  VectorOfUniqueEntries<IterDomain*> terminal_loop_ids;
  for (const ValGroup& group :
       idGraph(IdMappingMode::LOOP).disjointValSets().disjointSets()) {
    if (group->size() == 1) {
      terminal_loop_ids.pushBack(group->front()->as<IterDomain>());
    }

    // Don't select producer iter domains
    for (auto loop_id : *group) {
      if (info.p2c_ca_permissive_maps.find(loop_id->as<IterDomain>()) !=
          info.p2c_ca_permissive_maps.end()) {
        continue;
      }

      // It's terminal if there's no use group
      auto uses_it = id_model_.idUses().find(loop_id->as<IterDomain>());
      if (uses_it == id_model_.idUses().end() || uses_it->second.empty()) {
        terminal_loop_ids.pushBack(loop_id->as<IterDomain>());
        continue;
      }

      // If there's an output group that is not in the same group,
      // then it's a terminal ID
      bool all_outs_in_loop_group = true;
      for (auto use : uses_it->second) {
        if (std::any_of(
                use->outputs().begin(),
                use->outputs().end(),
                [&](Val* out) -> bool {
                  return group != idGraph(IdMappingMode::LOOP).toGroup(out);
                })) {
          all_outs_in_loop_group = false;
          break;
        }
      }

      if (!all_outs_in_loop_group) {
        terminal_loop_ids.pushBack(loop_id->as<IterDomain>());
      }
    }
  }
  return terminal_loop_ids;
}

void LoopPromotionMapBuilder::sanityCheckLoopPromotionMap(
    const std::unordered_map<ValGroup, IterDomain*>& loop_promotion_map) const {
  const auto& loop_graph = idGraph(IdMappingMode::LOOP);
  for (const ValGroup& loop_group :
       loop_graph.disjointValSets().disjointSets()) {
    // Non-leaf loop groups are not guaranteed to have valid
    // promotions. See for example FusionRepro1713, where root domains
    // are all grouped together but there's no valid promotion.
    if (loop_graph.hasUses(loop_group)) {
      continue;
    }
    // Make sure the loop group is promoted to a domain that is mapped
    // in the LOOP graph
    auto promotion_it = loop_promotion_map.find(loop_group);
    NVF_ERROR(
        promotion_it != loop_promotion_map.end(),
        "Loop promotion not found for ",
        nvfuser::toString(loop_group));
    IterDomain* promotion = promotion_it->second;
    // Make sure the promotion domain is also loop-mapped
    NVF_ERROR(
        loop_group->has(promotion),
        "Loop promotion not loop-mapped. Loop group: ",
        nvfuser::toString(loop_group),
        ". Promotion domain: ",
        promotion->name());
  }
}

std::unordered_map<ValGroup, IterDomain*> LoopPromotionMapBuilder::get(
    IdModel& id_model,
    const StatefulInliningInfo& inlining_info,
    LoopPromotionMapBuilderCallback* callback) {
  LoopPromotionMapBuilder builder(id_model, inlining_info, callback);
  return builder.build();
}

} // namespace nvfuser
