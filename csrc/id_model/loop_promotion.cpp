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

namespace nvfuser {

LoopPromotionMapBuilder::LoopPromotionMapBuilder(
    IdModel& id_model,
    const StatefulInliningInfo& inlining_info)
    : id_model_(id_model), inlining_info_(inlining_info) {}

void LoopPromotionMapBuilder::build() {
  auto& loop_graph = id_model_.idGraph(IdMappingMode::LOOP);

  std::cerr << nvfuser::idGroupsString(loop_graph);
  std::cerr << "Size: " << inlining_info_.ordered_p_ca_ids.size() << std::endl;
#if 0
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
  ValGraph iel_graph = buildIntersection(
      id_model_.idGraph(IdMappingMode::EXACT), id_model_.idGraph(IdMappingMode::LOOP), false);

  // Step 1: Build a map of the IEL groups of root broadcast domains
  // to resolving domains.
  std::unordered_map<ValGroup, IterDomain*> iel_promotion_map =
      buildInlineRootResolutionMap(iel_graph, inlining_info_);

  // Step 2: Propagate the root promotions to intermediate and leaf groups.
  // At this point, the promotion may not be final as the analysis is
  // localized to IEL groups. The map is used in the next step to
  // build mappings of the loop groups.
  propagatePromotionsInIELGraph(iel_graph, iel_promotion_map);

  // Step 3: Determine the promotion of each loop graph based on the
  // IEL promotion map. For each loop group, examine all the IEL
  // promotions and find the most representative one that captures all
  // the dependent input domains of the loop group
  std::unordered_map<ValGroup, IterDomain*> loop_promotion_map =
      projectIELPromotionToLoopGraph(
          iel_graph,
          iel_promotion_map,
          idGraph(IdMappingMode::LOOP),
          inlining_info);

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
      idGraph(IdMappingMode::LOOP),
      loop_promotion_map);

  // Step 5: Find the final promotion of each loop group based on the
  // final IEL promotion map
  auto final_loop_promotion_map = projectIELPromotionToLoopGraph(
      iel_graph,
      final_iel_promotion_map,
      idGraph(IdMappingMode::LOOP),
      inlining_info);

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

  // Update the Step-3 map to the latest LOOP graph
  loop_promotion_map =
      updateValGroupIdMap(loop_promotion_map, idGraph(IdMappingMode::LOOP));

  // Insert the updated Step-3 results into the Step-5 resutls. Note
  // that this insertion does not overwrite the existing mappings.
  final_loop_promotion_map.insert(
      loop_promotion_map.begin(), loop_promotion_map.end());

  sanityCheckLoopPromotionMap(final_loop_promotion_map);
#endif
}

std::unordered_map<ValGroup, IterDomain*> LoopPromotionMapBuilder::get(
    IdModel& id_model,
    const StatefulInliningInfo& inlining_info) {
  LoopPromotionMapBuilder builder(id_model, inlining_info);
  builder.build();
  return builder.loop_promotion_map_;
}

} // namespace nvfuser
