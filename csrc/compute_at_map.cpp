// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <compute_at_map.h>

#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <transform_iter.h>

#include <tuple>

namespace nvfuser {
namespace {

// Is the provided IterDomain an Leaf of provided TensorView and within its
// computeAtPosition.
// If outside computeAt axis, we don't want to directly map consumer/producer in
// the loop mapping as they are not sharing the same loop.
bool idIsAComputeAtLeafDomain(
    IterDomain* id,
    TensorView* producer_tv,
    TensorView* consumer_tv) {
  auto begin = producer_tv->getLoopDomain().begin();
  auto end = producer_tv->getLoopDomain().begin() +
      producer_tv->getComputePosition(consumer_tv);
  return std::find(begin, end, id) != end;
}

// Is the provided IterDomain an Leaf of provided TensorView
bool idIsALeafDomain(IterDomain* id, TensorView* tv) {
  auto begin = tv->getLoopDomain().begin();
  auto end = tv->getLoopDomain().end();
  return std::find(begin, end, id) != end;
}

} // namespace

IterDomainGraph::IterDomainGraph(Fusion* fusion, bool allow_self_mapping) {
  build(fusion);

  if (!allow_self_mapping) {
    NVF_ERROR(
        !hasSelfMapping(),
        "Unsupported domain mapping detected in ",
        std::get<0>(*self_mapping_info_)->toString(),
        ". ",
        std::get<3>(*self_mapping_info_),
        " domains, ",
        std::get<1>(*self_mapping_info_)->toString(),
        " and ",
        std::get<2>(*self_mapping_info_)->toString(),
        ", are mapped with each other.");
  }
}

//! Map corresponding inputs and outputs of swizzle op together
//!  on the given disjoint set, if the given id is an output
//!  of a swizzle operator.
//!
//! The current usage of swizzle operator is local to each tensor
//!  itself, so they should not affect exact or permissive mapping
//!  between iterdomains on different tensor domains.
//! TODO:
//!   Exact mapping based index hoisting of swizzled iterdomains
//!   is disabled currently and will be re-enabled in the next
//!   few build out steps.
void mapMaybeSwizzleOp(
    DisjointSets<IterDomain*>& disjoint_sets,
    IterDomain* id) {
  if (auto swizzle_2d = dynamic_cast<Swizzle2D*>(id->definition())) {
    // Map each input to its corresponding output on the given
    // disjoint set if this is a loop swizzle. Loop swizzles don't impact
    // indexing, only iteration order.
    if (swizzle_2d->swizzleMode() == SwizzleMode::Loop) {
      disjoint_sets.mapEntries(swizzle_2d->inX(), swizzle_2d->outX());
      disjoint_sets.mapEntries(swizzle_2d->inY(), swizzle_2d->outY());
    }
  }
}

bool IterDomainGraph::exprsMap(
    Expr* first,
    Expr* second,
    bool forward,
    const DisjointSets<IterDomain*>& id_map) {
  if (first == nullptr || second == nullptr) {
    return false;
  }

  if (typeid(*first) != typeid(*second)) {
    return false;
  }

  NVF_ERROR(
      first->isA<Merge>() || first->isA<Split>() || first->isA<Resize>(),
      "Merge, split and resize are the only expressions supported through root to logical operations in compute at map, but found:\n",
      first->toString());

  auto first_ids = ir_utils::filterByType<IterDomain>(
                       forward ? first->inputs() : first->outputs())
                       .vector();

  auto second_ids = ir_utils::filterByType<IterDomain>(
                        forward ? second->inputs() : second->outputs())
                        .vector();

  NVF_ERROR(
      first_ids.size() == second_ids.size(),
      "Expected number of ",
      (forward ? "inputs" : "outputs"),
      " to match for\n",
      first->toString(),
      second->toString());

  {
    std::vector<std::pair<IterDomain*, IterDomain*>> zipped_ids;

    std::transform(
        first_ids.begin(),
        first_ids.end(),
        second_ids.begin(),
        std::back_inserter(zipped_ids),
        [](IterDomain* first, IterDomain* second) {
          return std::make_pair(first, second);
        });

    if (std::any_of(
            zipped_ids.begin(),
            zipped_ids.end(),
            [&](std::pair<IterDomain*, IterDomain*> id_pair) {
              return !id_map.strictAreMapped(id_pair.first, id_pair.second);
            })) {
      return false;
    }
  }

  if (first->isA<Merge>() && !forward) {
    // Can't back prop through merge without making sure one dimension actually
    // is identical extents.
    auto merge0 = first->as<Merge>();
    auto merge1 = second->as<Merge>();

    auto extent_0o = merge0->outer()->extent();
    auto extent_0i = merge0->inner()->extent();
    auto extent_1o = merge1->outer()->extent();
    auto extent_1i = merge1->inner()->extent();

    auto extent_0_match = extent_0o->sameAs(extent_1o) ||
        (extent_0o->isConstInt() && extent_1o->isConstInt() &&
         extent_0o->evaluate().as<int64_t>() ==
             extent_1o->evaluate().as<int64_t>());

    auto extent_1_match = extent_0i->sameAs(extent_1i) ||
        (extent_0i->isConstInt() && extent_1i->isConstInt() &&
         extent_0i->evaluate().as<int64_t>() ==
             extent_1i->evaluate().as<int64_t>());

    if (!(extent_0_match || extent_1_match)) {
      return false;
    }
  }

  if (first->isA<Split>()) {
    auto first_split = first->as<Split>();
    auto second_split = second->as<Split>();
    if (!first_split->factor()->sameAs(second_split->factor()) ||
        first_split->innerSplit() != second_split->innerSplit()) {
      return false;
    }
  }

  if (first->isA<Resize>()) {
    auto first_resize = first->as<Resize>();
    auto second_resize = second->as<Resize>();
    if (!first_resize->leftExpand()->sameAs(second_resize->leftExpand()) ||
        !first_resize->rightExpand()->sameAs(second_resize->rightExpand())) {
      return false;
    }
  }

  return true;
}

// Given first and second Exprs "match"
//   Expr type matches
//   IterDomain's in the inputs and outputs exact match, (including argument
//     position positions)
//   Paramters like Split's factor "match" (exact match on integers could be
//     better, as today it will just check it's the same symbol or evaluated to
//     the same constant. However, we know all the extents of all the
//     IterDomain's that exact map with eachother are the same value.
void IterDomainGraph::mapThroughExpr(Expr* first, Expr* second, bool forward) {
  if (first == nullptr || second == nullptr) {
    return;
  }

  if (!exprsMap(first, second, forward, exact_nodes_)) {
    return;
  }

  auto first_ids = ir_utils::filterByType<IterDomain>(
                       forward ? first->outputs() : first->inputs())
                       .vector();
  auto second_ids = ir_utils::filterByType<IterDomain>(
                        forward ? second->outputs() : second->inputs())
                        .vector();
  NVF_ERROR(
      first_ids.size() == second_ids.size(),
      "This should be unreachable, if transformation expressions match, their number of inputs and outputs should as well.\n However found:\n",
      first->toString(),
      "\nand\n",
      second->toString());
  for (auto out_i : arange(first_ids.size())) {
    exact_nodes_.mapEntries(first_ids[out_i], second_ids[out_i]);
    permissive_nodes_.mapEntries(first_ids[out_i], second_ids[out_i]);
    permissive_resize_nodes_.mapEntries(first_ids[out_i], second_ids[out_i]);
  }
}

namespace {

// Returns the first pair of id's in ids detected to match eachother on the
// permissive map of the ID graph. TODO: what this is really looking for is if
// there's any overlapping between the iter domains in the provided set.
//
// i.e. if we have:
// tv0 = arange(6).view({3, 2})
// tv1 = tv0[3, 2].t()
// tv2 = tv0[3, 2].view({2, 3})
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
// The elements in tv1 {3, 1, 4, 2}, map respectively to the elements in tv2 {1,
// 2, 3, 4}. The reason this is so important is it means that generating tv3 is
// no longer a trivially parallelizable problem (if we include the dag all the
// way to tv0). So tv0's axes cannot be inlined across both the tv0 and tv1
// path. This breaks some assumptions we have today in schedulers that will
// assume tv2 can be trivially inlined/parallelized. Instead we'd need to take
// into consideration the effective communication going on here, so that we pull
// multiple values of tv0 to compute tv3.
std::optional<std::pair<IterDomain*, IterDomain*>> detectMappablePair(
    const std::vector<IterDomain*>& ids,
    const IterDomainGraph& id_graph,
    IdMappingMode mode) {
  for (auto id1 : ids) {
    for (auto id2 : ids) {
      if (id1 == id2) {
        continue;
      }
      if (mode == IdMappingMode::EXACT) {
        if (id_graph.exactNodes().disjointSetMap().at(id1)->has(id2)) {
          return std::make_pair(id1, id2);
        }
      } else if (mode == IdMappingMode::PERMISSIVE) {
        if (id_graph.permissiveNodes().disjointSetMap().at(id1)->has(id2)) {
          return std::make_pair(id1, id2);
        }
      } else if (mode == IdMappingMode::LOOP) {
        if (id_graph.loopNodes().disjointSetMap().at(id1)->has(id2)) {
          return std::make_pair(id1, id2);
        }
      } else {
        NVF_THROW("Unrecognized IdMappingMode mode.");
      }
    }
  }

  return {};
}

// It is assumed that for any tensor represented by a list of domains,
// those domains should never be mapped with each other. It may be
// possible to lift this assumption, but it's unclear if it could
// matter in practice.
std::optional<std::tuple<TensorView*, IterDomain*, IterDomain*, std::string>>
findFirstSelfMapping(Fusion* fusion, const IterDomainGraph& id_graph) {
  for (auto tv : fusion->allTvs()) {
    // For each tensor, make sure root, logical and loop domains
    // should not include domains that are mapped with another domain
    // in the same set of domains. This may be overly conservative,
    // and it maybe enough to check the root domains.

    // logical domains
    auto self_mappped_root_pair = detectMappablePair(
        tv->getLogicalDomain(), id_graph, IdMappingMode::EXACT);
    if (self_mappped_root_pair.has_value()) {
      return std::make_tuple(
          tv,
          self_mappped_root_pair->first,
          self_mappped_root_pair->second,
          "Logical");
    }

    // root domains
    if (tv->hasRoot()) {
      auto self_mappped_rf_pair = detectMappablePair(
          tv->getRootDomain(), id_graph, IdMappingMode::EXACT);
      if (self_mappped_rf_pair.has_value()) {
        return std::make_tuple(
            tv,
            self_mappped_rf_pair->first,
            self_mappped_rf_pair->second,
            "Root");
      }
    }

    // Leaf domains
    auto self_mappped_loop_pair =
        detectMappablePair(tv->getLoopDomain(), id_graph, IdMappingMode::LOOP);
    if (self_mappped_loop_pair.has_value()) {
      return std::make_tuple(
          tv,
          self_mappped_loop_pair->first,
          self_mappped_loop_pair->second,
          "Leaf");
    }
  }
  return std::nullopt;
}

} // namespace

void IterDomainGraph::build(Fusion* fusion) {
  FusionGuard fg(fusion);

  // Initialize a node for every iteration domain
  for (auto tv : fusion->allTvs()) {
    const auto& domain = tv->getLoopDomain();
    auto all_ids = tv->domain()->allIDs();

    for (auto id : all_ids) {
      // Check if this id is an logical id in the logical domain
      bool is_rfactor_domain_id = id->isRFactorProduct() &&
          std::find(
              tv->getLogicalDomain().begin(),
              tv->getLogicalDomain().end(),
              id) != tv->getLogicalDomain().end();
      bool is_loop_id =
          std::find(domain.begin(), domain.end(), id) != domain.end();
      initializeId(id, is_rfactor_domain_id, is_loop_id);
    }
  }

  // All ID's are initialized, start connecting them on the permissive, exact,
  // and loop dimensions.

  for (auto expr : fusion->exprs()) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }

    auto tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());
    TensorView* first_output_tv = nullptr;

    for (auto c_tv : tv_outputs) {
      if (ir_utils::hasUniformSiblings(expr)) {
        if (first_output_tv == nullptr) {
          first_output_tv = c_tv;
        } else {
          // Map multi outputs of an expression to each other. c is current
          // output, and f as first output. Keep consistent with the later
          // section of producer and consumers. Which here producer is now
          // "first output", and consumer is still consumer. One exception is
          // how the domains left of CA positions are handled in the Parallel
          // map. Those domains are not mapped in producer and consumer
          // mappings as they do not share loops, but are mapped in the
          // case of mapping multiple outputs since they do share the
          // same loops.

          NVF_ERROR(
              c_tv->getMaybeRootDomain().size() ==
                  first_output_tv->getMaybeRootDomain().size(),
              "Multiple outputs with mismatched dimensions is not supported. ",
              "Only supported case is welford op where all outputs tvs have identical domains.");
          // p->f, c->c
          std::unordered_map<IterDomain*, IterDomain*> c2f_root_map;
          for (const auto i :
               arange(first_output_tv->getMaybeRootDomain().size())) {
            c2f_root_map.insert(std::make_pair(
                c_tv->getMaybeRootDomain()[i],
                first_output_tv->getMaybeRootDomain()[i]));
          }

          // Multi output mapping, outputs are required to have the same domain
          // and same transformations, so they can be mapped in
          // permissive/exact, and when within compute at position of
          // getLoopDomain() in the parallel map.
          auto replay_FasC = BestEffortReplay(
              first_output_tv->getLoopDomain(),
              c_tv->getLoopDomain(),
              c2f_root_map);

          // Map the entire replay map between the multiple
          // consumers
          auto c2f_disjoint_sets = replay_FasC.getIterDomainEquivalence();
          for (const auto& disjoint_set : c2f_disjoint_sets.disjointSets()) {
            if (disjoint_set->empty()) {
              continue;
            }
            auto id0 = *disjoint_set->begin();
            for (auto id1 : disjoint_set->vector()) {
              permissive_nodes_.mapEntries(id0, id1);
              permissive_resize_nodes_.mapEntries(id0, id1);
              exact_nodes_.mapEntries(id0, id1);
              sibling_sets_.mapEntries(id0, id1);
            }
          }

          // Map all entries for the Loop map as they share the same loops.
          for (auto f_id : first_output_tv->getLoopDomain()) {
            auto disjoint_set = c2f_disjoint_sets.getDisjointSetOf(f_id);
            auto id0 = *(disjoint_set.begin());
            for (auto id1 : disjoint_set) {
              loop_nodes_.mapEntries(id0, id1);
            }
          }
        }
      }

      auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());

      for (auto p_tv : tv_inputs) {
        auto pairwise_map = PairwiseLogicalDomainMap(p_tv, c_tv);

        // Look for matching ID transformations in producer and consumer, replay
        // producer as consumer. We use the symmetric API of BestEffortReplay so
        // that both broadcast and squeeze are handled correctly.
        //
        // Note on the boolean flags: swizzles are skipped in both
        // producer and consumer but resizes are not.
        const auto permissive_disjoint_sets =
            BestEffortReplay::replayPasC(
                p_tv, c_tv, -1, pairwise_map, true, true, false)
                .getIterDomainEquivalence();

        // Permissive-Resize map allows mappings of resize inputs and
        // outputs as well as those indirectly accessed domains by
        // gather-scatter like ops
        //
        // TODO: clean this up. Maybe this can be just the PERMISSIVE
        // map? Revisit after the ID map refactor.
        //
        // Note on the boolean flags: swizzles and resizes are skipped
        // in the permissive-resize map
        const auto pairwise_resize_map =
            PairwiseLogicalDomainMap(p_tv, c_tv).mapIndexedDomains(true);
        const auto permissive_resize_disjoint_sets =
            BestEffortReplay::replayPasC(
                p_tv, c_tv, -1, pairwise_resize_map, true, true, true)
                .getIterDomainEquivalence();

        // For exact mapings do not map any broadcast dimensions to
        // non-broadcast dimensions. Prevent any broadcasted axes being mapped
        // to non-broadcasted axes.
        auto exact_c2p_logical_map = PairwiseLogicalDomainMap(p_tv, c_tv)
                                         .mapBroadcast(false)
                                         .mapConsumerToProducer();

        // Same as permissive above but for exact
        auto exact_replay_PasC = BestEffortReplay(
            p_tv->getLoopDomain(),
            c_tv->getLoopDomain(),
            exact_c2p_logical_map);

        const auto& exact_c2p_map = exact_replay_PasC.getReplay();

        for (auto c_id : getSortedKeys(exact_c2p_map, Statement::lessThan)) {
          auto p_id = exact_c2p_map.at(c_id);
          exact_nodes_.mapEntries(c_id, p_id);
          consumers_.at(p_id).pushBack(c_id);
          producers_.at(c_id).pushBack(p_id);

          // Add the swizzle inputs to the same
          //  disjoint set as well if either c_id
          //  or p_id is swizzle output.
          mapMaybeSwizzleOp(exact_nodes_, p_id);
          mapMaybeSwizzleOp(exact_nodes_, c_id);
        }

        auto p_ids_vec = p_tv->domain()->allIDs();
        auto c_ids_vec = c_tv->domain()->allIDs();
        std::unordered_set<IterDomain*> p_ids(
            p_ids_vec.begin(), p_ids_vec.end());
        std::unordered_set<IterDomain*> c_ids(
            c_ids_vec.begin(), c_ids_vec.end());

        for (auto& dset : permissive_disjoint_sets.disjointSets()) {
          auto& vec = dset->vector();
          for (auto i : arange(vec.size())) {
            auto id1 = vec[i];
            permissive_nodes_.mapEntries(id1, vec[0]);

            // Add the swizzle inputs to the same
            //  disjoint set as well if either c_id
            //  or p_id is swizzle output.
            mapMaybeSwizzleOp(permissive_nodes_, id1);

            for (auto j : arange(i + 1, vec.size())) {
              auto id2 = vec[j];
              if (p_ids.count(id1) && c_ids.count(id2)) {
                if (idIsAComputeAtLeafDomain(id1, p_tv, c_tv) &&
                    idIsALeafDomain(id2, c_tv)) {
                  loop_nodes_.mapEntries(id1, id2);
                }
              }
              if (c_ids.count(id1) && p_ids.count(id2)) {
                if (idIsAComputeAtLeafDomain(id2, p_tv, c_tv) &&
                    idIsALeafDomain(id1, c_tv)) {
                  loop_nodes_.mapEntries(id1, id2);
                }
              }
            }
          }
        }

        // Mostly the same as the above for the permissive map but
        // nothing to do for the loop map.
        // The producer and consumer maps are based on the most
        // permissive mappings, so they are set using the
        // permissive-resize mappings.
        for (auto& dset : permissive_resize_disjoint_sets.disjointSets()) {
          auto& vec = dset->vector();
          for (auto i : arange(vec.size())) {
            auto id1 = vec[i];
            permissive_resize_nodes_.mapEntries(id1, vec[0]);
            mapMaybeSwizzleOp(permissive_resize_nodes_, id1);
            for (auto j : arange(i + 1, vec.size())) {
              auto id2 = vec[j];
              if (p_ids.count(id1) && c_ids.count(id2)) {
                consumers_.at(id1).pushBack(id2);
                producers_.at(id2).pushBack(id1);
              }
              if (c_ids.count(id1) && p_ids.count(id2)) {
                producers_.at(id1).pushBack(id2);
                consumers_.at(id2).pushBack(id1);
              }
            }
          }
        }
      }
    }
  }

  // Explicitly map through root to logical transformations, if we have an op
  // like:
  //
  // T1[x, y*z] = view(T0[x*y, z])
  // T3[x, y*z] = view(T2[x*y, z])
  // T4 = T0 + T2
  //
  // We want to map T1 and T3's root to logical transformations together by
  // playing the transformations forward since their root domains map. If
  // instead we have:
  //
  // T1[x, y*z] = view(T0[x*y, z])
  // T3[x, y*z] = view(T2[x*y, z])
  // T4 = T1 + T3
  //
  // Then we wouldn't have a mapping of T1 and T3's root domain, we'd have a
  // mapping of their logical domain, so we would want to map T1 and T3's
  // root to logical transformations starting at their logical domains.
  //
  // Therefore we'll explicitly map root to logical transformation iteration
  // domains forward and backwards. Something similar could happen with root of
  // logical domains, though it seems mapping rfactor reduction domains aren't
  // that important. Mapping view transformations is more important since view
  // is part of the compute definition so having the map through the
  // transformations makes it easy to check if different view operations are
  // consistent with eachother.

  auto all_tvs = fusion->allTvs();
  std::vector<TensorView*> all_consumer_tvs;
  std::copy_if(
      all_tvs.begin(),
      all_tvs.end(),
      std::back_inserter(all_consumer_tvs),
      [](TensorView* tv) { return !tv->isFusionInput() && tv->hasRoot(); });

  // IterDomains could have multiple uses defined in the fusion if multiple
  // transformations were redefined (more than one transform propagation pass
  // was run and retransformed sections of the graph). We're going to make a new
  // uses map so we can easily process the actual uses of IterDomains. We
  // actually only need logical uses for this section of mapping, so we'll limit
  // this map to only root to logical transformations.
  std::unordered_map<IterDomain*, Expr*> logical_id_uses;

  // Order of traversal is important for processing all the logical ids as the
  // first pass will go forward through expressions and the second pass will
  // traverse backwards through them. ID's will be unique in this vector,
  // enforced when building it since it's built with logical_id_uses.
  std::vector<IterDomain*> logical_id_order;

  // Grab all the logical ids.
  for (auto consumer_tv : all_consumer_tvs) {
    auto exprs = StmtSort::getExprsBetween(
        {consumer_tv->getMaybeRootDomain().begin(),
         consumer_tv->getMaybeRootDomain().end()},
        {consumer_tv->getLogicalDomain().begin(),
         consumer_tv->getLogicalDomain().end()});
    for (auto expr : exprs) {
      auto logical_inp_ids = ir_utils::filterByType<IterDomain>(expr->inputs());
      NVF_ERROR(
          expr->isA<Split>() || expr->isA<Merge>() || expr->isA<Resize>() ||
              expr->isA<Swizzle>(),
          "Wasn't expecting the expression type of:\n",
          expr->toString(),
          "\nto be an expression defined in an root to logical transformation.");
      for (auto logical_inp_id : logical_inp_ids) {
        NVF_ERROR(
            logical_id_uses.find(logical_inp_id) == logical_id_uses.end(),
            "Was expecting iter domains to only have one active transformation but found id ",
            logical_inp_id->toString(),
            " used in\n",
            logical_id_uses.at(logical_inp_id),
            "\nand\n",
            expr->toString());
        logical_id_uses.emplace(logical_inp_id, expr);
        logical_id_order.push_back(logical_inp_id);
      }
    }
    for (auto logical_id : consumer_tv->getLogicalDomain()) {
      if (logical_id->isRFactorProduct()) {
        logical_id_uses.emplace(logical_id, nullptr);
        logical_id_order.push_back(logical_id);
      }
    }
  }

  // if prop_forward we're going forward through transformations and
  // expressions, meaning if inputs of expressions map then we map their
  // outputs, otherwise we're traversing backwards, meaning if outputs of
  // expressions map then we map their inputs.
  for (auto prop_forward : {true, false}) {
    std::unordered_set<Expr*> visited_exprs;

    for (auto logical_id_i : arange(logical_id_order.size())) {
      auto first_logical_id = prop_forward
          ? logical_id_order[logical_id_i]
          : logical_id_order[logical_id_order.size() - 1 - logical_id_i];

      // At should be safe since we made logical_id_order and logical_id_uses at
      // the same time so they should have the same exact entries.
      auto first_expr = prop_forward ? logical_id_uses.at(first_logical_id)
                                     : first_logical_id->definition();

      if (first_expr == nullptr) {
        continue;
      }

      // logical_id_uses are guaranteed to be a valid expr, but
      // first_logical_id->definition() may not be part of the valid
      // exprs
      if (!prop_forward) {
        if (std::any_of(
                first_expr->inputs().begin(),
                first_expr->inputs().end(),
                [&](Val* id_input) {
                  return !all_ids_.has(id_input->as<IterDomain>());
                })) {
          continue;
        }
      }

      if (visited_exprs.find(first_expr) != visited_exprs.end()) {
        continue;
      }
      visited_exprs.emplace(first_expr);

      // Only need to be concerned here with mapping across root iter
      // domains, so isolate out those.
      auto all_exact_map_ids = exact_nodes_.getDisjointSetOf(first_logical_id);
      std::vector<IterDomain*> exact_map_rf_ids;
      std::copy_if(
          all_exact_map_ids.vector().begin(),
          all_exact_map_ids.vector().end(),
          std::back_inserter(exact_map_rf_ids),
          [](IterDomain* id) { return id->isRFactorProduct(); });

      for (auto exact_map_rf_id : exact_map_rf_ids) {
        if (exact_map_rf_id == first_logical_id) {
          continue;
        }
        // If there's an input with an logical domain we could have an exact
        // mapped logical id that's on the input meaning it wouldn't have an
        // entry in logical_id_uses
        auto other_use =
            logical_id_uses.find(exact_map_rf_id) == logical_id_uses.end()
            ? nullptr
            : logical_id_uses.at(exact_map_rf_id);
        auto other_expr =
            prop_forward ? other_use : exact_map_rf_id->definition();

        if (other_expr == nullptr) {
          continue;
        }

        if (visited_exprs.find(other_expr) != visited_exprs.end()) {
          continue;
        }

        mapThroughExpr(first_expr, other_expr, prop_forward);
      }
    }
  }

  // Adds more mappings from IdModel if available
  auto expand_by_id_model = [](DisjointSets<IterDomain*>& nodes,
                               IdMappingMode mode) {
    if (!GpuLower::hasCurrent() || !GpuLower::current()->hasIdModel()) {
      return;
    }

    const ValGraph& graph = GpuLower::current()->idModel().idGraph(mode);
    for (const auto& vg : graph.disjointValSets().disjointSets()) {
      IterDomain* first_id = nullptr;
      for (const auto& val : *vg) {
        auto id = val->as<IterDomain>();
        if (!nodes.mappingExists(id)) {
          continue;
        }
        if (first_id == nullptr) {
          first_id = id;
        } else if (!nodes.strictAreMapped(first_id, id)) {
          nodes.mapEntries(first_id, id);
        }
      }
    }
  };

  // Expand the exact sets with the IdModel exact graph so that
  // the legacy and new indexers would produce less mismatching
  // results.
  expand_by_id_model(exact_nodes_, IdMappingMode::EXACT);
  // Expand the permissive sets with the IdModel exact graph. The
  // permissive IdModel graph may be used instead, but the exact graph
  // seems sufficient to fill the gap with IdModel
  expand_by_id_model(permissive_nodes_, IdMappingMode::EXACT);

  innermost_nodes_ = permissive_resize_nodes_;
  // Build almost exact map by forwarding through broadcast axes
  almost_exact_nodes_ = exact_nodes_;
  std::unordered_set<Expr*> visited;
  auto all_elements = exact_nodes_.getAllElements();
  for (auto entry : all_elements.vector()) {
    if (entry->definition() == nullptr) {
      continue;
    }
    auto def = entry->definition();
    if (!visited.emplace(def).second) {
      continue;
    }

    // If there's an input that is not included in the map, this expr
    // should not be considered
    if (std::ranges::any_of(def->inputs(), [&](Val* inp) {
          return !allIds().has(inp->as<IterDomain>());
        })) {
      continue;
    }

    if (auto merge = dynamic_cast<Merge*>(def)) {
      if (merge->inner()->extent()->isOneInt()) {
        almost_exact_nodes_.mapEntries(merge->outer(), merge->out());
        innermost_nodes_.mapEntries(merge->outer(), merge->out());
      } else {
        // maps to inner dimension, even though it's not an identical mapping.
        // This is used for transpose scheduler to map inner loop dimensions
        innermost_nodes_.mapEntries(merge->inner(), merge->out());
      }
      if (merge->outer()->extent()->isOneInt()) {
        almost_exact_nodes_.mapEntries(merge->inner(), merge->out());
      }
    } else if (auto split = dynamic_cast<Split*>(def)) {
      if (split->factor()->isOneInt()) {
        if (split->innerSplit()) {
          almost_exact_nodes_.mapEntries(split->in(), split->outer());
        } else {
          almost_exact_nodes_.mapEntries(split->in(), split->inner());
        }
      }
      if (split->factor()->isOneInt() && split->innerSplit()) {
        innermost_nodes_.mapEntries(split->in(), split->outer());
      } else {
        // maps to inner dimension, even though it's not an identical mapping.
        // This is used for transpose scheduler to map inner loop dimensions
        innermost_nodes_.mapEntries(split->in(), split->inner());
      }
    }
  }

  expand_by_id_model(almost_exact_nodes_, IdMappingMode::ALMOSTEXACT);

  self_mapping_info_ = findFirstSelfMapping(fusion, *this);
}

void IterDomainGraph::initializeId(
    IterDomain* id,
    bool is_rfactor_id,
    bool is_loop_id) {
  permissive_nodes_.initializeSet(id);
  permissive_resize_nodes_.initializeSet(id);
  exact_nodes_.initializeSet(id);
  if (is_loop_id) {
    loop_nodes_.initializeSet(id);
  }
  consumers_[id] = {};
  producers_[id] = {};
  sibling_sets_.initializeSet(id);

  all_ids_.pushBack(id);

  if (is_rfactor_id) {
    rfactor_ids_.emplace(id);
  }
}

ComputeAtMap::ComputeAtMap(Fusion* fusion, bool allow_self_mapping)
    : id_graph_(fusion, allow_self_mapping),
      concretized_bcasts_(fusion),
      fusion_(fusion) {
  build(fusion);
}

void ComputeAtMap::build(Fusion* fusion) {
  buildUniqueExactExprMaps();
  buildConcreteIds();
  buildUniqueExactExprMaps();
}

void ComputeAtMap::validateAndPropagatePType() {
  for (const auto& loop_disjoint_set : id_graph_.loopNodes().disjointSets()) {
    ParallelType common_ptype = ParallelType::Serial;
    for (auto id : loop_disjoint_set->vector()) {
      auto id_ptype = id->getParallelType();
      NVF_ERROR(
          id_ptype == common_ptype || id_ptype == ParallelType::Serial ||
              common_ptype == ParallelType::Serial,
          "Issue validating parallel type disjoint ptype is, ",
          common_ptype,
          " but found in the set the id: ",
          id->toString());
      common_ptype =
          common_ptype == ParallelType::Serial ? id_ptype : common_ptype;
    }

    for (auto id : loop_disjoint_set->vector()) {
      id->parallelize(common_ptype);
    }
  }
}

void ComputeAtMap::allocateIndexVariables() {
  // Run through all disjoint sets registered in loop map,
  //  all lowered ForLoop will correspond to one of the disjoint sets
  //  and we only need one index variable for each set.
  // All domains parallelized by computation warp groups share the same index
  // variable. This occurs because their loops are merged into a single loop
  // during the circular buffer pass, separating data loading for different warp
  // groups.
  Val* computation_warp_group_index = nullptr;
  for (const auto& loop_disjoint_set : id_graph_.loopNodes().disjointSets()) {
    ParallelType ptype = ParallelType::Serial;

    // We don't allocate any index variable for domains which
    // are parallelized accross devices
    if (auto result = std::find_if(
            loop_disjoint_set->vector().begin(),
            loop_disjoint_set->vector().end(),
            [](IterDomain* id) { return id->isDeviceDim(); });
        result != loop_disjoint_set->vector().end()) {
      loop_index_variable_map_[loop_disjoint_set.get()] = fusion_->zeroVal();
      continue;
    }

    // first allocate thread and grid parallel indices:
    //  The validation pass will check that the parallel bindings within the
    //  loop nodes are consistent so all the loops within this disjoint set
    //  will be realized implicitly using parallel index variables.

    if (auto result = std::find_if(
            loop_disjoint_set->vector().begin(),
            loop_disjoint_set->vector().end(),
            [](IterDomain* id) { return id->isThread(); });
        result != loop_disjoint_set->vector().end()) {
      ptype = (*result)->getParallelType();
      loop_index_variable_map_[loop_disjoint_set.get()] =
          NamedScalar::getParallelIndex(ptype);
      continue;
    }

    // All loops in this set are non-parallel, non-concretized broadcast
    //  iterdomains, their "index variable" should be zero.
    if (std::all_of(
            loop_disjoint_set->vector().begin(),
            loop_disjoint_set->vector().end(),
            [](IterDomain* id) { return id->isBroadcast(); })) {
      loop_index_variable_map_[loop_disjoint_set.get()] = fusion_->zeroVal();
      continue;
    }

    // Allocate variable for the iterdomains:
    auto concrete_loop_id_it = concrete_id_cache_.find(loop_disjoint_set);
    NVF_ERROR(
        concrete_loop_id_it != concrete_id_cache_.end(),
        "Concrete id not computed");

    auto concrete_loop_id = concrete_loop_id_it->second;

    // Need to allocate circular buffered loop differently.
    if (GpuLower::current()->circularBufferInfo().isCircularBufferedIterDomain(
            concrete_loop_id)) {
      // Allocate index variable for each stage of the circular buffered loop.
      circular_buffered_loop_index_variable_map_[loop_disjoint_set.get()] =
          std::make_unique<CircularBufferIndices>();
      for (auto i :
           arange(static_cast<int>(CircularBufferLoopStage::EndOfStages))) {
        auto stage = static_cast<CircularBufferLoopStage>(i);
        circular_buffered_loop_index_variable_map_[loop_disjoint_set.get()]
            ->emplace(stage, IrBuilder::create<Val>(DataType::Index));
      }
    } else if (GpuLower::current()
                   ->circularBufferInfo()
                   .isIndependentComputeWarpGroupsIterDomain(
                       concrete_loop_id)) {
      if (!computation_warp_group_index) {
        computation_warp_group_index = IrBuilder::create<Val>(DataType::Index);
      }
      loop_index_variable_map_[loop_disjoint_set.get()] =
          computation_warp_group_index;
    } else {
      // Everything now should be serial concrete loops,
      //   we just allocate a loop index integer for each set of loops.
      loop_index_variable_map_[loop_disjoint_set.get()] =
          IrBuilder::create<Val>(DataType::Index);
    }
  }
}

Val* ComputeAtMap::getIndexVariable(
    IterDomain* id,
    CircularBufferLoopStage circular_buffer_loop_stage) const {
  NVF_ERROR(
      id_graph_.loopNodes().mappingExists(id),
      "Index Variable: no index variable allocated as ",
      id->toString(),
      " is not registered in loop map");
  const auto* loop_set = &(id_graph_.loopNodes().getDisjointSetOf(id));

  // Check if this loop was modified by circular buffer pass.
  bool is_circular_buffer_iterdomain =
      GpuLower::current()->circularBufferInfo().isCircularBufferedIterDomain(
          id);

  if (is_circular_buffer_iterdomain) {
    // Use dedicated circular buffer index variable if the loop is circular
    // buffer loop
    if (circular_buffer_loop_stage == CircularBufferLoopStage::NotApplicable) {
      // The circular buffered loop stages are created after the loop nest
      //  lowering phase so this function will be queried before the circular
      //  buffer pass. At that point, no for loop has any circular buffer
      //  stage defined, and we just default to using the main stage index.
      circular_buffer_loop_stage = CircularBufferLoopStage::Main;
    }
    return circular_buffered_loop_index_variable_map_.at(loop_set)->at(
        circular_buffer_loop_stage);
  } else {
    return loop_index_variable_map_.at(loop_set);
  }
}

bool ComputeAtMap::areMapped(
    IterDomain* id0,
    IterDomain* id1,
    IdMappingMode mode) const {
  return disjointSetOf(id0, mode)->has(id1);
}

IterDomain* ComputeAtMap::computeConcreteId(
    IterDomain* id,
    IdMappingMode mode) {
  const auto& disjoint_set_shared_ptr = disjointSetOf(id, mode);

  NVF_ERROR(
      !disjoint_set_shared_ptr->vector().empty(),
      "Empty disjoint set found for ",
      id->toString());

  if (disjoint_set_shared_ptr->vector().size() == 1) {
    // If only one entry in the disjoint set, by definition the existing ID has
    // to be the concrete ID.
    return disjoint_set_shared_ptr->vector().front();
  }

  // Grab a set of candidate concrete_ids, we track towards the consumers in the
  // ID group as one of those is guaranteed to be a valid concrete id.
  VectorOfUniqueEntries<IterDomain*> maybe_concrete_ids;
  for (auto id : disjoint_set_shared_ptr->vector()) {
    bool id_output = true;
    for (auto consumer_id : id_graph_.consumers().at(id).vector()) {
      if (disjoint_set_shared_ptr->has(consumer_id)) {
        id_output = false;
        break;
      }
    }
    if (id_output) {
      maybe_concrete_ids.pushBack(id);
    }
  }

  // Shouldn't ever happen, it would mean there's an error somewhere in the
  // graph.
  NVF_ERROR(
      !maybe_concrete_ids.vector().empty(),
      "No potential concrete_id's found for ",
      id->toString());

  if (maybe_concrete_ids.vector().size() == 1) {
    return maybe_concrete_ids.vector().front();
  }

  // Broadcast resolution is what we have to figure out here. So if we traverse
  // back from loop domain to logical inputs through the exact map, if there's
  // an operation with a broadcast input that's resolved within the history all
  // of the domains in all of the logical_ids, then the concrete ID must resolve
  // that broadcast.
  //
  // (1) Compute "traversed IDs" which is every exact disjoint set starting at
  // all maybe concrete ID's traversing back through exact map.
  //
  // (2) Check all broadcast sets, remove from "traversed IDs" any broadcast set
  // that has its broadcast resolved ID within "traversed IDs", and all
  // IterDomains dependant on that broadcast.
  //
  // (3) Start at all "traversed IDs" set that has an logical domain, traverse
  // backwards to inputs and remove every exact ID set from "traversed IDs".
  //
  // Remove (2) and (3) from (1) and we have the iteration domains we must
  // resolve. The concrete ID must be in that set.
  //
  // Find any maybe concrete ID through the same iter/broadcast counting as
  // before as it should work fine.

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      maybe_concrete_exact_sets;

  for (auto maybe_concrete_id : maybe_concrete_ids) {
    maybe_concrete_exact_sets.pushBack(
        disjointSetOf(maybe_concrete_id, IdMappingMode::EXACT));
  }

  // Going to iteratively modify this to be all sets that the concrete ID needs
  // to cover
  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      all_exact_sets_covered =
          getAllDisjointSetProducers(maybe_concrete_exact_sets);

  // Remove all broadcast domains that are resolved within the history of any of
  // the maybe concrete sets.
  {
    // All broadcast exact sets in all_exact_sets_covered that are resolved by
    // IterDomains in all_exact_sets_covered
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        resolved_broadcasts;

    for (const auto& exact_set : all_exact_sets_covered) {
      NVF_ERROR(
          !exact_set->vector().empty(),
          "Cannot compute concrete id of empty set.");
      auto c_id = getConcreteMappedID(
          exact_set->vector().front(), IdMappingMode::EXACT);

      if (!c_id->isBroadcast()) {
        continue;
      }

      bool concretized_in_group = false;
      for (auto bcast_id : exact_set->vector()) {
        auto concretized_ids =
            concretized_bcasts_.allConcretizedDomains(bcast_id);
        for (auto concretized_id : concretized_ids) {
          if (all_exact_sets_covered.has(
                  disjointSetOf(concretized_id, IdMappingMode::EXACT))) {
            concretized_in_group = true;
            break;
          }
        }
        if (concretized_in_group) {
          break;
        }
      }

      if (concretized_in_group) {
        resolved_broadcasts.pushBack(exact_set);
      }
    }

    // Need to remove all uses of broadcast dims that are resolved in this
    // group, and all their uses.
    auto all_resolved_broadcast_uses =
        getAllDisjointSetConsumers(resolved_broadcasts);

    // Remove broadcast resolved sets from all_exact_sets_covered by effectively
    // doing an inplace copy_if
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        tmp_all_exact_sets_covered;
    std::swap(tmp_all_exact_sets_covered, all_exact_sets_covered);
    for (const auto& entry : tmp_all_exact_sets_covered) {
      if (all_resolved_broadcast_uses.has(entry)) {
        continue;
      }
      all_exact_sets_covered.pushBack(entry);
    }
  }

  // Remove all domains in the history of sets marked as rfactor.
  {
    // All exact sets in the history of an logical domain
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        produces_logical_dom;
    for (const auto& exact_set : all_exact_sets_covered) {
      if (produces_logical_dom.has(exact_set)) {
        // Already processed
        continue;
      }
      if (std::none_of(
              exact_set->vector().begin(),
              exact_set->vector().end(),
              [&](IterDomain* id) { return isRfactor(id); })) {
        continue;
      }
      VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
          root_to_logical_history = getAllDisjointSetProducers({exact_set});
      for (const auto& entry : root_to_logical_history) {
        // Leave logical exact set, unless it's in the history of another
        // logical domain.
        if (entry != exact_set) {
          produces_logical_dom.pushBack(entry);
        }
      }
    }

    // Remove all sets in root to logical history from all_exact_sets_covered by
    // effectively doing an inplace copy_if
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        tmp_all_exact_sets_covered;
    std::swap(tmp_all_exact_sets_covered, all_exact_sets_covered);
    for (const auto& entry : tmp_all_exact_sets_covered) {
      if (produces_logical_dom.has(entry)) {
        continue;
      }
      all_exact_sets_covered.pushBack(entry);
    }
  }

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      input_ids;

  {
    // Remove any concrete id that's not still in all_exact_sets_covered,
    // basically copy_if
    decltype(maybe_concrete_ids) tmp_maybe_concrete_ids;
    std::swap(maybe_concrete_ids, tmp_maybe_concrete_ids);
    for (auto entry : tmp_maybe_concrete_ids) {
      if (all_exact_sets_covered.has(
              disjointSetOf(entry, IdMappingMode::EXACT))) {
        maybe_concrete_ids.pushBack(entry);
      }
    }
  }

  NVF_ERROR(
      !maybe_concrete_ids.vector().empty(),
      "No potential concrete_id's found for disjoint set ",
      disjoint_set_shared_ptr->toString());

  if (maybe_concrete_ids.vector().size() == 1) {
    return maybe_concrete_ids.vector().front();
  }

  // The concrete_id should have the most roots it can trace back to that are
  // iter domains, (non-broadcast/non-reduction). We don't trace back through
  // view operations, so the one with the most iter root domains is the concrete
  // ID.
  IterDomain* concrete_id = nullptr;
  int max_iter_root_count = 0;
  int max_bcast_root_count = 0;

  for (auto maybe_concrete_id : maybe_concrete_ids.vector()) {
    auto concrete_id_root_sets = getInputDisjointSetsOf(maybe_concrete_id);

    int bcast_root_count = (int)std::count_if(
        concrete_id_root_sets.vector().begin(),
        concrete_id_root_sets.vector().end(),
        [&](std::shared_ptr<VectorOfUniqueEntries<IterDomain*>> set) {
          return set->vector()[0]->isBroadcast();
        });

    int iter_root_count =
        (int)concrete_id_root_sets.vector().size() - bcast_root_count;
    if (iter_root_count > max_iter_root_count ||
        (iter_root_count == max_iter_root_count &&
         bcast_root_count > max_bcast_root_count)) {
      max_iter_root_count = iter_root_count;
      max_bcast_root_count = bcast_root_count;
      concrete_id = maybe_concrete_id;
    }
  }

  NVF_ERROR(
      concrete_id != nullptr,
      "No concrete_id found for disjoint set ",
      disjoint_set_shared_ptr->toString());

  return concrete_id;
}

void ComputeAtMap::buildConcreteIds() {
  // For the exact map just select the first ID since they're all exactly the
  // same size, it doesn't matter which is selected. This should be run-to-run
  // deterministic but which ID gets selected her depends on the traversal order
  // generating the set (compute at map build).
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.exactNodes().disjointSets()) {
    NVF_ERROR(
        !disjoint_set_shared_ptr->vector().empty(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    concrete_id_cache_[disjoint_set_shared_ptr] = first_id;
  }

  // The following two algorithms seem quite wasteful. Should find a more
  // efficient way to compute concrete IDs.
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.permissiveNodes().disjointSets()) {
    NVF_ERROR(
        !disjoint_set_shared_ptr->vector().empty(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::PERMISSIVE);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }

  // Same as exact computation
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.almostExactNodes().disjointSets()) {
    NVF_ERROR(
        !disjoint_set_shared_ptr->vector().empty(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::ALMOSTEXACT);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }

  for (const auto& disjoint_set_shared_ptr :
       id_graph_.loopNodes().disjointSets()) {
    NVF_ERROR(
        !disjoint_set_shared_ptr->vector().empty(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::LOOP);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }

  for (const auto& disjoint_set_shared_ptr :
       id_graph_.permissiveResizeNodes().disjointSets()) {
    NVF_ERROR(
        !disjoint_set_shared_ptr->vector().empty(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id =
        computeConcreteId(first_id, IdMappingMode::PERMISSIVE_RESIZE);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }
}

bool ComputeAtMap::areExactExprs(Expr* expr_1, Expr* expr_2) {
  if (typeid(*expr_1) != typeid(*expr_2)) {
    return false;
  }

  if (expr_1->isA<Swizzle2D>()) {
    auto swizzle_1 = expr_1->as<Swizzle2D>();
    auto swizzle_2 = expr_2->as<Swizzle2D>();
    if (swizzle_1->swizzleType() != swizzle_2->swizzleType() ||
        swizzle_1->swizzleMode() != swizzle_2->swizzleMode()) {
      return false;
    }
  }

  if (expr_1->isA<Swizzle>()) {
    auto swizzle_1 = expr_1->as<Swizzle>();
    auto swizzle_2 = expr_2->as<Swizzle>();
    if (swizzle_1->swizzleType() != swizzle_2->swizzleType()) {
      return false;
    }
  }

  NVF_ERROR(
      expr_1->inputs().size() == expr_2->inputs().size() &&
          expr_1->outputs().size() == expr_2->outputs().size(),
      "Expr traversal doesn't support variable number of inputs and outputs.");

  for (auto input_i : arange(expr_1->inputs().size())) {
    if (expr_1->inputs()[input_i]->isA<IterDomain>() &&
        !areMapped(
            expr_1->inputs()[input_i]->as<IterDomain>(),
            expr_2->inputs()[input_i]->as<IterDomain>(),
            IdMappingMode::EXACT)) {
      // Inputs don't exact map in the right order
      return false;
    }
  }

  for (auto output_i : arange(expr_1->outputs().size())) {
    if (expr_1->outputs()[output_i]->isA<IterDomain>() &&
        !areMapped(
            expr_1->outputs()[output_i]->as<IterDomain>(),
            expr_2->outputs()[output_i]->as<IterDomain>(),
            IdMappingMode::EXACT)) {
      // Outputs don't exact map in the right order
      return false;
    }
  }
  // Expr's have exact mapped inputs and outputs, including parameters of the
  // transformation.
  return true;
}

void ComputeAtMap::buildUniqueExactExprMaps() {
  // Start by building definitions
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.exactNodes().disjointSets()) {
    std::vector<Expr*> definitions;

    // N^2 in number of unique transformations, this might be better to do
    // when generating the map.
    for (auto id : disjoint_set_shared_ptr->vector()) {
      if (id->definition() != nullptr) {
        auto id_inputs =
            ir_utils::filterByType<IterDomain>(id->definition()->inputs());
        // If any input ID is not included in the map, this definition
        // should not be included either.
        if (std::any_of(id_inputs.begin(), id_inputs.end(), [&](auto id_input) {
              return !idExistsInMap(id_input);
            })) {
          continue;
        }
        if (std::any_of(id_inputs.begin(), id_inputs.end(), [&](auto id_input) {
              return disjoint_set_shared_ptr->has(id_input);
            })) {
          // Definition to this exact map, shouldn't be marked as a definition
          // to traverse on the exact map.

          // This is a WAR for FusionSimpleSwizzle2_CUDA wher there is a pattern
          // like:
          //
          // tv0[32, 32]
          // tv0->swizzle(Swizzle2DType::ZShape, 0, 1);
          //
          // each root domain is exact mapped with the outputs of the swizzle.
          // So the pre and post swizzle ID is in an exact set, but that exact
          // set also has the swizzle as a definition that leads to itself.
          //
          // TODO: Try to formalize this better in the exact ID traversal. Right
          // now its just interfering with concrete ID detection.
          continue;
        }
        bool match = false;
        for (auto recorded_def : definitions) {
          if (areExactExprs(id->definition(), recorded_def)) {
            match = true;
            break;
          }
        }
        if (!match) {
          definitions.push_back(id->definition());
        }
      }
    }
    unique_exact_definitions_[disjoint_set_shared_ptr] = definitions;
  }

  // Use definitions to build uses
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.exactNodes().disjointSets()) {
    // Make sure uses is always initialized even there are no uses.
    if (unique_exact_uses_.find(disjoint_set_shared_ptr) ==
        unique_exact_uses_.end()) {
      unique_exact_uses_[disjoint_set_shared_ptr] = {};
    }

    auto definition_it =
        unique_exact_definitions_.find(disjoint_set_shared_ptr);

    if (definition_it == unique_exact_definitions_.end()) {
      continue;
    }

    const auto& definitions = definition_it->second;

    for (auto definition : definitions) {
      auto inp_ids = ir_utils::filterByType<IterDomain>(definition->inputs());
      for (auto inp : inp_ids) {
        auto inp_disjoint_set_shared_ptr =
            disjointSetOf(inp, IdMappingMode::EXACT);
        // Initialize uses entry
        if (unique_exact_uses_.find(inp_disjoint_set_shared_ptr) ==
            unique_exact_uses_.end()) {
          unique_exact_uses_[inp_disjoint_set_shared_ptr] = {};
        }

        auto& uses = unique_exact_uses_.at(inp_disjoint_set_shared_ptr);

        bool already_added = false;
        for (auto other_use : uses) {
          if (areExactExprs(definition, other_use)) {
            already_added = true;
            break;
          }
        }
        if (already_added) {
          continue;
        }

        if (!already_added) {
          uses.push_back(definition);
        }
      }
    }
  }
}

IterDomain* ComputeAtMap::getConcreteMappedID(
    IterDomain* id,
    IdMappingMode mode) const {
  auto disjoint_set_shared_ptr = disjointSetOf(id, mode);

  NVF_ERROR(
      !disjoint_set_shared_ptr->vector().empty(),
      "Empty disjoint set found for ",
      id->toString());

  auto cache_it = concrete_id_cache_.find(disjoint_set_shared_ptr);

  NVF_ERROR(
      cache_it != concrete_id_cache_.end(),
      "Could not find concrete id for: ",
      id->toString(),
      " with mode ",
      mode);

  return cache_it->second;
}

namespace {

std::string idGraphNodesToString(
    const ComputeAtMap& ca_map,
    IdMappingMode mode) {
  std::stringstream ss;
  // Sort vectors before printing so that the resulting output is
  // printed deterministically
  auto disjoint_sets = ca_map.getIdSets(mode).disjointSets();
  std::sort(
      disjoint_sets.begin(),
      disjoint_sets.end(),
      [&](const auto& set1, const auto& set2) {
        if (set1->empty()) {
          return true;
        } else if (set2->empty()) {
          return false;
        } else {
          auto concrete_id1 = ca_map.getConcreteMappedID(set1->front(), mode);
          auto concrete_id2 = ca_map.getConcreteMappedID(set2->front(), mode);
          return Statement::lessThan(concrete_id1, concrete_id2);
        }
      });
  for (const auto& s_ptr : disjoint_sets) {
    const auto& set = *s_ptr;
    IterDomain* concrete_id = nullptr;
    if (!set.empty()) {
      auto id = set.front();
      concrete_id = ca_map.getConcreteMappedID(id, mode);
    }
    ss << "  {";
    for (auto entry : set.vector()) {
      ss << abstractToString(entry);
      if (entry == concrete_id) {
        ss << "*";
      }
      if (entry != set.back()) {
        ss << "; ";
      }
    }
    ss << " }\n";
  }
  return ss.str();
}

} // namespace

std::string ComputeAtMap::toString() const {
  std::stringstream ss;
  ss << "Compute at map { \n";
  ss << "Exact map:\n" << idGraphNodesToString(*this, IdMappingMode::EXACT);
  ss << "Almost Exact map:\n"
     << idGraphNodesToString(*this, IdMappingMode::ALMOSTEXACT);
  ss << "Loop map:\n" << idGraphNodesToString(*this, IdMappingMode::LOOP);
  ss << "Permissive map:\n"
     << idGraphNodesToString(*this, IdMappingMode::PERMISSIVE);
  ss << "Permissive-Resize map:\n"
     << idGraphNodesToString(*this, IdMappingMode::PERMISSIVE_RESIZE);
  ss << "Consumer maps:\n";
  for (auto key : getSortedKeys(id_graph_.consumers(), Statement::lessThan)) {
    auto consumers = id_graph_.consumers().at(key);
    std::sort(consumers.begin(), consumers.end(), Statement::lessThan);
    ss << "  " << key->toString() << " :: " << consumers.toString() << "\n";
  }

  ss << "Producer maps:\n";
  for (auto key : getSortedKeys(id_graph_.producers(), Statement::lessThan)) {
    VectorOfUniqueEntries<IterDomain*> producers =
        id_graph_.producers().at(key);
    std::sort(producers.begin(), producers.end(), Statement::lessThan);
    ss << "  " << key->toString() << " :: " << producers.toString() << "\n";
  }

  ss << "Sibling map:\n" << id_graph_.siblings().toString() << "\n";

  ss << "} compute at map" << std::endl;
  return ss.str();
}

bool ComputeAtMap::isRfactor(IterDomain* ref_id) const {
  return id_graph_.rfactorIds().find(ref_id) != id_graph_.rfactorIds().end();
}

std::vector<IterDomain*> ComputeAtMap::getLogicalDomainsOfIdGroup(
    IterDomain* ref_id,
    IdMappingMode mode) const {
  auto disjoint_set = disjointSetOf(ref_id, mode);
  std::vector<IterDomain*> logical_ids;
  for (auto disjoint_id : disjoint_set->vector()) {
    if (id_graph_.rfactorIds().find(disjoint_id) !=
        id_graph_.rfactorIds().end()) {
      logical_ids.push_back(disjoint_id);
    }
  }
  return logical_ids;
}

const std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>& ComputeAtMap::
    disjointSetOf(IterDomain* id, IdMappingMode mode) const {
  NVF_ERROR(
      idExistsInMap(id),
      id->toString(),
      " has not been processed in this Compute At Map, yet the disjoint set for it was requested.");
  return getIdSets(mode).disjointSetMap().at(id);
}

const DisjointSets<IterDomain*>& ComputeAtMap::getIdSets(
    IdMappingMode mode) const {
  switch (mode) {
    case IdMappingMode::EXACT:
      return id_graph_.exactNodes();
    case IdMappingMode::ALMOSTEXACT:
      return id_graph_.almostExactNodes();
    case IdMappingMode::LOOP:
      return id_graph_.loopNodes();
    case IdMappingMode::PERMISSIVE:
      return id_graph_.permissiveNodes();
    case IdMappingMode::PERMISSIVE_RESIZE:
      return id_graph_.permissiveResizeNodes();
    case IdMappingMode::INNERMOST:
      return id_graph_.innermostNodes();
    default:
      NVF_THROW("Error with mapping mode provided.");
  }
}

bool ComputeAtMap::idExistsInMap(IterDomain* id, IdMappingMode mode) const {
  return getIdSets(mode).disjointSetMap().find(id) !=
      getIdSets(mode).disjointSetMap().end();
}

VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
ComputeAtMap::getInputDisjointSetsOf(IterDomain* of_id, bool stop_at_logical) {
  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      input_disjoint_sets;

  VectorOfUniqueEntries<IterDomain*> inputs;
  // This deque could be VectorOfUniqueEntries
  std::deque<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>> to_visit(
      {disjointSetOf(of_id, IdMappingMode::EXACT)});
  std::unordered_set<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      visited;
  while (!to_visit.empty()) {
    auto currently_visiting = to_visit.front();
    to_visit.pop_front();
    if (!visited.emplace(currently_visiting).second) {
      continue;
    }
    auto defs_it = unique_exact_definitions_.find(currently_visiting);
    NVF_ERROR(
        defs_it != unique_exact_definitions_.end(),
        "unique_exact_definitions_ wasn't correctly generated, missing the disjoint set:\n",
        currently_visiting->toString());

    // If there's no definition, we've found an input.
    if (defs_it->second.empty()) {
      input_disjoint_sets.pushBack(currently_visiting);
      continue;
    }

    if (stop_at_logical &&
        std::any_of(
            currently_visiting->vector().begin(),
            currently_visiting->vector().end(),
            [&](IterDomain* id) { return isRfactor(id); })) {
      input_disjoint_sets.pushBack(currently_visiting);
      continue;
    }

    // Traverse producers of current disjoint set and collect unique exact
    // disjoint set producers
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        producers_of_currently_visiting;

    for (auto def : defs_it->second) {
      auto id_inps = ir_utils::filterByType<IterDomain>(def->inputs());
      for (auto id_inp : id_inps) {
        producers_of_currently_visiting.pushBack(
            disjointSetOf(id_inp, IdMappingMode::EXACT));
      }
    }

    // Add producers to visit if not already there
    for (const auto& producer : producers_of_currently_visiting.vector()) {
      if (visited.find(producer) == visited.end()) {
        to_visit.push_back(producer);
      }
    }
  }

  return input_disjoint_sets;
}

VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
ComputeAtMap::getAllDisjointSetProducers(
    const VectorOfUniqueEntries<
        std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>& exact_sets)
    const {
  // This deque could be VectorOfUniqueEntries
  std::deque<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>> to_visit(
      {exact_sets.vector().begin(), exact_sets.vector().end()});

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      visited;

  while (!to_visit.empty()) {
    auto currently_visiting = to_visit.front();
    to_visit.pop_front();
    if (!visited.pushBack(currently_visiting)) {
      continue;
    }
    auto defs_it = unique_exact_definitions_.find(currently_visiting);
    NVF_ERROR(
        defs_it != unique_exact_definitions_.end(),
        "unique_exact_definitions_ wasn't correctly generated, missing the disjoint set:\n",
        currently_visiting->toString());

    // Traverse producers of current disjoint set and collect unique exact
    // disjoint set producers
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        producers_of_currently_visiting;

    for (auto def : defs_it->second) {
      auto id_inps = ir_utils::filterByType<IterDomain>(def->inputs());
      for (auto id_inp : id_inps) {
        producers_of_currently_visiting.pushBack(
            disjointSetOf(id_inp, IdMappingMode::EXACT));
      }
    }

    // Add producers to visit if not already there
    for (const auto& producer : producers_of_currently_visiting.vector()) {
      if (!visited.has(producer)) {
        to_visit.push_back(producer);
      }
    }
  }

  return visited;
}

VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
ComputeAtMap::getAllDisjointSetConsumers(
    const VectorOfUniqueEntries<
        std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>& exact_sets)
    const {
  // This deque could be VectorOfUniqueEntries
  std::deque<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>> to_visit(
      {exact_sets.vector().begin(), exact_sets.vector().end()});

  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      visited;

  while (!to_visit.empty()) {
    auto currently_visiting = to_visit.front();
    to_visit.pop_front();
    if (!visited.pushBack(currently_visiting)) {
      continue;
    }
    auto uses_it = unique_exact_uses_.find(currently_visiting);
    NVF_ERROR(
        uses_it != unique_exact_uses_.end(),
        "unique_exact_uses_ wasn't correctly generated, missing the disjoint set:\n",
        currently_visiting->toString());

    // Traverse consumers of current disjoint set and collect unique exact
    // disjoint set consumers
    VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
        consumers_of_currently_visiting;

    for (auto uses : uses_it->second) {
      auto id_outs = ir_utils::filterByType<IterDomain>(uses->outputs());
      for (auto id_out : id_outs) {
        consumers_of_currently_visiting.pushBack(
            disjointSetOf(id_out, IdMappingMode::EXACT));
      }
    }

    // Add consumers to visit if not already there
    for (const auto& consumer : consumers_of_currently_visiting.vector()) {
      if (!visited.has(consumer)) {
        to_visit.push_back(consumer);
      }
    }
  }

  return visited;
}

void IterDomainGraph::updateComputeWith(TensorView* compute_with_tv) {
  NVF_ERROR(
      compute_with_tv->hasResolvedComputeWith(),
      "Invalid tensor: ",
      compute_with_tv->toString());

  // Can use any consumer this tensor is computed with
  auto consumer_tv = compute_with_tv->getComputeWithConsumers().at(0);

  for (auto pos = compute_with_tv->getComputeAtPosition();
       pos < compute_with_tv->getComputeWithPosition();
       ++pos) {
    auto id = compute_with_tv->axis(pos);

    // Find the matching consumer ID using the permissive map
    auto it = std::find_if(
        consumer_tv->getLoopDomain().begin(),
        consumer_tv->getLoopDomain().end(),
        [&](auto consumer_id) {
          return permissiveNodes().disjointSetMap().at(id)->has(consumer_id);
        });
    NVF_ERROR(
        it != consumer_tv->getLoopDomain().end(),
        "No consumer loop ID of tensor ",
        consumer_tv->toString(),
        " permissively mapped with: ",
        id->toString());

    IterDomain* consumer_id = *it;

    loop_nodes_.mapEntries(id, consumer_id);
  }
}

void ComputeAtMap::updateComputeWith(TensorView* compute_with_tv) {
  NVF_ERROR(
      compute_with_tv->hasResolvedComputeWith(),
      "Invalid tensor: ",
      compute_with_tv->toString());

  id_graph_.updateComputeWith(compute_with_tv);

  // Update the LOOP concrete IDs
  for (const auto& disjoint_set_shared_ptr :
       id_graph_.loopNodes().disjointSets()) {
    NVF_ERROR(
        !disjoint_set_shared_ptr->vector().empty(),
        "Cannot compute concrete id of empty set.");
    auto first_id = disjoint_set_shared_ptr->vector().front();
    auto concrete_id = computeConcreteId(first_id, IdMappingMode::LOOP);
    concrete_id_cache_[disjoint_set_shared_ptr] = concrete_id;
  }
}

} // namespace nvfuser
