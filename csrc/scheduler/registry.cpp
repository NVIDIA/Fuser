// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <c10/util/irange.h>
#include <disjoint_set.h>
#include <executor_utils.h>
#include <expr_evaluator.h>
#include <instrumentation.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <root_domain_map.h>
#include <scheduler/debug_utils.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/pointwise.h>
#include <scheduler/registry.h>
#include <scheduler/transpose.h>
#include <scheduler/utils.h>

#include <limits>

#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {

namespace {
// TODO: Deduplicate from compute_at.cpp
std::deque<std::deque<TensorView*>> tvChains(
    std::deque<std::deque<Val*>> val_chains) {
  std::deque<std::deque<TensorView*>> tv_chains(val_chains.size());
  for (const auto i : c10::irange(val_chains.size())) {
    auto tv_iterable = ir_utils::filterByType<TensorView>(val_chains[i]);
    tv_chains[i] =
        std::deque<TensorView*>(tv_iterable.begin(), tv_iterable.end());
  }
  return tv_chains;
}

bool rejectScheduleFusionInputRequirement(
    Expr* expr,
    ScheduleHeuristic schedule_stragety) {
  if (!expr->input(0)->isFusionInput()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_stragety,
        "First input of ",
        expr->getOpString(),
        " must be fusion input.");
    return true;
  }
  if (expr->input(0)->uses().size() > 1) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_stragety,
        "First input of ",
        expr->getOpString(),
        " can only be used by ",
        expr->getOpString());
    return true;
  }
  return false;
}

bool rejectScheduleForSelectLikeOps(
    Fusion* fusion,
    ScheduleHeuristic schedule_strategy) {
  for (auto expr : fusion->exprs()) {
    // For now, only relax the input requirement with take_along_axis.
    // TODO: remove this requirement entirely
    if ((expr->isOneOf<SelectOp, IndexSelectOp>() ||
         (expr->isA<TorchGatherOp>() &&
          !expr->as<TorchGatherOp>()->exactSizes())) &&
        rejectScheduleFusionInputRequirement(expr, schedule_strategy)) {
      return true;
    }
  }
  return false;
}

class SchedulerTopologyChecker {
 public:
  // Checks if any broadcasts are resolved after a reduction that don't follow
  // the normalization pattern
  static bool hasNonNormalizePostReductionBCast(Fusion* fusion) {
    auto all_vals = fusion->usedMathVals();
    std::vector<TensorView*> reduction_tvs;
    for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
      if (tv->hasReduction() &&
          !(fusion == tv->fusion() && tv->isFusionInput())) {
        reduction_tvs.push_back(tv);
      }
    }

    // All tensor views that are eventually consumed to produce a reduction,
    // includes reduction tensor views.
    std::unordered_set<TensorView*> pre_reduction_tvs;

    {
      auto pre_reduction_vals = DependencyCheck::getAllValsBetween(
          {fusion->inputs().begin(), fusion->inputs().end()},
          {reduction_tvs.begin(), reduction_tvs.end()});
      auto pre_reduction_tv_vector =
          ir_utils::filterByType<TensorView>(pre_reduction_vals);
      pre_reduction_tvs = std::unordered_set<TensorView*>(
          pre_reduction_tv_vector.begin(), pre_reduction_tv_vector.end());
    }

    // Track which tensor views we've validated so we don't do it again.
    std::unordered_set<TensorView*> validated_resolved_tvs;

    // Run forward (towards outputs) from reductions on any path that isn't
    // before another reduction. Look for resolved broadcasts. If a resolved
    // broadcast is found, start there and propagate backwards. Track the id's
    // that were resolved and make sure there's a mapping to a TensorView before
    // a reduction.
    for (auto red_tv : reduction_tvs) {
      auto forward_tv_chains =
          tvChains(DependencyCheck::getAllUseChains(red_tv));
      // Propagate forward from reduction through all uses of the reduction
      for (auto forward_tv_dep_chain : forward_tv_chains) {
        TensorView* forward_running_producer = nullptr;
        TensorView* forward_running_consumer = forward_tv_dep_chain.front();
        forward_tv_dep_chain.pop_front();
        while (!forward_tv_dep_chain.empty()) {
          forward_running_producer = forward_running_consumer;
          forward_running_consumer = forward_tv_dep_chain.front();
          forward_tv_dep_chain.pop_front();

          if (std::none_of(
                  forward_running_producer->getMaybeRFactorDomain().begin(),
                  forward_running_producer->getMaybeRFactorDomain().end(),
                  [](IterDomain* id) { return id->isBroadcast(); })) {
            // If there's no broadcast axes in producer it doesn't need to be
            // checked
            continue;
          }

          // If consumer is before another reduction it doesn't need to be
          // checked
          if (pre_reduction_tvs.count(forward_running_consumer)) {
            break;
          }

          // If consumer was already validated it doesn't need to be checked
          if (validated_resolved_tvs.count(forward_running_consumer)) {
            continue;
          }

          auto forward_pairwise_root_map = PairwiseRootDomainMap(
              forward_running_producer, forward_running_consumer);
          auto forward_p2c_root_map =
              forward_pairwise_root_map.mapProducerToConsumer(
                  forward_running_producer->domain(),
                  forward_running_consumer->domain());

          // These are the ids we will have to resolve. As we resolve them we'll
          // remove them from this vector. If this vector ends up empty, then
          // we've resolved everything we need to. This is a pair so as we
          // traverse we can map the id through the traversal. The first entry
          // in the pair will be the original id so we can reset it if it's not
          // resolved before the next traversal. The second ID will be
          // propagated as we map the IDs through the backward traversal.
          std::vector<std::pair<IterDomain*, IterDomain*>> ids_to_resolve;

          // Check if any TensorViews have a resolved broadcast
          for (auto entry : forward_p2c_root_map) {
            auto p_id = entry.first;
            auto c_id = entry.second;
            if (p_id->isBroadcast() && !c_id->isBroadcast()) {
              ids_to_resolve.emplace_back(c_id, c_id);
            }
          }

          if (ids_to_resolve.empty()) {
            continue;
          }

          // Only because of api limitations in getAllDependencyChains
          auto inputs_of_forward_running_consumer =
              IterVisitor::getInputsTo({forward_running_consumer});
          auto tv_inputs_of_forward_running_consumer =
              ir_utils::filterByType<TensorView>(
                  inputs_of_forward_running_consumer);

          for (auto input_of_forward_running_consumer :
               tv_inputs_of_forward_running_consumer) {
            if (pre_reduction_tvs.find(input_of_forward_running_consumer) ==
                pre_reduction_tvs.end()) {
              // If this input isn't an input to a reduction, no point
              // traversing the dependency chains as we know we can't validate
              // this broadcast through chains to this input
              continue;
            }

            auto backward_tv_chains =
                tvChains(DependencyCheck::getAllDependencyChains(
                    input_of_forward_running_consumer,
                    forward_running_consumer));

            for (auto backward_tv_chain : backward_tv_chains) {
              if (ids_to_resolve.empty()) {
                break;
              }

              for (auto& pair : ids_to_resolve) {
                pair.second = pair.first;
              }

              TensorView* backward_running_producer = backward_tv_chain.back();
              TensorView* backward_running_consumer = nullptr;
              backward_tv_chain.pop_back();

              TORCH_INTERNAL_ASSERT(
                  backward_running_producer == forward_running_consumer);

              while (!backward_tv_chain.empty()) {
                backward_running_consumer = backward_running_producer;
                backward_running_producer = backward_tv_chain.back();
                backward_tv_chain.pop_back();

                std::vector<IterDomain*> running_resolved_ids;

                auto backward_pairwise_root_map = PairwiseRootDomainMap(
                    backward_running_producer, backward_running_consumer);

                auto backward_c2p_root_map =
                    backward_pairwise_root_map.mapConsumerToProducer(
                        backward_running_consumer->domain(),
                        backward_running_producer->domain());

                // Mark if producer is a producer of a reduction
                bool producer_resolves =
                    pre_reduction_tvs.count(backward_running_producer);

                bool at_leat_one_id_mapped = false;
                for (size_t entry_i = ids_to_resolve.size(); entry_i > 0;
                     entry_i--) {
                  auto orig_id = ids_to_resolve[entry_i - 1].first;
                  auto running_id = ids_to_resolve[entry_i - 1].second;
                  if (backward_c2p_root_map.find(running_id) !=
                      backward_c2p_root_map.end()) {
                    at_leat_one_id_mapped = true;
                    if (producer_resolves &&
                        !backward_c2p_root_map.at(running_id)->isBroadcast()) {
                      // If mapped, and producer is a producer of a reduction,
                      // we can resolve this id
                      ids_to_resolve.erase(
                          ids_to_resolve.begin() + (int64_t)entry_i - 1);
                    } else {
                      ids_to_resolve[entry_i - 1] = std::make_pair(
                          orig_id, backward_c2p_root_map.at(running_id));
                    }
                  }
                }
                if (!at_leat_one_id_mapped) {
                  // If no id's map any more, go to the next chain
                  break;
                }

                if (ids_to_resolve.empty()) {
                  break;
                }
              }
            }
          } // for(auto input_of_forward_running_consumer :
            // tv_inputs_of_forward_running_consumer){

          // if all ids were not resolved, then we've found an instance of a
          // bad broadcast resolution after reduction
          if (!ids_to_resolve.empty()) {
            return true;
          }

        } // while (!forward_tv_dep_chain.empty()) {
      } // for (auto forward_tv_dep_chain : forward_tv_chains) {
    } // for (auto red_tv : reduction_tvs)
    return false;
  }

  // Checks if any broadcasts are resolved after a reduction, this shouldn't be
  // accepted in the single reduction or multi-reduction scheduler
  static bool hasPostReductionBCast(Fusion* fusion) {
    auto all_vals = fusion->usedMathVals();
    for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
      // Reductions can have multiple outputs, so do this on all found reduction
      // tensor views
      if (tv->hasReduction() && !tv->isFusionInput()) {
        auto tv_chains = tvChains(DependencyCheck::getAllUseChains(tv));
        // Propagate forward from reduction through all uses of the reduction
        for (auto tv_dep_chain : tv_chains) {
          TensorView* running_producer = nullptr;
          TensorView* running_consumer = tv_dep_chain.front();
          tv_dep_chain.pop_front();
          while (!tv_dep_chain.empty()) {
            running_producer = running_consumer;
            running_consumer = tv_dep_chain.front();
            tv_dep_chain.pop_front();

            auto pairwise_root_map =
                PairwiseRootDomainMap(running_producer, running_consumer);
            auto p2c_root_map = pairwise_root_map.mapProducerToConsumer(
                running_producer->domain(), running_consumer->domain());

            // Check if any TensorViews have a resolved broadcast
            for (auto entry : p2c_root_map) {
              auto p_id = entry.first;
              auto c_id = entry.second;
              if (p_id->isBroadcast() && !c_id->isBroadcast()) {
                return true;
              }
            }
          }
        }
      }
    }
    return false;
  }

  // Checks if there's any unsupported operations post reduction. If outer
  // reduction we can fuse some pointwise ops if they don't require
  // broadcasting (checked in hasPostReductionBCast). For inner reductions we
  // cannot fuse any binary like operation (includes operations like shift that
  // we're not fusing right now) involving "new" inputs (not going through a
  // reduction).
  static bool supportedPostReductionFusion(
      Fusion* fusion,
      std::vector<TensorView*> reduction_tvs) {
    TORCH_INTERNAL_ASSERT(!reduction_tvs.empty());
    bool fastest_dim_reduction = true;
    auto red_root_dom = reduction_tvs[0]->getRootDomain();
    for (size_t i = red_root_dom.size(); i > 0; i--) {
      if (red_root_dom[i - 1]->isBroadcast()) {
        continue;
      } else if (red_root_dom[i - 1]->isReduction()) {
        fastest_dim_reduction = true;
        break;
      } else {
        fastest_dim_reduction = false;
        break;
      }
    }

    // When checking post reduction vals, we need to make sure
    //  we are really checking paths starting from all outputs
    //  of multi-output reductions, i.e. welford/grouped reduction. The
    //  reduction_tv vector is assumed to only have one of them.
    std::unordered_set<Val*> reduction_tv_set(
        reduction_tvs.begin(), reduction_tvs.end());

    for (auto red : reduction_tvs) {
      if (red->definition()) {
        if (ir_utils::isReductionOp(red->definition())) {
          auto outs = red->definition()->outputs();
          for (auto out_tv : ir_utils::filterByType<TensorView>(outs)) {
            reduction_tv_set.insert(out_tv);
          }
        }
      }
    }

    // If reductions are on fastest dim, don't fuse any operations (after
    // reductions) that requires an input that is not an input to the
    // reductions.
    if (fastest_dim_reduction) {
      auto post_reduction_vals = DependencyCheck::getAllValsBetween(
          reduction_tv_set,
          {fusion->outputs().begin(), fusion->outputs().end()});

      if (post_reduction_vals.empty()) {
        return true;
      }

      auto reduction_inputs = IterVisitor::getInputsTo(
          {reduction_tvs.begin(), reduction_tvs.end()});

      for (auto tv : ir_utils::filterByType<TensorView>(
               post_reduction_vals.begin(), post_reduction_vals.end())) {
        if (tv->definition() == nullptr) {
          continue;
        }

        auto tv_inputs = IterVisitor::getInputsTo({tv});

        if (std::any_of(
                tv_inputs.begin(),
                tv_inputs.end(),
                [&reduction_inputs](Val* inp) {
                  return inp->isA<TensorView>() &&
                      std::find(
                          reduction_inputs.begin(),
                          reduction_inputs.end(),
                          inp) == reduction_inputs.end();
                })) {
          return false;
        }
      }
    }

    return true;
  }

  /* Returns if any non-trivial views are not before the reference. For example:
   *     t0
   *    /  \
   *  view ref
   *   |
   *   t1
   * This could be important as transform propagation from a reference backwards
   * through a view should always work, but transform propagation form a
   * reference forward through a view could interfere with the view transforms.
   */
  static bool hasViewNotBeforeRef(
      Fusion* fusion,
      const std::vector<TensorView*>& reference_tvs) {
    std::vector<TensorView*> view_tvs;
    auto view_ops = ir_utils::getViewOps(fusion);
    for (auto view_op : view_ops) {
      auto tv_outs = ir_utils::filterByType<TensorView>(view_op->outputs());
      for (auto entry : tv_outs) {
        view_tvs.push_back(entry);
      }
    }

    if (view_tvs.empty()) {
      return false;
    }

    // Terrible complexity, may be worth improving, but is a compile time
    // check.
    for (auto ref_tv : reference_tvs) {
      for (auto view_tv : view_tvs) {
        if (!DependencyCheck::isDependencyOf(view_tv, ref_tv)) {
          return true;
        }
      }
    }

    return false;
  }

  // Checks if there's any gather-like ops that result in non-resolved
  // broadcast domains and then get squeezed before reaching reduction
  // TVs. The reduction scheduler uses reduction TVs as a scheduling
  // reference, so that won't be able to schedule the broadcast ID if
  // squeezed and its corresponding index-accessed producer ID, and
  // any IDs that the producer ID depends on.
  //
  // This analysis has some similarity as DomainMap. Can be
  // consolidated?
  static bool hasGatherToBroadcastBeforeReduction(
      Fusion* fusion,
      const std::vector<TensorView*>& reduction_tvs) {
    std::vector<Val*> reduction_inputs;
    const auto all_exprs = DependencyCheck::getAllExprsBetween(
        {fusion->inputs().begin(), fusion->inputs().end()},
        {reduction_tvs.begin(), reduction_tvs.end()});

    // Grab all consumer domains of indexed domains by gather-like ops
    std::unordered_set<IterDomain*> broadcast_consumer_of_indexed_ids;
    for (auto expr : all_exprs) {
      auto in_tv = ir_utils::getTvInput(expr);
      // Fusion input does not conflict
      if (in_tv == nullptr || in_tv->isFusionInput()) {
        continue;
      }
      // In the case of select, there's no consumer domain, and thus
      // there's no way to schedule the indexed producer domain
      if (expr->isA<SelectOp>()) {
        return true;
      }
      if (auto consumer_of_indexed_producer =
              ir_utils::getConsumerOfIndexedProducerID(expr)) {
        if (consumer_of_indexed_producer->isBroadcast()) {
          broadcast_consumer_of_indexed_ids.insert(
              consumer_of_indexed_producer);
        }
      }
    }

    if (broadcast_consumer_of_indexed_ids.empty()) {
      return false;
    }

    // If the broadcast IDs are mapped with the reduction TVs, the
    // reduction scheduler should be able to schedule the gather
    // output TVs. This mapping can be PERMISSIVE as the broadcast IDs
    // may be concretized. ExactRootDomainMap may be enough as
    // broadcasts should not be removed by rfactor exprs.

    // Consider reusing a CA map
    ComputeAtMap ca_map(fusion);
    // All of reduction TVs are mapped, so doesn't matter which
    // reduction tv to use
    auto ref_tv = reduction_tvs.at(0);
    return std::any_of(
        broadcast_consumer_of_indexed_ids.begin(),
        broadcast_consumer_of_indexed_ids.end(),
        [&ca_map, &ref_tv](IterDomain* broadcast_consumer_id) {
          // Check if this broadcast ID has no mapping
          // with the reference TV.
          return std::none_of(
              ref_tv->getRootDomain().begin(),
              ref_tv->getRootDomain().end(),
              [&](IterDomain* red_tv_root_id) {
                return ca_map.areMapped(
                    broadcast_consumer_id,
                    red_tv_root_id,
                    IdMappingMode::PERMISSIVE);
              });
        });
  }
};

bool isConnectedFusionGraph(Fusion* fusion) {
  if (fusion->outputs().empty()) {
    // Trivial case interpreted as connected
    return true;
  }

  // A set of connected components on the fusion graph
  DisjointSets<Val*> component_sets;

  TORCH_INTERNAL_ASSERT(
      !fusion->outputs().empty(), "Fusion without output is not supported");
  auto output0 = fusion->outputs()[0];
  component_sets.initializeSet(output0);

  // Iterate through all used exprs
  for (auto expr : fusion->exprs()) {
    TORCH_INTERNAL_ASSERT(
        !expr->outputs().empty(), "unknown expr with zero output");

    // Each expr maps all its inputs and
    //  outputs to the same component
    auto output0 = expr->output(0);
    for (auto input : ir_utils::filterByType<TensorView>(expr->inputs())) {
      component_sets.mapEntries(output0, input);
    }
    for (auto output : expr->outputs()) {
      component_sets.mapEntries(output0, output);
    }
  }

  // Map aliased outputs
  for (auto alias_it : fusion->ioAlias()) {
    component_sets.mapEntries(alias_it.first, alias_it.second);
  }

  // Check connected-ness:
  //  If there is no independent compute flow
  // on this fusion graph, all outputs will be
  // equivalent/connected to the first output.
  for (auto output : fusion->outputs()) {
    if (!component_sets.strictAreMapped(output0, output)) {
      return false;
    }
  }
  return true;
}

// Returns if a fusion cannot transformed into a consistent format since we
// can't transform forward through view operations, for exmaple:
//
// tv0[I0, I1, I2]
// tv1[I0*I1, I2] = view(tv0)
// tv2[I0, I1*I2] = view(tv0)
//
// If we start transform propagation at either tv1 or tv2, it would require
// "replaying forward" through the other. If we started at tv1 we'd have to be
// able to take tv2[I0, I1*I2] and transform it to [I0*I1, I2], however this
// would "undo" the view transformation which we do not support today.
//
// Returns true if a scenario like above is found in the fusion.
bool requiresForwardViewReplay(Fusion* fusion, ComputeAtMap& ca_map) {
  // Track the uses of the rfactor domains in the fusion. If an rfactor domain
  // is used in more than one way it means the above situation is being
  // encountered.
  //
  // tv1 root: [I0rf, I1rf, I2] -> rfactor [I0*I1rf, I2]
  // tv1 root: [I0, I1rf, I2rf] -> rfactor [I0, I1*I2rf]
  //
  // Here we can see I1rf is used in two view transformations, one to I0*I1rf,
  // and the other to I1*I2rf.

  // Track the transformation each exact disjoint rfactor set is used in. If
  // more than one is detected we can't support transforming the fusion into a
  // consistent format.
  std::unordered_map<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>, Expr*>
      unique_exact_uses;

  // Don't check compute uses directly, as IterDomain->uses() isn't protected
  // from going outside the TensorViews between registered inputs and outputs of
  // the fusion. If there are view operations defined in the fusion container
  // (because of how segmentation works) but not between registered input and
  // outputs, that could be picked up as inconsistent view transformations.
  //
  // It would be unlikely this would be picked up as a conflict as we check
  // which definitions were registered in the compute at map for matching
  // transformations. However, we may want to support scheduling after
  // transformations which could map to those views not on the input->output
  // path.

  // Look through all definitions associated with producing rfactor outputs.
  // Mark those as an active use of the rfactor, if two are detected, return
  // true.
  for (const auto& disjoint_set_shared_ptr :
       ca_map.idGraph().exactNodes().disjointSets()) {
    // Make sure there's at least one rfactor domain in the set, otherwise we
    // don't need to check anything from this set.
    if (!std::any_of(
            disjoint_set_shared_ptr->vector().begin(),
            disjoint_set_shared_ptr->vector().end(),
            [](IterDomain* id) { return id->isRFactorProduct(); })) {
      continue;
    }

    // Grab all the unique definitions detected to consume the iter domains in
    // this set
    auto unique_defs =
        ca_map.uniqueExactDefinitions(disjoint_set_shared_ptr->back());

    // Iterate through the all the rfactor iter domains
    for (auto id_rfactor_product : disjoint_set_shared_ptr->vector()) {
      if (!id_rfactor_product->isRFactorProduct()) {
        continue;
      }

      // Grab the rfactor definition
      auto rfactor_def = id_rfactor_product->definition();

      if (rfactor_def == nullptr) {
        // Guard segfault if there isn't a definition for this iter domain
        continue;
      }

      // rfactor_def can be Resize, but resize transformation is not
      // replayed, so mismatch doesn't matter
      if (rfactor_def->isA<Resize>()) {
        continue;
      }

      // If one output of the expression is an rfactor ID all of them should be
      auto def_outs =
          ir_utils::filterByType<IterDomain>(rfactor_def->outputs());
      TORCH_INTERNAL_ASSERT(
          std::all_of(
              def_outs.begin(),
              def_outs.end(),
              [](IterDomain* id) { return id->isRFactorProduct(); }),
          "This function does not support outputs of transformations with mismatching rfactor flags. ",
          "If one output is rfactor all should be rfactor.");

      // If outputs are rfactor all the inputs should be as well. It doesn't
      // make sense to have transforms on non-rfactor domains that produce
      // rfactor domains.
      auto def_inps = ir_utils::filterByType<IterDomain>(rfactor_def->inputs());
      TORCH_INTERNAL_ASSERT(
          std::all_of(
              def_inps.begin(),
              def_inps.end(),
              [](IterDomain* id) { return id->isRFactorProduct(); }),
          "Inputs producing an rfactor domain, should be marked as rfactor but found:\n  ",
          rfactor_def->toString());

      // Check which definition in the unique exact definition set this
      // definition matches to:
      // TODO: Why does it need to check all unique defs? It actually
      // only looks at those that are exact with rfactor_def and
      // adds those unique_defs, which are all exactly mapped, to the
      // unique_exact_use map. Since the objective of this analysis is
      // to find non-exact exprs using the same exact ID set, it seems
      // it's just sufficient to register rfactor_def as a user of the
      // exact set.
      for (auto unique_def : unique_defs) {
        if (ca_map.areExactExprs(rfactor_def, unique_def)) {
          // Check if we already have an expression that consumes an
          // equivalent of any of the input rfactor domains. If so and it's
          // not the already registered transformation, return true
          for (auto inp : def_inps) {
            auto inp_disjoint_set =
                ca_map.disjointSetOf(inp, IdMappingMode::EXACT);
            // Initialize the use entry for this set (if it doesn't already
            // exist)
            if (unique_exact_uses.find(inp_disjoint_set) ==
                unique_exact_uses.end()) {
              unique_exact_uses[inp_disjoint_set] = nullptr;
            }

            if (unique_exact_uses.at(inp_disjoint_set) == nullptr) {
              // If expression is null pointer register this unique_def
              unique_exact_uses[inp_disjoint_set] = unique_def;
            } else if (!ca_map.areExactExprs(
                           unique_exact_uses[inp_disjoint_set], unique_def)) {
              // Two transformations that don't match on matching rfactor
              // domains found, return true.
              return true;
            }
          }
          // Expression already mapped, stop trying to match expressions
          break;
        }
      }
    }
  }
  // No inconsistent rfactor uses found, we can safely transform this graph.
  return false;
}

// Returns if view interferes with how we want to treat the reference, being at
// least a 2D reduction schedule but maybe a 3D reduction schedule.
bool reductionInterferingView(
    Fusion* fusion,
    const ComputeAtMap& ca_map,
    TensorView* reduction_reference) {
  // Make sure the view doesn't interfere with how we'll want to schedule
  // it. If we might want to do a 3D scheduler make sure views are disjoint
  // based on what the 3D scheduler's merges would be.

  // Utility to take dimensions out of the vector that we've already
  // processed or don't want to process.
  auto remove_dims = [](const std::vector<IterDomain*>& dims,
                        std::unordered_set<IterDomain*> to_remove) {
    std::vector<IterDomain*> dims_removed;
    std::copy_if(
        dims.begin(),
        dims.end(),
        std::back_inserter(dims_removed),
        [&](IterDomain* id) { return to_remove.find(id) == to_remove.end(); });
    return dims_removed;
  };

  std::vector<IterDomain*> dims = reduction_reference->getMaybeRFactorDomain();

  // The disjoint groups we need for this scheduler
  std::vector<std::vector<IterDomain*>> groups;

  // Do this three times as we could have a 3D scheduler at maximum
  for (auto dimension : c10::irange(3)) {
    // Tracker for this group
    std::vector<IterDomain*> current_dims;

    // Tracker of what we've already processed to remove from dims
    std::unordered_set<IterDomain*> processed;

    for (auto i : c10::irange(dims.size())) {
      auto dim_i = dims.size() - i - 1;
      if (dims[dim_i]->isReduction() != dims[dims.size() - 1]->isReduction()) {
        if (dimension == 0) {
          // First dimension must be contiguous merges
          break;
        } else {
          // Other dimensions can be non contiguous merges
          continue;
        }
      }
      current_dims.push_back(dims[dim_i]);
      processed.emplace(dims[dim_i]);
    }

    // Don't add empty group (would happen if it's a 2D scheduler not 3D)
    if (!current_dims.empty()) {
      groups.push_back(current_dims);
      dims = remove_dims(dims, processed);
    }
  }

  TORCH_INTERNAL_ASSERT(
      dims.empty(), "Error processing ", dims, " in registry.cpp.");

  // Make sure groups are disjoint based on view

  auto disjoint_rfactor_sets = scheduler_utils::disjointRFactorSets(fusion);
  auto disjoint_set_information = scheduler_utils::getDisjointRFactorSetsOf(
      fusion, reduction_reference, disjoint_rfactor_sets);

  // Convert id's in groups to disjoint_set_ids of disjoint_set_information
  std::vector<std::vector<int>> disjoint_groups;

  for (const auto& group : groups) {
    std::vector<int> disjoint_id_sets;
    for (auto id : group) {
      auto find_it = std::find(
          reduction_reference->getMaybeRFactorDomain().begin(),
          reduction_reference->getMaybeRFactorDomain().end(),
          id);
      TORCH_INTERNAL_ASSERT(
          find_it != reduction_reference->getMaybeRFactorDomain().end(),
          "Issue with view analysis on reduction like schedule, with reference: ",
          reduction_reference->toString());
      auto rfactor_pos = std::distance(
          reduction_reference->getMaybeRFactorDomain().begin(), find_it);
      TORCH_INTERNAL_ASSERT(
          rfactor_pos < (int)disjoint_set_information.disjoint_set_ids.size(),
          "Error computing disjoint group on the rfactor domain of ",
          reduction_reference->toString());
      disjoint_id_sets.push_back(
          disjoint_set_information.disjoint_set_ids[rfactor_pos]);
    }
    disjoint_groups.push_back(disjoint_id_sets);
  }

  // Make sure there's no intersection between the groups, otherwise view
  // will interfere with the schedule. TODO: Make this better complexity,
  // since it should be relatively small int vectors of a small total nDims,
  // not too worried about it now.

  for (auto first_dim_i : c10::irange(disjoint_groups.size())) {
    for (auto second_dim_i = first_dim_i + 1;
         second_dim_i < disjoint_groups.size();
         ++second_dim_i) {
      auto first_group = disjoint_groups[first_dim_i];
      auto second_group = disjoint_groups[second_dim_i];
      for (auto first_disjoint_id : first_group) {
        for (auto second_disjoint_id : second_group) {
          if (first_disjoint_id == second_disjoint_id) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

PrimDataType getTensorIndexType(TensorView* tv, ExpressionEvaluator& ee) {
  TORCH_INTERNAL_ASSERT(
      !tv->isFusionInput(),
      "This function is not supposed to be used for fusion inputs: ",
      tv->toString());

  auto non_contig = std::any_of(
      tv->domain()->contiguity().begin(),
      tv->domain()->contiguity().end(),
      [](const auto contig) { return contig.has_value() && !contig.value(); });

  // When a fusion output is non-contiguous, currently there's no
  // way to obtain its strides. This is an interface problem and
  // should be fixed.
  if (tv->isFusionOutput() && non_contig) {
    return PrimDataType::Int;
  }

  // This function should not be used for fusion inputs, so any
  // non-contig tensor means a fusion intermediate tensor. However,
  // since we don't support non-contiguous intermediates, there must be
  // something wrong.
  TORCH_INTERNAL_ASSERT(
      !non_contig, "Unexpected non-contiguous tensor found: ", tv->toString());

  // Note that at this point tensors are not scheduled yet. Each
  // tensor may end up being inlined, stored on Shared or Local, but
  // the index type is currently supposed to be determined before
  // any of scheduling decisions is made, so we may end up making a
  // conservative decision.
  // TODO: Consider index type resolution after segmentation and
  // scheduling. At that point we have the final scheduled fusions
  // with which we can make more precise analyses. It would require
  // that the scheduling and segmentation should not have any
  // assumption about the index type as it may change.
  int64_t stride = 1;
  KernelIndexTypeCompute index_type_helper;
  for (auto i = tv->getMaybeRFactorDomain().size(); i > 0; --i) {
    auto id = tv->getMaybeRFactorDomain().at(i - 1);
    if (id->isReduction() || id->isStride() || id->isBroadcast()) {
      continue;
    }

    auto extent = ee.evaluate(id->extent());
    // We could also just conservatively use 64-bit indexing if the
    // extent size is not determined, but this should be possible to
    // evaluate.
    TORCH_INTERNAL_ASSERT(
        extent.has_value(),
        "Axis with unknown extent found: ",
        id->toString(),
        ", tensor: ",
        tv->toString());

    auto extent_int = extent->as<int64_t>();

    TORCH_INTERNAL_ASSERT(
        extent_int >= 0, "Unexpected size of axis: ", extent_int);

    if (extent_int > 0) {
      if (index_type_helper.addDim(extent->as<int64_t>(), stride) ==
          PrimDataType::Int) {
        return PrimDataType::Int;
      }
      stride *= extent->as<int64_t>();
    }
  }

  return index_type_helper.getType();
}

// Check inputs, outputs and intermediates
// Intermediates are contiguous, so strides are not necessary
// Strides are required for inputs and also maybe for outputs as
// they may be non-contiguous. However, in our current interface,
// output strides are not available, so if there's any outputs that
// are non contiguous, need to fall back to 64-bit indexing
PrimDataType getIndexTypeOfKernel(
    Fusion* fusion,
    const std::vector<TensorView*>& all_tvs,
    const KernelArgumentHolder& inputs,
    ExpressionEvaluator& ee) {
  if (inputs.getSmallestIndexTypeOfArguments() == PrimDataType::Int) {
    return PrimDataType::Int;
  }

  for (auto tv : all_tvs) {
    // Fusion input tensors are included in the args parameter, and
    // they are checked separately
    if (tv->isFusionInput()) {
      continue;
    }

    if (getTensorIndexType(tv, ee) == PrimDataType::Int) {
      return PrimDataType::Int;
    }
  }

  return PrimDataType::Int32;
}

} // namespace

SchedulerRuntimeInfo::SchedulerRuntimeInfo(
    Fusion* complete_fusion,
    KernelArgumentHolder args,
    PrecomputedValues* precomputed_values,
    const std::vector<TensorView*>& all_tvs,
    std::optional<PrimDataType> forced_index_type)
    : complete_fusion_(complete_fusion) {
  TORCH_INTERNAL_ASSERT(
      complete_fusion_->inputs().size() == args.size(),
      "Invalid number of arguments passed in for provided fusion group.");

  expression_evaluator_ = getExpressionEvaluator(args, precomputed_values);

  if (forced_index_type.has_value()) {
    index_type_ = forced_index_type.value();
  } else {
    index_type_ = getIndexTypeOfKernel(
        complete_fusion_,
        all_tvs.empty() ? ir_utils::allTvs(complete_fusion_) : all_tvs,
        args,
        *expression_evaluator_);
  }

  // Convert all abstract tensor args into tensor args and do tensor stride
  // inference
  std::vector<TensorView*> tvs;
  tvs.reserve(complete_fusion_->inputs().size());
  for (auto val : complete_fusion_->inputs()) {
    tvs.emplace_back(dynamic_cast<TensorView*>(val));
  }
  args.getBuffer(index_type_, tvs);

  for (auto inp_i : c10::irange(static_cast<int64_t>(args.size()))) {
    auto kernel_arg = args[inp_i];
    // Note: we are skipping CpuScalar tensor here
    if (auto tensor_arg_abstract =
            dynamic_cast<const TensorArgAbstract*>(kernel_arg)) {
      auto fusion_inp = complete_fusion_->inputs()[inp_i];
      auto data_ptr = tensor_arg_abstract->getPointer();
      input_ptrs_[fusion_inp] = (size_t)data_ptr;

      // find and push discontiguous stride
      auto dtype_size = dataTypeSize(tensor_arg_abstract->getDataType());
      input_discontig_strides_[fusion_inp] = {};
      auto dims = tensor_arg_abstract->getRank();
      int64_t expected_stride = 1;
      for (auto dim = dims - 1; dim >= 0; dim--) {
        auto size = tensor_arg_abstract->getSize((int)dim);
        if (size <= 1) {
          continue;
        }
        auto stride = tensor_arg_abstract->getStride((int)dim);
        if (stride != expected_stride) {
          input_discontig_strides_[fusion_inp].push_back(stride * dtype_size);
          expected_stride = stride;
        }
        expected_stride *= size;
      }
    }
  }
}

SchedulerRuntimeInfo::SchedulerRuntimeInfo(
    Fusion* complete_fusion,
    const at::ArrayRef<c10::IValue>& aten_inputs)
    : SchedulerRuntimeInfo(
          complete_fusion,
          KernelArgumentHolder::createKernelArgumentHolder(aten_inputs)) {}

// TODO: Output tensors could have an alignment that is not 16 Bytes passed in
// from user.
size_t SchedulerRuntimeInfo::ptrOf(TensorView* tv) const {
  if (input_ptrs_.find(tv) != input_ptrs_.end()) {
    return input_ptrs_.at(tv);
  }
  return max_alignment_size_in_byte;
}

std::unique_ptr<ExpressionEvaluator> SchedulerRuntimeInfo::
    getExpressionEvaluator(
        const KernelArgumentHolder& args,
        PrecomputedValues* precomputed_values) {
  std::unique_ptr<ExpressionEvaluator> ee =
      std::make_unique<ExpressionEvaluator>();
  if (precomputed_values) {
    ee->bindPrecomputedValues(precomputed_values);
  } else {
    *ee = executor_utils::bindInputs(args, complete_fusion_);
  }
  return ee;
}

size_t SchedulerRuntimeInfo::computeAlignmentSize(size_t ptr_address) {
  size_t alignment_size = 1;
  size_t next_alignment_size = 2;

  while (next_alignment_size <= max_alignment_size_in_byte &&
         ptr_address % next_alignment_size == 0) {
    alignment_size = next_alignment_size;
    next_alignment_size *= 2;
  }
  return alignment_size;
}

size_t SchedulerRuntimeInfo::getAlignmentSize(TensorView* tv) {
  auto alignment_entry = alignment_map_.find(tv);
  if (alignment_entry != alignment_map_.end()) {
    return alignment_entry->second;
  }

  auto alignment_size = SchedulerRuntimeInfo::computeAlignmentSize(ptrOf(tv));
  auto strides_it = input_discontig_strides_.find(tv);
  if (strides_it != input_discontig_strides_.end()) {
    for (auto stride : strides_it->second) {
      alignment_size = std::min(
          alignment_size, SchedulerRuntimeInfo::computeAlignmentSize(stride));
    }
  }
  alignment_map_[tv] = alignment_size;
  return alignment_size;
}

// Gets maximum vectorizable width of tv, assumes we can merge across all
// iteration domains if contiguous. Cannot permute the dimensions to fix
// contiguity.
size_t SchedulerRuntimeInfo::getMaxVectorizableWidth(TensorView* tv) {
  // Gets the vectorizable width of the tv starting from the inner most
  // dimension, working its way towards the outer most dimension, if they're
  // contiguous. Ignores broadcast and reduction domains.
  auto max_vectorword_map_it_ = max_vectorword_map_.find(tv);
  if (max_vectorword_map_it_ != max_vectorword_map_.end()) {
    return max_vectorword_map_it_->second;
  }

  // If we don't have an record, either it is a tv with innermost broadcast,
  // or it is an intermediate tensor allocated by fuser. Logic copied to get
  // root according to scheduler_utils::innerMostRootDim.
  auto tv_root = tv->hasReduction() && tv->hasRFactor()
      ? tv->getRootDomain()
      : tv->getMaybeRFactorDomain();

  auto tv_root_no_reductions = TensorDomain::noReductions(tv_root);

  auto contiguity = tv->domain()->contiguity();
  // Appears after reductions the reduction domain often has a contiguity entry.
  // This only matters if the result of the reduction is an output
  if (contiguity.size() == tv_root.size() &&
      contiguity.size() != tv_root_no_reductions.size()) {
    std::vector<std::optional<bool>> new_contiguity;
    for (auto i : c10::irange(tv_root.size())) {
      if (!tv_root[i]->isReduction()) {
        new_contiguity.push_back(contiguity[i]);
      }
    }
    contiguity = new_contiguity;
  }
  tv_root = tv_root_no_reductions;

  auto tv_root_size = tv_root.size();

  // Filter out 0-dim tensors
  if (tv_root_size < 1) {
    return 1;
  }

  // Filter out mismatched contiguity info
  if (tv_root_size != contiguity.size()) {
    return 1;
  }

  size_t item_size = dataTypeSize(tv->dtype(), getIndexType());

  // Alignment should always at least be the data type size
  TORCH_INTERNAL_ASSERT(getAlignmentSize(tv) % item_size == 0);
  size_t max_vector_size = getAlignmentSize(tv) / item_size;

  if (max_vector_size == 1) {
    return 1;
  }

  size_t numel = 1;
  for (auto i : c10::irange(tv_root_size)) {
    auto root_i = tv_root_size - i - 1;
    auto root_id = tv_root[root_i];

    if (root_id->extent()->isOneInt() || root_id->isBroadcast()) {
      continue;
    }

    // Not contiguous
    auto contiguity_opt = contiguity.at(root_i);
    TORCH_INTERNAL_ASSERT(contiguity_opt.has_value());
    if (!*contiguity_opt) {
      break;
    }

    auto dim_size = expression_evaluator_->evaluate(root_id->extent());
    // Inference failed for some reason, assume not-contiguous at this point
    if (!dim_size.has_value()) {
      break;
    }

    // Still contiguous
    numel *= dim_size->as<int64_t>();
  }

  // Assuming intermediate tensors have friendly alignment, and
  //  all contiguity true. Determine the largest power of 2 below
  //  innermost dimension size for the word size of vectorizaiton
  size_t vector_size = 1;
  size_t next_vector_size = 2;
  while (next_vector_size <= max_vector_size &&
         next_vector_size <= (size_t)numel && numel % next_vector_size == 0) {
    vector_size = next_vector_size;
    next_vector_size *= 2;
  }

  // save output to avoid re-compute
  max_vectorword_map_[tv] = vector_size;

  return vector_size;
}

// Gets the vectorizable width of the inner most dimension of tv if it's
// contiguous. Ignores inner most dimensions that are broadcast or reduction.
size_t SchedulerRuntimeInfo::getInnerDimVectorizableWidth(TensorView* tv) {
  auto inner_vectorword_map_it_ = inner_vectorword_map_.find(tv);
  if (inner_vectorword_map_it_ != inner_vectorword_map_.end()) {
    return inner_vectorword_map_it_->second;
  }

  // If we don't have an record, either it is a tv with innermost broadcast,
  // or it is an intermediate tensor allocated by fuser. Logic copied to get
  // root according to scheduler_utils::innerMostRootDim.
  auto tv_root = tv->hasReduction() && tv->hasRFactor()
      ? tv->getRootDomain()
      : tv->getMaybeRFactorDomain();

  auto tv_root_no_reductions = TensorDomain::noReductions(tv_root);

  auto contiguity = tv->domain()->contiguity();
  // Appears after reductions the reduction domain often has a contiguity entry.
  // This only matters if the result of the reduction is an output
  if (contiguity.size() == tv_root.size() &&
      contiguity.size() != tv_root_no_reductions.size()) {
    std::vector<std::optional<bool>> new_contiguity;
    for (auto i : c10::irange(tv_root.size())) {
      if (!tv_root[i]->isReduction()) {
        new_contiguity.push_back(contiguity[i]);
      }
    }
    contiguity = new_contiguity;
  }
  tv_root = tv_root_no_reductions;

  auto tv_root_no_reductions_size = tv_root_no_reductions.size();

  // Filter out 0-dim tensors
  if (tv_root_no_reductions_size < 1) {
    return 1;
  }

  // Filter out mismatched contiguity info
  if (tv_root_no_reductions_size != contiguity.size()) {
    return 1;
  }

  auto inner_most_dim = scheduler_utils::innerMostRootDim(tv);

  int id_pos = -1;
  for (auto root_i : c10::irange((int)tv_root_no_reductions_size)) {
    if (tv_root_no_reductions[root_i] == inner_most_dim) {
      id_pos = root_i;
      break;
    }
  }

  // Something went wrong with finding the inner most dimension, just
  // return 1.
  if (id_pos == -1) {
    return 1;
  }

  // If the inner most dimension is not contiguous return 1
  auto contiguity_opt = contiguity.at(id_pos);
  TORCH_INTERNAL_ASSERT(contiguity_opt.has_value());
  if (!*contiguity_opt) {
    return 1;
  }

  size_t item_size = dataTypeSize(tv->dtype(), getIndexType());

  // Alignment should always at least be the data type size
  TORCH_INTERNAL_ASSERT(getAlignmentSize(tv) % item_size == 0);
  size_t max_vector_size = getAlignmentSize(tv) / item_size;

  // Assuming intermediate tensors have friendly alignment, and
  //  all contiguity true. Determine the largest power of 2 below
  //  innermost dimension size for the word size of vectorizaiton
  size_t vector_size = 1;
  size_t next_vector_size = 2;
  auto maybe_inner_dimension_size =
      expression_evaluator_->evaluate(inner_most_dim->extent());
  TORCH_INTERNAL_ASSERT(maybe_inner_dimension_size.has_value());
  size_t inner_dimension_size = maybe_inner_dimension_size->as<int64_t>();

  while (next_vector_size <= max_vector_size &&
         next_vector_size <= inner_dimension_size &&
         inner_dimension_size % next_vector_size == 0) {
    vector_size = next_vector_size;
    next_vector_size *= 2;
  }

  // save output to avoid re-compute
  inner_vectorword_map_[tv] = vector_size;

  return vector_size;
}

bool SchedulerEntry::sameAs(const SchedulerEntry* other) {
  return heuristic_ == other->heuristic_ && params_->sameAs(other->params_);
}

namespace {

static bool checkPatternEquivalence(
    TensorView* out_tv0,
    TensorView* out_tv1,
    const ComputeAtRootDomainMap& root_map) {
  const auto& out_root0 = out_tv0->getRootDomain();
  const auto& out_root1 = out_tv1->getRootDomain();
  const auto domain0 = out_tv0->domain();
  const auto domain1 = out_tv1->domain();

  auto it0 = out_root0.begin();
  auto it1 = out_root1.begin();

  auto skip_broadcast = [&]() {
    while (it0 != out_root0.end() && (*it0)->isBroadcast()) {
      it0++;
    }
    while (it1 != out_root1.end() && (*it1)->isBroadcast()) {
      it1++;
    }
  };

  skip_broadcast();
  while (it0 != out_root0.end() && it1 != out_root1.end()) {
    if ((*it0)->isReduction() != (*it1)->isReduction()) {
      return false;
    }
    if (!root_map.canMap(domain0, (*it0), domain1, (*it1))) {
      return false;
    }
    it0++;
    it1++;
    skip_broadcast();
  }

  return it0 == out_root0.end() && it1 == out_root1.end();
}

// Reusing some code from lowering specifically in lower_trivial_broadcast.cpp
// ConcretizedBroadcastDomains::maybeNonUniquelyConcretized this checks if
// there's a broadcast iteration domain that's being broadcasted to seemingly
// different extents, meaning we don't know in the kernel if the dimension is
// being broadcasted to one size multiple times or different sizes. This is a
// hard to optimize problem and likely indicates we shouldn't be fusing.
bool hasNonUniqueBcast(Fusion* fusion) {
  ConcretizedBroadcastDomains concretize_info(fusion);

  for (auto tv : ir_utils::allTvs(fusion)) {
    for (auto id : tv->getRootDomain()) {
      if (concretize_info.maybeNonUniquelyConcretized(id)) {
        return true;
      }
    }
  }
  return false;
}

//! Scheduler interface:
//!    Each of the scheduler needs to provide 3 interface functions:
//!
//!      1. canScheduleCompileTime(Fusion* fusion) :
//!
//!        This function contains compiled-time checks on the graph itself
//!        without runtime input information. Only `fusion` is given in the
//!        argument to make sure only compile-time available info is needed in
//!        the check.
//!
//!        This function is to be called exactly once on each segmented group
//!        created in a segmented fusion so this part will not contribute to
//!        dynamic shape latency.
//!
//!     2. canScheduleRunTime(
//!            Fusion* fusion,
//!            SchedulerRuntimeInfo& runtime_info,
//!           HeuristicSummary* data_cache = nullptr):
//!        This function contains all canSchedule checks that will have to
//!        involve runtime input information, and will be run both by the
//!        segmenter and the kernel cache. The latency of this function will
//!        contribute to dynamic shape latency so `data_cache` should be used as
//!        much as possible to save re-computation.
//!
//!     3. schedule(fusion):
//!
//!        This function will be called when compiling a kernel. It should apply
//!        scheduling to the given fusion

//! NoOp scheduler represents the case where scheduler will
//!  not do any scheduling operations and forward the un-scheduled
//!  fusion directly to code generation and kernel compilation.
//!
//! Typical use case of this scheduler is to handle edge cases
//!  such as where all tensors are size-1 or size-0.
class NoOpScheduler : public SchedulerEntry {
  //! Provides a dummy heuristic type to ensure
  //!  unified interface on NoOp scheduler.
  class NoOpHeuristic : public HeuristicParams {
   public:
    using HeuristicParams::HeuristicParams;

    size_t hash() const override {
      return 0;
    }
    std::shared_ptr<HeuristicParams> clone() const override {
      return std::make_shared<NoOpHeuristic>();
    }
    bool sameAs(const std::shared_ptr<HeuristicParams>& other) const override {
      auto other_casted = std::dynamic_pointer_cast<ReductionParams>(other);
      return other_casted != nullptr && other_casted->cparams == cparams;
    };
  };

 public:
  explicit NoOpScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::NoOp) {
    params_ = std::make_shared<NoOpHeuristic>("", runtime_info.getIndexType());
  }

  //! Check if the no-op heuristics apply in given fusion
  static bool canScheduleCompileTime(Fusion* fusion) {
    if (fusion->isNoOp()) {
      return true;
    }
    // Check there're no non-trivial reduction ops.
    for (auto reduction : ir_utils::getReductionOps(fusion)) {
      for (auto output :
           ir_utils::filterByType<TensorView>(reduction->outputs())) {
        auto concrete_dimension =
            TensorDomain::noReductions(output->getRootDomain());
        auto all_nonzero = std::none_of(
            concrete_dimension.begin(),
            concrete_dimension.end(),
            [](IterDomain* id) { return id->extent()->isZeroInt(); });
        if (all_nonzero) {
          scheduler_debug_utils::canScheduleRejectReason(
              ScheduleHeuristic::NoOp,
              "reduction of non-zero elements is not supported");
          return false;
        }
      }
    }

    // Check that all outputs are either broadcast or ignored reduction.
    for (auto out_tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
      auto concrete_dimension = TensorDomain::noReductions(
          TensorDomain::noBroadcasts(out_tv->getLeafDomain()));
      if (!concrete_dimension.empty()) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::NoOp, "output has a concrete dimension");
        return false;
      }
    }

    // Check that inputs of all select/gather-like ops are fusion inputs
    if (rejectScheduleForSelectLikeOps(fusion, ScheduleHeuristic::NoOp)) {
      return false;
    }

    // We have verified that all iterdomains on all output tv's are trivial
    // reductions,
    //  broadcasts or zero-sized. Therefore accepting this fusion for NoOp
    //  scheduling.
    return true;
  }

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    // TODO:
    //  Pipe through dynamic zero checks.
    return true;
  }

  void schedule(Fusion* fusion) override {
    // Schedule is no-op.
    return;
  }

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    // Heuristics is no-op.
    return;
  }
};

class ReductionScheduler : public SchedulerEntry {
 public:
  explicit ReductionScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::Reduction) {
    computeHeuristics(fusion, runtime_info, data_cache);
  }

  //! Check if the reduction heuristics apply in given fusion
  static bool canScheduleCompileTime(Fusion* fusion) {
    // Needs at least one reduction to consider.
    if (ir_utils::getReductionOps(fusion).empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction, "No reduction op to schedule");
      return false;
    }

    if (ir_utils::filterByType<TensorView>(fusion->inputs()).empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction,
          "Scheduling not supported with no input");
      return false;
    }

    // Check that inputs of all select/gather-like ops are fusion inputs
    if (rejectScheduleForSelectLikeOps(fusion, ScheduleHeuristic::Reduction)) {
      return false;
    }

    // Fusions handled by reduction scheduler cannot have MmaOp.
    if (!ir_utils::getMmaOps(fusion).empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction, "no support for mma ops.");
      return false;
    }

    auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);

    if (reduction_tvs.empty()) {
      // Use pointwise logic
      return false;
    }

    if (hasNonUniqueBcast(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction,
          "Broadcasting dimension might be broadcasting to multiple sizes.");
      return false;
    }

    if (!ir_utils::getViewOps(fusion).empty()) {
      ComputeAtMap ca_map(fusion);
      if (requiresForwardViewReplay(fusion, ca_map)) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Reduction,
            "Fusion requires view being reversible.");
        return false;
      }

      // Reduction scheduler simply uses reduction_tvs[0] as the reference, if
      // that changes, this needs to be changed.
      if (reductionInterferingView(fusion, ca_map, reduction_tvs[0])) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Reduction,
            "View may interfere with reduction scheduling.");
        return false;
      }
    }

    // Make sure reduction axes are consistent through the fusion
    auto reduction_ops = ir_utils::getReductionOps(fusion);
    if (reduction_ops.size() > 1) {
      // Before examining the reduction axes want to quickly
      //   check the reductions have the same axis width
      //   to avoid building root domain map in easier cases
      bool valid_axis_count = false;
      size_t axis_count = 0;
      auto reduction_root_size = [](TensorView* red_tv) {
        size_t count = 0;
        for (auto id : red_tv->getRootDomain()) {
          if (!id->isBroadcast()) {
            count++;
          }
        }
        return count;
      };

      for (auto red : reduction_tvs) {
        if (!valid_axis_count) {
          valid_axis_count = true;
          axis_count = reduction_root_size(red);
        } else {
          if (reduction_root_size(red) != axis_count) {
            scheduler_debug_utils::canScheduleRejectReason(
                ScheduleHeuristic::Reduction,
                "Inconsistent reduction axes ",
                red,
                "is not ",
                axis_count);
            return false;
          }
        }
      }

      // Use root domain map to check the reduction ops have the same axes
      FusionGuard fg(fusion);
      ComputeAtRootDomainMap root_map;
      root_map.build(true);

      // red_ops.size()>1 checked before
      for (size_t it = 1; it < reduction_tvs.size(); it++) {
        if (!checkPatternEquivalence(
                reduction_tvs[it - 1], reduction_tvs[it], root_map)) {
          scheduler_debug_utils::canScheduleRejectReason(
              ScheduleHeuristic::Reduction,
              "Un-mapped multi-reduction: ",
              reduction_tvs[it - 1],
              " ",
              reduction_tvs[it]);
          return false;
        }
      }
    }

    // Doesn't allow persistent kernels in this scheduler
    auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
    if (!persistent_buffer_info.persistent_buffers.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction,
          "need persistent buffers that reduction scheduler doesn't handle");
      return false;
    }

    if (!SchedulerTopologyChecker::supportedPostReductionFusion(
            fusion, reduction_tvs) ||
        SchedulerTopologyChecker::hasPostReductionBCast(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction,
          "has unsupported post reduction fusion");
      return false;
    }

    if (SchedulerTopologyChecker::hasGatherToBroadcastBeforeReduction(
            fusion, reduction_tvs)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction,
          "has unsupported gather-like ops before reduction");
      return false;
    }

    return true;
  }

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    return true;
  }

  void schedule(Fusion* fusion) override {
    FUSER_PERF_SCOPE("Schedule Single Reduction");
    scheduleReduction(fusion, reductionParams());
  }

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    params_ = getReductionHeuristics(fusion, runtime_info, data_cache);
    TORCH_INTERNAL_ASSERT(params_ != nullptr);
  }
};

class TransposeScheduler : public SchedulerEntry {
 public:
  explicit TransposeScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::Transpose) {
    computeHeuristics(fusion, runtime_info, data_cache);
  }

  static bool canScheduleCompileTime(Fusion* fusion) {
    // Temporarily disallow view in transpose scheduler
    // TODO Add more testing before enabling
    auto view_tvs = scheduler_utils::getViewTVs(fusion);
    if (!view_tvs.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Transpose, "No support for view op");
      return false;
    }

    // Check that inputs of all select/gather-like ops are fusion inputs
    if (rejectScheduleForSelectLikeOps(fusion, ScheduleHeuristic::Transpose)) {
      return false;
    }

    // Fusions handled by transpose scheduler cannot have MmaOp.
    if (!ir_utils::getMmaOps(fusion).empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Transpose, "no support for mma ops.");
      return false;
    }

    for (auto select : ir_utils::getSelectOps(fusion)) {
      auto root = TensorDomain::noReductions(
          select->input(0)->as<TensorView>()->getMaybeRFactorDomain());
      if (select->getIndexedID() == root[root.size() - 1]) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Transpose,
            "SelectOp on inner dim is not supported by transpose scheduler yet."
            "In transpose scheduler, we want to leave the select dim alone, instead of creating a tile for it.");
        return false;
      }
    }
    for (auto idx_sel : ir_utils::getIndexSelectOps(fusion)) {
      auto root = TensorDomain::noReductions(
          idx_sel->input(0)->as<TensorView>()->getMaybeRFactorDomain());
      if (idx_sel->getIndexedID() == root[root.size() - 1]) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Transpose,
            "IndexSelectOp on inner dim is not supported by transpose scheduler yet."
            "In transpose scheduler, we want to leave the select dim alone, instead of creating a tile for it.");
        return false;
      }
    }
    for (auto torch_gather : ir_utils::getTorchGatherOps(fusion)) {
      auto root = TensorDomain::noReductions(
          torch_gather->input(0)->as<TensorView>()->getMaybeRFactorDomain());
      if (torch_gather->dim() == (int)root.size() - 1) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Transpose,
            "TorchGatherOp on inner dim is not supported by transpose scheduler yet."
            "In transpose scheduler, we want to leave the select dim alone, instead of creating a tile for it.");
        return false;
      }
    }

    if (!hasAtLeastTwoValidGroups(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Transpose,
          "cannot find two mismatching inner most dimensions");
      return false;
    }

    auto reduction_ops = ir_utils::getReductionOps(fusion);

    if (!reduction_ops.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Transpose, "no support for reduction ops");
      return false;
    }

    if (hasNonUniqueBcast(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Transpose,
          "Broadcasting dimension might be broadcasting to multiple sizes.");
      return false;
    }

    return true;
  }

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    FUSER_PERF_SCOPE("TransposeScheduler::canScheduleRunTime");

    auto reason =
        getTransposeRuntimeRejectReason(fusion, data_cache, runtime_info);
    if (!reason.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Transpose, reason);
      return false;
    }
    return true;
  }

  void schedule(Fusion* fusion) override {
    FUSER_PERF_SCOPE("Schedule Transpose Fusion");
    scheduleTranspose(fusion, transposeParams());
  }

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    params_ = getTransposeHeuristics(fusion, runtime_info, data_cache);
    TORCH_INTERNAL_ASSERT(params_ != nullptr);
  }
};

class PointWiseScheduler : public SchedulerEntry {
 public:
  explicit PointWiseScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::PointWise) {
    computeHeuristics(fusion, runtime_info, data_cache);
  }

  static bool canScheduleCompileTime(Fusion* fusion) {
    //   Currently using the same path as the scheduler
    // to eliminate mismatch between canSchedule and
    // schedule pointwise.
    if (!hasReferenceTensorView(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::PointWise, "cannot find reference tensor");
      return false;
    }

    // Check that inputs of all select/gather-like ops are fusion inputs
    if (rejectScheduleForSelectLikeOps(fusion, ScheduleHeuristic::PointWise)) {
      return false;
    }

    // Fusions handled by pointwise scheduler cannot have MmaOp.
    if (!ir_utils::getMmaOps(fusion).empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::PointWise, "no support for mma ops.");
      return false;
    }

    if (!ir_utils::getViewOps(fusion).empty()) {
      ComputeAtMap ca_map(fusion);
      if (requiresForwardViewReplay(fusion, ca_map)) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::PointWise,
            "Fusion requires view being reversible.");
        return false;
      }
    }

    auto reduction_ops = ir_utils::getReductionOps(fusion);

    if (!reduction_ops.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::PointWise, "no support for reduction ops");
      return false;
    }

    if (hasNonUniqueBcast(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::PointWise,
          "Broadcasting dimension might be broadcasting to multiple sizes.");
      return false;
    }

    return true;
  }

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    auto can_schedule_transpose_entry =
        HeuristicSummaryEntry<HeuristicCompileTime::CanScheduleTranspose>(
            data_cache, [fusion]() {
              return std::make_unique<bool>(
                  TransposeScheduler::canScheduleCompileTime(fusion));
            });
    if (can_schedule_transpose_entry.get()) {
      auto reason =
          getTransposeRuntimeRejectReason(fusion, data_cache, runtime_info);
      return !reason.empty();
    }

    return true;
  }

  void schedule(Fusion* fusion) override {
    FUSER_PERF_SCOPE("Schedule PointWise Fusion");
    schedulePointwise(fusion, pointwiseParams());
  }

  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    params_ = getPointwiseHeuristics(fusion, runtime_info, data_cache);
    TORCH_INTERNAL_ASSERT(params_ != nullptr);
  }
};

class PersistentKernelScheduler : public SchedulerEntry {
 public:
  explicit PersistentKernelScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::Persistent) {
    computeHeuristics(fusion, runtime_info, data_cache);
  }

  void schedule(Fusion* fusion) override {
    FUSER_PERF_SCOPE("Schedule Persistent Fusion");
    schedulePersistentKernel(fusion, reductionParams());
  }

  static bool canScheduleCompileTime(Fusion* fusion) {
    // Needs at least one reduction to consider.
    auto reduction_ops = ir_utils::getReductionOps(fusion);
    if (reduction_ops.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "needs a reduction op");
      return false;
    }

    if (ir_utils::filterByType<TensorView>(fusion->inputs()).empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "Scheduling not supported with no input");
      return false;
    }

    // Check that inputs of all select/gather-like ops are fusion inputs
    if (rejectScheduleForSelectLikeOps(fusion, ScheduleHeuristic::Persistent)) {
      return false;
    }

    // Fusions handled by persistent kernel scheduler cannot have MmaOp.
    if (!ir_utils::getMmaOps(fusion).empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "no support for mma ops.");
      return false;
    }

    if (hasNonUniqueBcast(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "Broadcasting dimension might be broadcasting to multiple sizes.");
      return false;
    }

    auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);

    if (reduction_tvs.empty()) {
      // Use pointwise logic
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "no reduction tv");
      return false;
    }

    std::vector<TensorView*> inner_reduction_tvs;
    std::vector<TensorView*> outer_reduction_tvs;
    for (auto tv : reduction_tvs) {
      if (scheduler_utils::isFastestDimReduction(tv)) {
        inner_reduction_tvs.emplace_back(tv);
      } else {
        outer_reduction_tvs.emplace_back(tv);
      }
    }
    bool combined_inner_outer =
        !inner_reduction_tvs.empty() && !outer_reduction_tvs.empty();
    if (!checkReductionPattern(
            fusion, inner_reduction_tvs, outer_reduction_tvs)) {
      return false;
    }
    // If there is both inner and outer reduction, we use the first inner
    // reduction tv as reference, otherwise we use the first reduction tv,
    // whether it is inner or outer.
    TensorView* reference_tv =
        combined_inner_outer ? inner_reduction_tvs[0] : reduction_tvs[0];

    if (!ir_utils::getViewOps(fusion).empty()) {
      ComputeAtMap ca_map(fusion);
      if (requiresForwardViewReplay(fusion, ca_map)) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Persistent,
            "Fusion requires view being reversible.");
        return false;
      }

      // Persistent scheduler simply uses reference_tv as the reference, if
      // that changes, this needs to be changed.
      if (reductionInterferingView(fusion, ca_map, reference_tv)) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Persistent,
            "View may interfere with normalization scheduling.");
        return false;
      }
    }

    // Before examining the reduction axes want to quickly
    //   check the reductions have the same axis width
    //   to avoid building root domain map in easier cases
    bool valid_axis_count = false;
    size_t axis_count = 0;
    auto reduction_root_size = [](TensorView* red_tv) {
      size_t count = 0;
      for (auto id : red_tv->getRootDomain()) {
        if (!id->isBroadcast()) {
          count++;
        }
      }
      return count;
    };

    for (auto red : reduction_tvs) {
      if (!valid_axis_count) {
        valid_axis_count = true;
        axis_count = reduction_root_size(red);
      } else {
        if (reduction_root_size(red) != axis_count) {
          scheduler_debug_utils::canScheduleRejectReason(
              ScheduleHeuristic::Persistent,
              "inconsistent reduction root size: ",
              red->toString(),
              ", expected: ",
              axis_count);
          return false;
        }
      }
    }

    // Only accept persistent kernels
    auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
    if (persistent_buffer_info.persistent_buffers.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "no persistent buffer identified");
      return false;
    }

    if (SchedulerTopologyChecker::hasNonNormalizePostReductionBCast(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "unsupported post reduction normalization");
      return false;
    }

    if (SchedulerTopologyChecker::hasGatherToBroadcastBeforeReduction(
            fusion, reduction_tvs)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "has unsupported gather-like ops before normalization");
      return false;
    }

    return true;
  }

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    FUSER_PERF_SCOPE("PersistentKernelScheduler::canSchedule");
    auto reduction_tv_entry =
        HeuristicSummaryEntry<HeuristicCompileTime::ReductionTVs>(
            data_cache, [&fusion]() {
              return std::make_unique<std::vector<TensorView*>>(
                  scheduler_utils::getReductionTvs(fusion));
            });

    auto& reduction_tvs = reduction_tv_entry.get();
    bool inner_reduction = false;
    bool outer_reduction = false;
    TensorView* first_inner_reduction_tv = nullptr;
    for (auto tv : reduction_tvs) {
      if (scheduler_utils::isFastestDimReduction(tv)) {
        first_inner_reduction_tv = tv;
        inner_reduction = true;
      } else {
        outer_reduction = true;
      }
    }
    if (inner_reduction && outer_reduction) {
      if (!checkCombinedReductionShape(runtime_info, reduction_tvs)) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Persistent,
            "Inner dim of combined reduction should be a multiplication of a quarter warp and max vectorization factor!");
        return false;
      }
    }
    // If there is both inner and outer reduction, we use the first inner
    // reduction tv to get properties, otherwise we use the first reduction tv,
    // whether it is inner or outer.
    auto reference_tv = inner_reduction && outer_reduction
        ? first_inner_reduction_tv
        : reduction_tvs[0];

    auto properties =
        scheduler_utils::getProperties(fusion, runtime_info, reference_tv);

    if (!properties.fastest_dim_reduction) {
      return canScheduleRunTimeOuter(
          fusion, runtime_info, data_cache, reduction_tvs, properties);
    }

    // pair of persistent_buffer_size and available_persistent_buffer_size
    const std::pair<int64_t, int64_t> buffer_size = getPersistentBufferSize(
        fusion, runtime_info, data_cache, reduction_tvs);
    const int64_t persistent_buffer_size = buffer_size.first;
    const int64_t available_persistent_buffer_size = buffer_size.second;

    const int64_t device_multiprocessor_count =
        (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    if (persistent_buffer_size > available_persistent_buffer_size) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "not enough registers for persistece");
      return false;
    }

    const int64_t device_max_threads_per_multiprocessor =
        (int64_t)at::cuda::getCurrentDeviceProperties()
            ->maxThreadsPerMultiProcessor;

    const int64_t warp_size = at::cuda::warp_size();

    // Maximum number of iteration dimensions we can have and still be
    // persistent.
    const int64_t max_multi_reduction_factor = scheduler_utils::safeDiv(
        available_persistent_buffer_size, persistent_buffer_size);

    const int64_t required_sm_per_norm =
        ceilDiv(persistent_buffer_size, scheduler_utils::register_file_size);

    // If the persistence requires over half the device don't do grid
    // persistence as we can't overlap the grid comms.
    if (required_sm_per_norm >
        scheduler_utils::safeDiv(device_multiprocessor_count, 3)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "requires over half GPU persistence.");
      return false;
    }

    const int64_t norm_per_sm =
        ceilDiv(scheduler_utils::register_file_size, persistent_buffer_size);

    // If outer reduction, don't go persistent if we can't fit half a warp in
    // the iter domain of the persistent reduction.
    if (!properties.fastest_dim_reduction &&
        !(norm_per_sm >= warp_size / 2 ||
          max_multi_reduction_factor >= warp_size)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "not enough threads");
      return false;
    }

    // Don't go persistent if we can't use a small fraction of the
    // available SMs yet have a large reduction size.
    if ( // Large reduction dim
        properties.total_reduction_numel >=
            device_max_threads_per_multiprocessor * 4 &&
        properties.total_iteration_numel <
            (properties.fastest_dim_reduction
                 ? scheduler_utils::safeDiv(device_multiprocessor_count, 8)
                 // Make sure we at least use a quarter of the device * a
                 // half warp
                 : (warp_size / 8) * device_multiprocessor_count)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "not enough blocks");
      return false;
    }

    return true;
  }

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    params_ = getPersistentHeuristics(fusion, runtime_info, data_cache);
    TORCH_INTERNAL_ASSERT(params_ != nullptr);
  }

  static bool checkReductionPattern(
      Fusion* fusion,
      const std::vector<TensorView*>& inner_reduction_tvs,
      const std::vector<TensorView*>& outer_reduction_tvs) {
    // Use root domain map to check the reduction ops have the same axes
    FusionGuard fg(fusion);
    ComputeAtRootDomainMap root_map;
    root_map.build(true);

    // check inner and outer reductions seperately
    for (const auto& rtvs : {inner_reduction_tvs, outer_reduction_tvs}) {
      for (const auto it : c10::irange(1, rtvs.size())) {
        if (!checkPatternEquivalence(rtvs[it - 1], rtvs[it], root_map)) {
          scheduler_debug_utils::canScheduleRejectReason(
              ScheduleHeuristic::Persistent,
              "unmapped reduction ",
              rtvs[it - 1],
              " and ",
              rtvs[it]);
          return false;
        }
      }
    }
    // combined inner and outer reduction is of general purpose but only tested
    // for layer norm backward
    if (!inner_reduction_tvs.empty() && !outer_reduction_tvs.empty()) {
      if (!normalization_scheduler_utils::checkIfReductionsAreInnerOuter(
              inner_reduction_tvs, outer_reduction_tvs)) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Persistent,
            "to use combined reduction, inner reduction tensor should be [I,I,...,R,R] and outer reduction tensor should be [R,R,...,I,I]");
        return false;
      }

      if (!normalization_scheduler_utils::hasSharedInput(
              inner_reduction_tvs, outer_reduction_tvs)) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Persistent,
            "to use combined reduction, inner reduction and outer reduction should have shared input.");
        return false;
      }

      if (!normalization_scheduler_utils::
              isConnectedOnlyThroughReductionProducer(
                  inner_reduction_tvs, outer_reduction_tvs)) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Persistent,
            "to use combined reduction, inner reduction and outer reduction should not have shared consumer, their consumers should not have shared non-outer-reduction producer.");
        return false;
      }
    }
    return true;
  }

  static bool checkCombinedReductionShape(
      SchedulerRuntimeInfo& runtime_info,
      const std::vector<TensorView*>& reduction_tvs) {
    // In combined_inner_outer_reduction, the inner dim should be a
    // multiplication of a quarter warp and vectorization factor. Otherwise,
    // will use segregated version. Since inner reduction dim is splitted by
    // bdimx, this ensures the largest possible bdimx can be at least of a
    // quarter warp. So we have enough bdimx threads to cover the iteration
    // domain of the outer reductions to avoid low performance.
    const int64_t quarter_warp =
        at::cuda::getCurrentDeviceProperties()->warpSize / 4;
    for (auto tv : reduction_tvs) {
      int64_t n_elements = 1;
      const int64_t vectorization_factor = 16 /
          (int64_t)dataTypeSize(tv->getDataType().value(),
                                runtime_info.getIndexType());
      const int64_t n_elements_factor = quarter_warp * vectorization_factor;
      const bool is_inner_reduction =
          scheduler_utils::isFastestDimReduction(tv);
      for (auto id : tv->getMaybeRFactorDomain()) {
        // check reduction domain for inner reduction and iteration domain for
        // outer reduction
        if (id->isReduction() == is_inner_reduction) {
          auto id_size =
              runtime_info.expressionEvaluator().evaluate(id->extent());
          TORCH_INTERNAL_ASSERT(
              id_size.has_value(), "Could not infer reduction dim size.");
          n_elements *= id_size->as<int64_t>();
        }
      }
      if (n_elements % n_elements_factor) {
        return false;
      }
    }
    return true;
  }

  static std::pair<int64_t, int64_t> getPersistentBufferSize(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache,
      const std::vector<TensorView*>& reduction_tvs) {
    auto persistent_buffer_info_entry =
        HeuristicSummaryEntry<HeuristicCompileTime::PersistentBufferInfo>(
            data_cache, [&fusion]() {
              return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                  scheduler_utils::persistentBuffers(fusion));
            });

    auto& persistent_buffer_info = persistent_buffer_info_entry.get();

    auto persistent_buffer_size_info = scheduler_utils::persistentBufferSize(
        fusion, runtime_info, persistent_buffer_info, data_cache);

    // Note that projected buffer size can be zero
    auto persistent_buffer_size =
        persistent_buffer_size_info.projected_persistent_buffer_size == 0
        ? persistent_buffer_size_info.persistent_buffer_size
        : std::min(
              persistent_buffer_size_info.persistent_buffer_size,
              persistent_buffer_size_info.projected_persistent_buffer_size);

    // in combined_inner_outer_reduction, the partial results of outer
    // reductions must be persistent, allow register spill avoid segmentation
    int64_t inner_reduction_count = 0;
    int64_t outer_reduction_count = 0;
    std::vector<TensorView*> outer_reduction_tvs;
    for (auto tv : reduction_tvs) {
      if (scheduler_utils::isFastestDimReduction(tv)) {
        inner_reduction_count++;
      } else {
        outer_reduction_count++;
        outer_reduction_tvs.emplace_back(tv);
      }
    }
    const bool combined_inner_outer_reduction =
        inner_reduction_count && outer_reduction_count;
    if (combined_inner_outer_reduction) {
      persistent_buffer_size +=
          normalization_scheduler_utils::partialReductionBufferSize(
              outer_reduction_tvs, runtime_info);
    }
    // At this point, we use the full register file size only for the
    // inner-outer case. It does not mean the full size shouldn't be used
    // otherwise, but more detailed tuning of the heuristics would be required.
    const int64_t available_persistent_buffer_size =
        combined_inner_outer_reduction
        ? scheduler_utils::register_file_size_full
        : scheduler_utils::register_file_size;

    return std::make_pair(
        persistent_buffer_size, available_persistent_buffer_size);
  }

  static bool canScheduleRunTimeOuter(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache,
      const std::vector<TensorView*>& reduction_tvs,
      const scheduler_utils::TvProperties& properties) {
    FUSER_PERF_SCOPE("PersistentKernelScheduler::canScheduleRuntimeOuter");
    FusionGuard fg(fusion);

    const auto device_prop = at::cuda::getCurrentDeviceProperties();

    const int64_t sm_register_file_size =
        static_cast<int64_t>(device_prop->regsPerBlock * sizeof(int));

    auto persistent_buffer_info_entry =
        HeuristicSummaryEntry<HeuristicCompileTime::PersistentBufferInfo>(
            data_cache, [&fusion]() {
              return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                  scheduler_utils::persistentBuffers(fusion));
            });

    const auto& persistent_buffer_info = persistent_buffer_info_entry.get();

    auto persistent_buffer_size_info = scheduler_utils::persistentBufferSize(
        fusion, runtime_info, persistent_buffer_info, data_cache);

    // Note that projected buffer size can be zero
    auto persistent_buffer_size =
        persistent_buffer_size_info.projected_persistent_buffer_size == 0
        ? persistent_buffer_size_info.persistent_buffer_size
        : std::min(
              persistent_buffer_size_info.persistent_buffer_size,
              persistent_buffer_size_info.projected_persistent_buffer_size);

    const int64_t device_multiprocessor_count =
        (int64_t)device_prop->multiProcessorCount;

    const auto available_persistent_buffer_size =
        sm_register_file_size * device_multiprocessor_count;

    if (persistent_buffer_size > available_persistent_buffer_size) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "not enough registers for persistence");
      return false;
    }

    const int64_t vectorization_factor =
        (int64_t)vectorize_helper::getVectorizationFactor(
            runtime_info,
            reduction_tvs.at(0),
            data_cache,
            (int)reduction_tvs.at(0)->nDims() -
                (int)properties.inner_most_dimension_ndims);

    // Minimum required multi reduction factor.
    const int64_t min_multi_reduction_factor = vectorization_factor *
        normalization_scheduler_utils::PreferredLaunchConfig::kMinBdimx;

    const int64_t required_sm_per_norm = ceilDiv(
        persistent_buffer_size * min_multi_reduction_factor,
        sm_register_file_size);

    // If the persistence requires over half the device don't do grid
    // persistence as we can't overlap the grid comms.
    if (required_sm_per_norm >
        scheduler_utils::safeDiv(device_multiprocessor_count, 2)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "requires over half GPU persistence.",
          " required SMs per normalization: ",
          required_sm_per_norm);
      return false;
    }

    const bool is_cross_grid = required_sm_per_norm > 1;

    std::optional<normalization_scheduler_utils::GridOuterNormalizationParams>
        cross_grid_params;

    if (is_cross_grid) {
      // Don't try to be persistent unless at least 4-way vectorized
      // as register usage is hard to control
      // TODO: Is this necessary for block persistence as well?
      if (vectorization_factor < 4) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Persistent, "not enough vectorized");
        return false;
      }

      // Make sure there's a valid grid persistence launch config
      cross_grid_params =
          normalization_scheduler_utils::getGridOuterNormalizationParams(
              properties.total_reduction_numel,
              properties.total_iteration_numel,
              vectorization_factor,
              persistent_buffer_size);

      if (!cross_grid_params.has_value()) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Persistent, "no valid launch config found");
        return false;
      }
    }

    TORCH_INTERNAL_ASSERT(!is_cross_grid || cross_grid_params.has_value())

    // Maximum number of iteration dimensions we can have and still be
    // persistent.
    const int64_t max_multi_reduction_factor = scheduler_utils::safeDiv(
        is_cross_grid ? available_persistent_buffer_size
                      : sm_register_file_size,
        persistent_buffer_size);

    // Don't go persistent if we can't fit the minimum multi reduction
    // factor
    if (max_multi_reduction_factor < min_multi_reduction_factor) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "Not enough threads.",
          " Multi reduction factor, ",
          max_multi_reduction_factor,
          ", is smaller than minimum multi reduction factor, ",
          min_multi_reduction_factor);
      return false;
    }

    const int64_t max_used_sms = is_cross_grid
        ? ceilDiv(
              ceilDiv(properties.total_iteration_numel, vectorization_factor),
              cross_grid_params->launch_params.bdimx()) *
            cross_grid_params->launch_params.gdimy()
        : ceilDiv(
              properties.total_iteration_numel * persistent_buffer_size,
              sm_register_file_size);

    // Bandwidth suffers if the number of used SMs is small. This is
    // particularly impactful in the case of cross grid, so at least
    // half of the SMs are required to be used. In the case of cross
    // block, keep using the existing heuristics for now.
    if (is_cross_grid &&
        max_used_sms <
            scheduler_utils::safeDiv(device_multiprocessor_count, 2)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent,
          "cross grid - not enough used SMs: ",
          max_used_sms);
      return false;
    }

    const int64_t device_max_threads_per_multiprocessor =
        (int64_t)device_prop->maxThreadsPerMultiProcessor;
    const int64_t min_fraction_of_sms =
        scheduler_utils::safeDiv(device_multiprocessor_count, 8);
    if (properties.total_reduction_numel >=
            device_max_threads_per_multiprocessor * 4 && // Large reduction dim
        max_used_sms < min_fraction_of_sms) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "not enough used SMs");
      return false;
    }

    // The runtime kernel for grouped normal grid reductions is not
    // well tuned, and it turned out to be quite difficult to get
    // consistently better performances than non-persistent
    // schedules. Disabled for now.
    // TODO: Enable non-welford persistent reductions
    if (is_cross_grid &&
        std::any_of(
            reduction_tvs.begin(),
            reduction_tvs.end(),
            [](TensorView* reduction_tv) {
              return !reduction_tv->definition()->isA<WelfordOp>();
            })) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "non-Welford not enabled yet");
      return false;
    }

    // Had a hard time tuning on Titan RTX and V100 when the iteration
    // space is not evenly divided by threads and thread blocks. It
    // doesn't seem to be noticeably bad on A100, though. For now,
    // disable the schedule if not evenly divisible on Titan RTX and
    // V100, i.e., compute architecture version 7.
    // TODO: Revisit
    if (is_cross_grid &&
        (properties.total_iteration_numel %
             (vectorization_factor * cross_grid_params->launch_params.bdimx() *
              cross_grid_params->launch_params.gdimx()) !=
         0) &&
        device_prop->major == 7) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Persistent, "iteration not evenly divided");
      return false;
    }

    return true;
  }
};

class MatmulScheduler : public SchedulerEntry {
 public:
  explicit MatmulScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::Matmul) {
    computeHeuristics(fusion, runtime_info);
  }

  void schedule(Fusion* fusion) override {
    FUSER_PERF_SCOPE("Schedule Matmul Fusion");
    scheduleMatmul(fusion, matmulParams());
  }

  static bool canScheduleCompileTime(Fusion* fusion) {
    const auto msg = getMatmulCompileTimeRejectReason(fusion);
    if (!msg.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Matmul, msg);
      return false;
    }

    return true;
  }

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    FUSER_PERF_SCOPE("MatmulScheduler::canSchedule");
    auto reason =
        getMatmulRunTimeRejectReason(fusion, data_cache, runtime_info);
    if (!reason.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Matmul, reason);
      return false;
    }
    return true;
  }

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    params_ = getMatmulHeuristics(fusion, runtime_info, data_cache);
    TORCH_INTERNAL_ASSERT(params_ != nullptr);
  }
};

// Schedule Table
const std::vector<ScheduleHeuristic>& all_heuristics() {
  static const std::vector<ScheduleHeuristic> hlist = {
      ScheduleHeuristic::NoOp,
      ScheduleHeuristic::Reduction,
      ScheduleHeuristic::Transpose,
      ScheduleHeuristic::PointWise,
      ScheduleHeuristic::Persistent,
      ScheduleHeuristic::Matmul};
  return hlist;
}

//! A Utility for checking both dynamic and static part of
//!  can schedule
template <typename SchedulerType>
bool checkCanSchedule(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache = nullptr) {
  // If a data cache is given, the compile time part doesn't need to be checked,
  // since for all current use cases
  //  it has to pass all the compile time checks to create a data cache for this
  //  fusion.
  if (!data_cache) {
    if (!isConnectedFusionGraph(fusion)) {
      return false;
    }
    if (IterDomainGraph(fusion, /*allow_self_mapping=*/true).hasSelfMapping()) {
      return false;
    }
    if (!SchedulerType::canScheduleCompileTime(fusion)) {
      return false;
    }
  }

  return SchedulerType::canScheduleRunTime(fusion, runtime_info, data_cache);
}

} // namespace

// Simple dispatcher interface
bool SchedulerEntry::canSchedule(
    ScheduleHeuristic sh,
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  switch (sh) {
    case ScheduleHeuristic::NoOp:
      return checkCanSchedule<NoOpScheduler>(fusion, runtime_info, data_cache);
    case ScheduleHeuristic::PointWise:
      return checkCanSchedule<PointWiseScheduler>(
          fusion, runtime_info, data_cache);
    case ScheduleHeuristic::Reduction:
      return checkCanSchedule<ReductionScheduler>(
          fusion, runtime_info, data_cache);
    case ScheduleHeuristic::Persistent:
      return checkCanSchedule<PersistentKernelScheduler>(
          fusion, runtime_info, data_cache);
    case ScheduleHeuristic::Transpose:
      return checkCanSchedule<TransposeScheduler>(
          fusion, runtime_info, data_cache);
    case ScheduleHeuristic::Matmul:
      return checkCanSchedule<MatmulScheduler>(
          fusion, runtime_info, data_cache);
    default:
      TORCH_INTERNAL_ASSERT(false, "unreachable");
      return false;
  }
  return false;
}

std::unique_ptr<SchedulerEntry> SchedulerEntry::makeEntry(
    ScheduleHeuristic sh,
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  std::unique_ptr<SchedulerEntry> scheduler_entry = nullptr;
  switch (sh) {
    case ScheduleHeuristic::NoOp:
      scheduler_entry =
          std::make_unique<NoOpScheduler>(fusion, runtime_info, data_cache);
      break;
    case ScheduleHeuristic::PointWise:
      scheduler_entry = std::make_unique<PointWiseScheduler>(
          fusion, runtime_info, data_cache);
      break;
    case ScheduleHeuristic::Reduction:
      scheduler_entry = std::make_unique<ReductionScheduler>(
          fusion, runtime_info, data_cache);
      break;
    case ScheduleHeuristic::Persistent:
      scheduler_entry = std::make_unique<PersistentKernelScheduler>(
          fusion, runtime_info, data_cache);
      break;
    case ScheduleHeuristic::Transpose:
      scheduler_entry = std::make_unique<TransposeScheduler>(
          fusion, runtime_info, data_cache);
      break;
    case ScheduleHeuristic::Matmul:
      scheduler_entry =
          std::make_unique<MatmulScheduler>(fusion, runtime_info, data_cache);
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "unreachable");
  }

  return scheduler_entry;
}

// Simply loop through the list as baseline strategy
c10::optional<ScheduleHeuristic> SchedulerEntry::proposeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info) {
  for (auto sh : all_heuristics()) {
    if (canSchedule(sh, fusion, runtime_info)) {
      scheduler_debug_utils::canScheduleMessage("***Accepted*** as: ", sh);
      return sh;
    }
  }
  return c10::nullopt;
}

size_t SchedulerEntryHash::operator()(const SchedulerEntry& se) const {
  return se.params()->hash();
}

std::string toString(ScheduleHeuristic sh) {
  switch (sh) {
    case ScheduleHeuristic::NoOp:
      return "no-op";
    case ScheduleHeuristic::PointWise:
      return "pointwise";
    case ScheduleHeuristic::Reduction:
      return "reduction";
    case ScheduleHeuristic::Persistent:
      return "persistent";
    case ScheduleHeuristic::Transpose:
      return "transpose";
    case ScheduleHeuristic::Matmul:
      return "matmul";
    default:
      TORCH_INTERNAL_ASSERT(false, "undefined schedule");
  }
  return "";
}

std::ostream& operator<<(std::ostream& os, ScheduleHeuristic sh) {
  os << toString(sh);
  return os;
}

namespace {

//! CompileTimeInfo is the actual subclass of CompileTimeInfoBase that will
//!  be stored in the data cache. It owns a data_ state internally of the
//!  dataType defined within the entry class, which are listed in compile
//!  time info header.
template <typename EntryClass>
class CompileTimeInfo : public HeuristicCompileTime::CompileTimeInfoBase {
 public:
  CompileTimeInfo(std::unique_ptr<typename EntryClass::DataType> data)
      : CompileTimeInfoBase(EntryClass::EntryType), data_(std::move(data)) {}

  typename EntryClass::DataType* get() {
    return data_.get();
  }

 private:
  std::unique_ptr<typename EntryClass::DataType> data_;
};

} // namespace

HeuristicSummary::HeuristicSummary(
    Fusion* fusion,
    ScheduleHeuristic heuristic,
    SchedulerRuntimeInfo& runtime_info)
    : heuristic_(heuristic), recording_(true) {
  switch (heuristic) {
    case ScheduleHeuristic::NoOp:
      NoOpScheduler::canScheduleRunTime(fusion, runtime_info, this);
      break;
    case ScheduleHeuristic::PointWise:
      getPointwiseHeuristics(fusion, runtime_info, this);
      PointWiseScheduler::canScheduleRunTime(fusion, runtime_info, this);
      break;
    case ScheduleHeuristic::Reduction:
      getReductionHeuristics(fusion, runtime_info, this);
      ReductionScheduler::canScheduleRunTime(fusion, runtime_info, this);
      break;
    case ScheduleHeuristic::Persistent:
      getPersistentHeuristics(fusion, runtime_info, this);
      PersistentKernelScheduler::canScheduleRunTime(fusion, runtime_info, this);
      break;
    case ScheduleHeuristic::Transpose:
      getTransposeHeuristics(fusion, runtime_info, this);
      TransposeScheduler::canScheduleRunTime(fusion, runtime_info, this);
      break;
    case ScheduleHeuristic::Matmul: {
      const auto heuristics = getMatmulHeuristics(fusion, runtime_info, this);
      TORCH_INTERNAL_ASSERT(heuristics, "Failed to get matmul heuristics");
      const auto canSchedule =
          MatmulScheduler::canScheduleRunTime(fusion, runtime_info, this);
      TORCH_INTERNAL_ASSERT(
          canSchedule, "Could not schedule matmul (run time)");
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "unknown heuristic");
  }
  validate();
  recording_ = false;
}

void HeuristicSummary::validate() const {
  switch (heuristic_) {
    case ScheduleHeuristic::NoOp: {
      // TODO: need to cache the dynamically zero inputs?
      break;
    }
    case ScheduleHeuristic::Transpose:
    case ScheduleHeuristic::PointWise: {
      if (heuristic_ == ScheduleHeuristic::PointWise) {
        TORCH_INTERNAL_ASSERT(entry_type_map_.count(EntryType::DOMAIN_MAP));
        TORCH_INTERNAL_ASSERT(
            entry_type_map_.count(EntryType::REFERENCE_TENSORS));
        TORCH_INTERNAL_ASSERT(
            entry_type_map_.count(EntryType::VECTORIZABLE_INPUTS_AND_OUTPUTS));
        TORCH_INTERNAL_ASSERT(entry_type_map_.count(EntryType::VECTORIZE_MAPS));
        TORCH_INTERNAL_ASSERT(
            entry_type_map_.count(EntryType::BROADCAST_BYTE_MULTIPLES));
        TORCH_INTERNAL_ASSERT(
            entry_type_map_.count(EntryType::CAN_SCHEDULE_TRANSPOSE));
        auto can_schedule_transpose =
            entry_type_map_.at(EntryType::CAN_SCHEDULE_TRANSPOSE)
                ->as<CompileTimeInfo<
                    HeuristicCompileTime::CanScheduleTranspose>>()
                ->get();
        if (!*can_schedule_transpose) {
          break;
        }
      }
      TORCH_INTERNAL_ASSERT(
          entry_type_map_.count(EntryType::TRANSPOSE_DOMAIN_MAP));
      TORCH_INTERNAL_ASSERT(entry_type_map_.count(
          EntryType::INPUTS_AND_OUTPUTS_INNER_DIM_GROUPS));
      TORCH_INTERNAL_ASSERT(
          entry_type_map_.count(EntryType::REFERENCE_TENSORS_FOR_GROUPS));
      TORCH_INTERNAL_ASSERT(
          entry_type_map_.count(EntryType::INNER_MOST_DIMS_INFO));
      break;
    }
    case ScheduleHeuristic::Reduction: {
      TORCH_INTERNAL_ASSERT(entry_type_map_.count(EntryType::REDUCTION_TVS));
      TORCH_INTERNAL_ASSERT(
          entry_type_map_.count(EntryType::VECTORIZABLE_INPUTS_AND_OUTPUTS));
      TORCH_INTERNAL_ASSERT(entry_type_map_.count(EntryType::VECTORIZE_MAPS));
      TORCH_INTERNAL_ASSERT(
          entry_type_map_.count(EntryType::UNROLLABLE_INPUTS_AND_OUTPUTS));
      break;
    }
    case ScheduleHeuristic::Persistent: {
      TORCH_INTERNAL_ASSERT(entry_type_map_.count(EntryType::REDUCTION_TVS));
      TORCH_INTERNAL_ASSERT(
          entry_type_map_.count(EntryType::VECTORIZABLE_INPUTS_AND_OUTPUTS));
      TORCH_INTERNAL_ASSERT(entry_type_map_.count(EntryType::VECTORIZE_MAPS));
      TORCH_INTERNAL_ASSERT(
          entry_type_map_.count(EntryType::UNROLLABLE_INPUTS_AND_OUTPUTS));
      TORCH_INTERNAL_ASSERT(
          entry_type_map_.count(EntryType::PERSISTENT_BUFFER_INFO));
      // If check persistent factor only when persistent buffers needed.
      auto persistent_buffer_info =
          entry_type_map_.at(EntryType::PERSISTENT_BUFFER_INFO)
              ->as<
                  CompileTimeInfo<HeuristicCompileTime::PersistentBufferInfo>>()
              ->get();
      TORCH_INTERNAL_ASSERT(
          !persistent_buffer_info->persistent_buffers.empty() &&
          entry_type_map_.count(EntryType::SCOPE_PERSISTENT_FACTOR_INFO));
      break;
    }
    case ScheduleHeuristic::Matmul: {
      // TODO: add a proper set of checks
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "unknown heuristic");
  }
}

void HeuristicSummary::insert(HeuristicSummary::EntryOwningPtr new_entry) {
  TORCH_INTERNAL_ASSERT(
      recording_, "should only insert entries at recording phase");
  // Just override when insertion duplicates, equality not checked.
  entry_type_map_[new_entry->type()] = new_entry.get();
  entries_.emplace_back(std::move(new_entry));
}

template <typename EntryClass>
HeuristicSummaryEntry<EntryClass>::HeuristicSummaryEntry(
    HeuristicSummary* data_cache,
    MakerFnType fn) {
  using InfoType = CompileTimeInfo<EntryClass>;

  if (!data_cache || data_cache->isRecording()) {
    owned_data_ = fn();
    data_ptr_ = owned_data_.get();

    if (data_cache) {
      std::unique_ptr<HeuristicCompileTime::CompileTimeInfoBase> new_entry =
          std::make_unique<InfoType>(std::move(owned_data_));
      data_cache->insert(std::move(new_entry));
    }
  } else {
    data_ptr_ =
        data_cache->at(EntryClass::EntryType)->template as<InfoType>()->get();
  }
}

// Template instantiation for pre-defined cache entries
template class HeuristicSummaryEntry<HeuristicCompileTime::DomainMap>;
template class HeuristicSummaryEntry<HeuristicCompileTime::TransposeDomainMap>;
template class HeuristicSummaryEntry<HeuristicCompileTime::ReferenceTensors>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::ReferenceTensorsForGroups>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::VectorizableInputsAndOutputs>;
template class HeuristicSummaryEntry<HeuristicCompileTime::VectorizeMaps>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::InputsOutputsInnerDimGroups>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::UnrollableInputsAndOutputs>;
template class HeuristicSummaryEntry<HeuristicCompileTime::ReductionTVs>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::PersistentBufferInfo>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::ScopePersistentFactorInfo>;
template class HeuristicSummaryEntry<HeuristicCompileTime::BroadcastMultiples>;
template class HeuristicSummaryEntry<HeuristicCompileTime::InnerMostDimInfo>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::CanScheduleTranspose>;

} // namespace nvfuser
