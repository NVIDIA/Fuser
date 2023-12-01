// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <executor_kernel_arg.h>
#include <ir/utils.h>
#include <root_domain_map.h>
#include <scheduler/debug_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/utils.h>

namespace nvfuser {

namespace registry_utils {

bool checkPatternEquivalence(
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
  return false;
}

PrimDataType getTensorIndexType(TensorView* tv, ExpressionEvaluator& ee) {
  NVF_ERROR(
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
  NVF_ERROR(
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
    NVF_ERROR(
        extent.hasValue(),
        "Axis with unknown extent found: ",
        id->toString(),
        ", tensor: ",
        tv->toString());

    auto extent_int = extent.as<int64_t>();

    NVF_ERROR(extent_int >= 0, "Unexpected size of axis: ", extent_int);

    if (extent_int > 0) {
      if (index_type_helper.addDim(extent.as<int64_t>(), stride) ==
          PrimDataType::Int) {
        return PrimDataType::Int;
      }
      stride *= extent.as<int64_t>();
    }
  }

  return index_type_helper.getType();
}

} // namespace

// TODO: remove this requirement entirely
bool rejectScheduleForMemoryPromotion(
    Fusion* fusion,
    ScheduleHeuristic schedule_strategy) {
  for (auto expr : fusion->exprs()) {
    if (expr->isOneOf<SelectOp, IndexSelectOp, TorchGatherOp>()) {
      // For now, only relax the input requirement when it's
      // take_along_axis. Also since this would require memory
      // promotion, i.e., persistent global sync in the case of
      // block-parallel ops, it needs to be explictly enabled.
      if (expr->isA<TorchGatherOp>() &&
          expr->as<TorchGatherOp>()->exactSizes() &&
          isOptionEnabled(EnableOption::MemoryPromotion)) {
        continue;
      }
      if (rejectScheduleFusionInputRequirement(expr, schedule_strategy)) {
        return true;
      }
    }

    // Similarly, ops based resize, such as like slice, pad and cat,
    // may require memory promotion. Require them to be done with
    // fusion inputs unless explicitly allowed
    if (!isOptionEnabled(EnableOption::MemoryPromotion) &&
        std::any_of(
            expr->outputs().begin(), expr->outputs().end(), [](Val* output) {
              return output->isA<TensorView>() &&
                  ir_utils::hasResizedRfactor(output->as<TensorView>());
            })) {
      if (rejectScheduleFusionInputRequirement(expr, schedule_strategy)) {
        return true;
      }
    }
  }
  return false;
}

bool isConnectedFusionGraph(Fusion* fusion) {
  if (fusion->outputs().empty()) {
    // Trivial case interpreted as connected
    return true;
  }

  // A set of connected components on the fusion graph
  DisjointSets<Val*> component_sets;

  NVF_ERROR(
      !fusion->outputs().empty(), "Fusion without output is not supported");
  auto output0 = fusion->outputs()[0];
  component_sets.initializeSet(output0);

  // Iterate through all used exprs
  for (auto expr : fusion->exprs()) {
    NVF_ERROR(!expr->outputs().empty(), "unknown expr with zero output");

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
  for (Val* out : fusion->outputs()) {
    if (Val* in = fusion->getOutputAlias(out).first; in != nullptr) {
      component_sets.mapEntries(out, in);
    }
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
      NVF_ERROR(
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
      NVF_ERROR(
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

  NVF_ERROR(dims.empty(), "Error processing ", dims, " in registry.cpp.");

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
      NVF_ERROR(
          find_it != reduction_reference->getMaybeRFactorDomain().end(),
          "Issue with view analysis on reduction like schedule, with reference: ",
          reduction_reference->toString());
      auto rfactor_pos = std::distance(
          reduction_reference->getMaybeRFactorDomain().begin(), find_it);
      NVF_ERROR(
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

bool SchedulerTopologyChecker::hasNonNormalizePostReductionBCast(
    Fusion* fusion) {
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
    auto forward_tv_chains = tvChains(DependencyCheck::getAllUseChains(red_tv));
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
            forward_pairwise_root_map.mapProducerToConsumer();

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
                  input_of_forward_running_consumer, forward_running_consumer));

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

            NVF_ERROR(backward_running_producer == forward_running_consumer);

            while (!backward_tv_chain.empty()) {
              backward_running_consumer = backward_running_producer;
              backward_running_producer = backward_tv_chain.back();
              backward_tv_chain.pop_back();

              std::vector<IterDomain*> running_resolved_ids;

              auto backward_pairwise_root_map = PairwiseRootDomainMap(
                  backward_running_producer, backward_running_consumer);

              auto backward_c2p_root_map =
                  backward_pairwise_root_map.mapConsumerToProducer();

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
bool SchedulerTopologyChecker::hasPostReductionBCast(Fusion* fusion) {
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
          auto p2c_root_map = pairwise_root_map.mapProducerToConsumer();

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
bool SchedulerTopologyChecker::supportedPostReductionFusion(
    Fusion* fusion,
    std::vector<TensorView*> reduction_tvs) {
  NVF_ERROR(!reduction_tvs.empty());
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
        reduction_tv_set, {fusion->outputs().begin(), fusion->outputs().end()});

    if (post_reduction_vals.empty()) {
      return true;
    }

    auto reduction_inputs =
        IterVisitor::getInputsTo({reduction_tvs.begin(), reduction_tvs.end()});

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

// Checks if there's any gather-like ops that result in non-resolved
// broadcast domains and then get squeezed before reaching reduction
// TVs. The reduction scheduler uses reduction TVs as a scheduling
// reference, so that won't be able to schedule the broadcast ID if
// squeezed and its corresponding index-accessed producer ID, and
// any IDs that the producer ID depends on.
//
// This analysis has some similarity as DomainMap. Can be
// consolidated?
bool SchedulerTopologyChecker::hasGatherToBroadcastBeforeReduction(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs) {
  std::vector<Val*> reduction_inputs;
  const auto all_exprs = StmtSort::getExprsBetween(
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
        broadcast_consumer_of_indexed_ids.insert(consumer_of_indexed_producer);
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

} // namespace registry_utils

} // namespace nvfuser
