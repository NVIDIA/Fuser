// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/factor_amax_reduction.h>

#include <id_model/id_model.h>
#include <scheduler/utils.h>
#include <unordered_set>
#include <vector>

#include <ir/utils.h>

namespace nvfuser::preseg_passes {

namespace {

// Transform the provided tensor domain to two domains, a producer and
// consumer domain. These domains are created by taking axes and reducing them
// in the producer domain, and taking the remaining reduction axes and
// reducing them in the consumer domain.
std::pair<TensorDomain*, TensorDomain*> factorReductionDomain(
    TensorDomain* original_td,
    const std::vector<int64_t>& axes) {
  NVF_CHECK(!axes.empty(), "No axes provided to factorReductionDomain.");

  int64_t num_dims = original_td->nDims();

  NVF_CHECK((int64_t)axes.size() < num_dims);

  // Check that axes are valid
  std::for_each(axes.begin(), axes.end(), [num_dims](int64_t i) {
    NVF_CHECK(
        i >= -num_dims && i < num_dims,
        "factorReductionDomaain received an axis outside the number of dims in",
        "the tensor. Acceptable inclusive range is ",
        -num_dims,
        " to ",
        num_dims - 1);
  });

  NVF_CHECK(
      std::all_of(
          axes.begin(),
          axes.end(),
          [original_td](int64_t i) {
            return original_td->axis(i)->isReduction();
          }),
      "Cannot rfactor axes that are not reduction axes.");

  const std::vector<IterDomain*>& original_td_logical = original_td->logical();

  // Place iterDomain axes in a set to make searching easy
  std::unordered_set<IterDomain*> rfactor_logical_axes;
  std::transform(
      axes.begin(),
      axes.end(),
      std::inserter(rfactor_logical_axes, rfactor_logical_axes.begin()),
      [&original_td_logical](int64_t pos) {
        return original_td_logical.at(pos);
      });

  // Generate a new TensorDomain and set up a map from one logical to this one.
  std::vector<IterDomain*> new_producer_logical;
  new_producer_logical.reserve(original_td_logical.size());

  std::transform(
      original_td_logical.begin(),
      original_td_logical.end(),
      std::back_inserter(new_producer_logical),
      [&](IterDomain* id) {
        if (rfactor_logical_axes.find(id) != rfactor_logical_axes.end()) {
          // If this is a rfactor axis, it will be a reduction iterDomain in the
          // producer.
          return IterDomainBuilder(id->start(), id->extent())
              .stop_offset(id->stopOffset())
              .iter_type(IterType::Reduction)
              .build();
        } else if (id->isReduction()) {
          // If this is a reduction iterDomain but not a rfactor axis, convert
          // it to an iteration domain.
          return IterDomainBuilder(id->start(), id->extent())
              .stop_offset(id->stopOffset())
              .build();
        } else {
          return id->cloneWithoutRFactor();
        }
      });

  TensorDomain* producer_domain = IrBuilder::create<TensorDomain>(
      new_producer_logical,
      TensorDomain::getContiguityFilledWith(
          new_producer_logical, /*fill_value=*/true));

  std::vector<IterDomain*> new_consumer_logical;
  new_consumer_logical.reserve(original_td_logical.size() - axes.size());
  for (IterDomain* id : original_td_logical) {
    // If this is an rfactor axis, skip it at the consumer.
    if (rfactor_logical_axes.find(id) != rfactor_logical_axes.end()) {
      continue;
    }
    new_consumer_logical.push_back(id->cloneWithoutRFactor());
  }

  TensorDomain* consumer_domain = IrBuilder::create<TensorDomain>(
      new_consumer_logical,
      TensorDomain::getContiguityFilledWith(
          new_consumer_logical, /*fill_value=*/true));

  return std::make_pair(producer_domain, consumer_domain);
}

// This function is derived from TensorView::rfactor. It is used as a
// pre-scheduling operation, so there is not transformation replay for producer
// and consumer TensorViews.
void factorReductionTensorView(
    TensorView* consumer,
    const std::vector<int64_t>& axes) {
  FusionGuard fg(consumer->fusion());
  NVF_ERROR(consumer->nDims() > 0, "Tried reduction factor a 0-dim TensorView");
  NVF_CHECK(
      consumer->definition() != nullptr &&
          (consumer->definition()->isStrictlyOneOf<ReductionOp>()),
      "Error factoring out reduction axes from",
      consumer->toString(),
      " its definition is either a nullptr or not a reduction.");

  // Split tensor view into 2 parts
  auto&& [producer_domain, consumer_domain] =
      factorReductionDomain(consumer->domain(), axes);

  // Create the new producer
  TensorView* producer = IrBuilder::create<TensorView>(
      producer_domain, consumer->getDataType().value());

  // This TensorView is the consumer; Update its domain
  consumer->setDomain(consumer_domain);

  ReductionOp* this_reduction =
      dynamic_cast<ReductionOp*>(consumer->definition());
  // Setup dependency chain, inserting producer before this op.
  // Expr* producer_definition =
  IrBuilder::create<ReductionOp>(
      this_reduction->getReductionOpType(),
      this_reduction->init(),
      producer,
      this_reduction->in());

  // Expr* consumer_definition =
  IrBuilder::create<ReductionOp>(
      this_reduction->getReductionOpType(),
      this_reduction->init(),
      consumer,
      producer);
}

// Determine if TensorView has the desired UnaryOp definition
bool findUnaryDefinition(TensorView* tv, UnaryOpType op_type) {
  if (!tv->definition()->isA<UnaryOp>()) {
    return false;
  }

  UnaryOp* uop = tv->definition()->as<UnaryOp>();
  if (uop->getUnaryOpType() != op_type) {
    return false;
  }

  return true;
}

// Determine if TensorView has the desired ReductionOp definition
bool findReductionDefinition(TensorView* tv, BinaryOpType op_type) {
  if (!tv->definition()->isA<ReductionOp>()) {
    return false;
  }

  ReductionOp* rop = tv->definition()->as<ReductionOp>();
  if (rop->getReductionOpType() != op_type) {
    return false;
  }

  return true;
}

// Detect amax pattern using a finite state machine.
//
// There are six states in FSM.
// Start, Broadcast, Cast, Max-Reduction, Pass, Fail
//
// Pass State: Abs
// Fail States: Start, Invalid
// Intermediate States: Broadcast, Cast, MaxReduction
//
// Adjacency table for FSM.
// From Start to:
//  1) Cast
//  2) Broadcast
//  3) MaxReduction
//  4) Fail
//
// From Broadcast to:
//  1) MaxReduction
//  2) Fail
//
// From Cast to:
//  1) MaxReduction
//  2) Broadcast
//  3) Fail
//
// From Max-Reduction to:
//  1) Success if input TV definition is abs
//  2) Fail
//
// Return nullptr in Fail state.
// Return the reduction TensorView in Pass state.
TensorView* detectAmaxPattern(TensorView* tv) {
  enum State { Start, Invalid, Abs, Broadcast, Cast, MaxReduction };

  // Create state transition map
  std::unordered_map<State, std::vector<State>> valid_states;
  valid_states.emplace(
      Start, std::vector<State>({Broadcast, Cast, MaxReduction}));
  valid_states.emplace(Broadcast, std::vector<State>({MaxReduction}));
  valid_states.emplace(Cast, std::vector<State>({Broadcast, MaxReduction}));
  valid_states.emplace(MaxReduction, std::vector<State>({Abs}));

  // Initial state
  TensorView* max_reduction_tv = nullptr;
  TensorView* current_tv = tv;
  State current_state = Start;
  State next_state = Start;

  while (current_tv != nullptr) {
    // Get potential next state based on TensorView definition
    if (findUnaryDefinition(current_tv, UnaryOpType::Cast)) {
      next_state = Cast;
    } else if (findUnaryDefinition(current_tv, UnaryOpType::Abs)) {
      next_state = Abs;
      break;
    } else if (findReductionDefinition(current_tv, BinaryOpType::Max)) {
      next_state = MaxReduction;
    } else if (current_tv->definition()->isA<BroadcastOp>()) {
      next_state = Broadcast;
    } else {
      next_state = Invalid;
    }

    // Validate next state given current state
    bool is_valid = (valid_states.count(current_state) == 0) ||
        std::any_of(valid_states.at(current_state).begin(),
                    valid_states.at(current_state).end(),
                    [&](State s) { return s == next_state; });
    if (!is_valid) {
      next_state = Invalid;
    }

    // Transition to next state
    switch (next_state) {
      case Start:
      case Invalid: {
        // For these failure states, we clear max_reduction_tv.
        max_reduction_tv = nullptr;
        [[fallthrough]];
      }
      case Abs: {
        // For the pass state, we return max_reduction_tv.
        current_tv = nullptr;
        break;
      }
      case MaxReduction: {
        // Set max_reduction_tv
        max_reduction_tv = current_tv;
        [[fallthrough]];
      }
      case Broadcast:
      case Cast: {
        // Move to input TensorView of definition
        current_tv = current_tv->definition()->input(0)->as<TensorView>();
        break;
      }
    }
    current_state = next_state;
  }

  NVF_ERROR(
      max_reduction_tv == nullptr ||
      findReductionDefinition(max_reduction_tv, BinaryOpType::Max));
  return max_reduction_tv;
}

// Select a partial reduction TensorView to factor the amax reduction.
// Given the lack of information in the presegmentation pass, we select the
// partial reduction with shortest dependency chain with amax reduction.
TensorView* findUpstreamReduction(Fusion* fusion, TensorView* amax_reduction) {
  NVF_ERROR(
      amax_reduction != nullptr &&
          findReductionDefinition(amax_reduction, BinaryOpType::Max),
      "Expected max reduction TensorView");

  std::vector<TensorView*> reduction_tvs =
      scheduler_utils::getReductionTvs(fusion);

  // Short-circuit: Return nullptr if there is not any upstream reduction
  // TensorViews.
  if (reduction_tvs.empty()) {
    return nullptr;
  }

  // Find the closest upstream reduction that is not amax reduction.
  TensorView* upstream_reduction = nullptr;
  size_t distance = ULONG_MAX;
  for (TensorView* tv : reduction_tvs) {
    if (tv == amax_reduction) {
      continue;
    }

    std::deque<Val*> chain = DependencyCheck::getSingleDependencyChain(
        /*dependency=*/tv, /*of=*/amax_reduction);
    if (chain.empty()) {
      continue;
    }

    if (chain.size() < distance) {
      upstream_reduction = tv;
      distance = chain.size();
    }
  }
  return upstream_reduction;
}

// Find the subset of reduction iterDomains to factor from
// reference TensorView given upstream TensorView.
std::unordered_set<IterDomain*> findIdSubset(
    const DisjointSets<Val*>& exact_val_sets,
    TensorView* reference_tv,
    TensorView* upstream_tv) {
  // Gather reduction iterDomains from reference_tv
  std::unordered_set<IterDomain*> reduction_ids_for_reference_tv;
  std::copy_if(
      reference_tv->getLogicalDomain().begin(),
      reference_tv->getLogicalDomain().end(),
      std::inserter(
          reduction_ids_for_reference_tv, reduction_ids_for_reference_tv.end()),
      [](IterDomain* id) { return id->isReduction(); });

  // Return empty set if reference TensorView does not have any reduction
  // iterDomains.
  if (reduction_ids_for_reference_tv.empty()) {
    return std::unordered_set<IterDomain*>();
  }

  // Collect reduction ids from upstream_tv
  std::vector<IterDomain*> reduction_ids_for_upstream_tv;
  std::copy_if(
      upstream_tv->getLogicalDomain().begin(),
      upstream_tv->getLogicalDomain().end(),
      std::back_inserter(reduction_ids_for_upstream_tv),
      [](IterDomain* id) { return id->isReduction(); });

  // Return reduction ids for reference TensorView if upstream TensorView does
  // not have any reduction axes.
  if (reduction_ids_for_upstream_tv.empty()) {
    return reduction_ids_for_reference_tv;
  }

  // Get the intersection between reference and upstream TensorViews. Keep
  // reference iterDomain if any of the reduction iterDomains from upstream
  // TensorView are mapped via Exact IdGraph.
  std::unordered_set<IterDomain*> intersection;
  std::copy_if(
      reduction_ids_for_reference_tv.begin(),
      reduction_ids_for_reference_tv.end(),
      std::inserter(intersection, intersection.begin()),
      [&](IterDomain* subset_id) {
        return std::any_of(
            reduction_ids_for_upstream_tv.begin(),
            reduction_ids_for_upstream_tv.end(),
            [&](IterDomain* id) {
              return exact_val_sets.permissiveAreMapped(subset_id, id);
            });
      });

  return intersection;
}

// Get reduction indices to factor from current TensorView
//  * Scan through selected iterDomains
//  * Find corresponding match for this TensorView
//  * Return position for iterDomain in this TensorView
std::vector<int64_t> convertIterDomainToInteger(
    const DisjointSets<Val*>& exact_val_sets,
    const std::unordered_set<IterDomain*>& id_set,
    TensorView* tv) {
  std::vector<int64_t> indices;
  indices.reserve(id_set.size());

  std::transform(
      id_set.begin(),
      id_set.end(),
      std::back_inserter(indices),
      [&](IterDomain* id_from_set) {
        auto iter = std::find_if(
            tv->getLogicalDomain().begin(),
            tv->getLogicalDomain().end(),
            [&](IterDomain* id_from_tv) {
              return exact_val_sets.permissiveAreMapped(
                  id_from_set, id_from_tv);
            });
        return std::distance(tv->getLogicalDomain().begin(), iter);
      });

  return indices;
}

} // namespace

void FactorAmaxReductionPass::runPass(Fusion* fusion) {
  // Gather all amax reduction TensorViews
  std::vector<TensorView*> amax_reduction_tvs;
  for (Val* output : fusion->outputs()) {
    if (!output->isA<TensorView>()) {
      continue;
    }
    TensorView* amax_reduction = detectAmaxPattern(output->as<TensorView>());
    if (amax_reduction == nullptr) {
      continue;
    }
    amax_reduction_tvs.push_back(amax_reduction);
  }

  // Stop if we cannot find amax reduction pattern
  if (amax_reduction_tvs.empty()) {
    std::cout << "Failed to find amax reduction pattern." << std::endl;
    return;
  }

  // Create Exact Id Graph
  FusionGuard fg(fusion);
  IdModel id_model(fusion, /*build_graphs=*/false, /*allow_self_mapping=*/true);
  id_model.buildExactGraph();
  ValGraph exact_graph = id_model.idGraph(IdMappingMode::EXACT);
  const DisjointSets<Val*>& exact_val_sets = exact_graph.disjointValSets();

  for (TensorView* amax_reduction : amax_reduction_tvs) {
    // Detect dependency chain between amax and some reduction operation
    // Stop if we cannot find any compatible reduction tvs
    TensorView* upstream_tv = findUpstreamReduction(fusion, amax_reduction);
    if (upstream_tv == nullptr && upstream_tv != amax_reduction) {
      std::cout
          << "Failed to compatible reduction TensorView for amax reduction pattern"
          << std::endl;
      continue;
    }

    // Partition amax reduction axes into partial reduction.
    std::unordered_set<IterDomain*> reduction_id_subset =
        findIdSubset(exact_val_sets, amax_reduction, upstream_tv);
    NVF_ERROR(std::all_of(
        reduction_id_subset.begin(),
        reduction_id_subset.end(),
        [](IterDomain* id) { return id->isReduction(); }));

    // Map selected reduction iterDomains to integer axes
    std::vector<int64_t> rfactor_indices = convertIterDomainToInteger(
        exact_val_sets, reduction_id_subset, amax_reduction);

    size_t num_reduction_ids = std::count_if(
        amax_reduction->getLogicalDomain().begin(),
        amax_reduction->getLogicalDomain().end(),
        [](IterDomain* id) { return id->isReduction(); });

    // Skip if all IterDomains are used for this TensorView
    if (rfactor_indices.size() == num_reduction_ids) {
      continue;
    }

    // Create partial reduction given selected axes
    factorReductionTensorView(amax_reduction, rfactor_indices);
  }
}

} // namespace nvfuser::preseg_passes
