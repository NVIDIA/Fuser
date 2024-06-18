// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/factor_reduction.h>

#include <id_model/id_model.h>
#include <scheduler/utils.h>
#include <unordered_set>
#include <vector>

#include <ir/utils.h>

namespace nvfuser::preseg_passes {

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
      TensorDomain::getContiguityFilledWith(new_producer_logical, /*fill_value=*/true));

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
      TensorDomain::getContiguityFilledWith(new_consumer_logical, /*fill_value=*/true));

  return std::make_pair(producer_domain, consumer_domain);
}

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

// We are using a state machine to detect amax pattern.
//
// Start -> Cast
// Start -> MaxReduction
// Start -> Fail
//
// Cast -> MaxReduction
// Cast -> Fail
//
// Max_Reduction -> Success
// Max_Reduction -> Fail
//
// Fail -> Return nullptr
// Success -> Return reduction TensorView
TensorView* detectAmaxPattern(TensorView* tv) {
  enum { Start, Cast, MaxReduction } state = Start;

  TensorView* max_reduction_tv = nullptr;
  TensorView* current_tv = tv;
  std::cout << "==========" << std::endl;
  while (current_tv != nullptr) {
    switch (state) {
      case Start: {
        std::cout << "from start" << std::endl;
        if (findUnaryDefinition(current_tv, UnaryOpType::Cast)) {
          // Move state from Start to Cast if we have a Cast definition
          std::cout << " to cast" << std::endl;
          current_tv = current_tv->definition()->input(0)->as<TensorView>();
          state = Cast;
          break;
        } else if (findReductionDefinition(current_tv, BinaryOpType::Max)) {
          std::cout << " to max-red" << std::endl;
          // Move state from Start to MaxReduction if we have a Max reduction
          // definition
          max_reduction_tv = current_tv;
          current_tv = current_tv->definition()->input(0)->as<TensorView>();
          state = MaxReduction;
          break;
        }
        // Otherwise, move state from Start to Fail
        std::cout << " to fail" << std::endl;
        current_tv = nullptr;
        max_reduction_tv = nullptr;
        break;
      }
      case Cast: {
        std::cout << "from cast" << std::endl;
        if (findReductionDefinition(current_tv, BinaryOpType::Max)) {
          // Move state from Cast to MaxReduction if we have a Max reduction
          // definition
          std::cout << " to max-red" << std::endl;
          max_reduction_tv = current_tv;
          current_tv = current_tv->definition()->input(0)->as<TensorView>();
          state = MaxReduction;
          break;
        }
        // Otherwise, move state from Cast to Fail
        std::cout << " to fail" << std::endl;
        current_tv = nullptr;
        max_reduction_tv = nullptr;
        break;
      }
      case MaxReduction: {
        std::cout << "max_reduction" << std::endl;
        if (findUnaryDefinition(current_tv, UnaryOpType::Abs)) {
          // Move state from MaxReduction to Success if we have an Abs
          // definition
          std::cout << " to pass" << std::endl;
          current_tv = nullptr;
          break;
        } else {
          // Otherwise, move state from MaxReduction to Fail
          std::cout << " to fail" << std::endl;
          current_tv = nullptr;
          max_reduction_tv = nullptr;
          break;
        }
      }
    }
  }
  std::cout << "==========" << std::endl;
  NVF_ERROR(
      max_reduction_tv == nullptr ||
      findReductionDefinition(max_reduction_tv, BinaryOpType::Max));
  return max_reduction_tv;
}

std::vector<TensorView*> findAmaxReductionDependencies(
    Fusion* fusion,
    TensorView* amax_reduction) {
  NVF_ERROR(
      amax_reduction != nullptr &&
      findReductionDefinition(amax_reduction, BinaryOpType::Max));

  std::vector<TensorView*> upstream_reductions;

  std::vector<TensorView*> reduction_tvs =
      scheduler_utils::getReductionTvs(fusion);
  if (reduction_tvs.size() <= 1) {
    return upstream_reductions;
  }

  std::copy_if(
      reduction_tvs.begin(),
      reduction_tvs.end(),
      std::back_inserter(upstream_reductions),
      [amax_reduction](TensorView* tv) {
        if (tv == amax_reduction) {
          return false;
        }
        if (!DependencyCheck::isDependencyOf(
                /*dependency=*/tv, /*of=*/amax_reduction)) {
          return false;
        }
        return true;
      });

  return upstream_reductions;
}

void FactorReductionPass::runPass(Fusion* fusion) {
  std::vector<Val*> output_tvs;

  // start from outputs tvs
  std::copy_if(
      fusion->outputs().begin(),
      fusion->outputs().end(),
      std::back_inserter(output_tvs),
      [](Val* v) { return v->isA<TensorView>(); });

  for (Val* output_tv : output_tvs) {
    TensorView* amax_reduction = detectAmaxPattern(output_tv->as<TensorView>());
    // Stop if we cannot find amax reduction pattern
    if (amax_reduction == nullptr) {
      std::cout << "Failed to find amax reduction pattern" << std::endl;
      continue;
    }

    // Detect dependency chain between amax and some reduction operation
    std::vector<TensorView*> dependency_tvs =
        findAmaxReductionDependencies(fusion, amax_reduction);
    // Stop if we cannot find any compatible reduction tvs
    if (dependency_tvs.empty()) {
      std::cout
          << "Failed to compatible reduction TensorView for amax reduction pattern"
          << std::endl;
      continue;
    }

    // Given TensorViews, partition reduction axes into compatible sets.
    FusionGuard fg(fusion);
    IdModel id_model(
        fusion, /*build_graphs=*/false, /*allow_self_mapping=*/true);
    id_model.buildExactGraph();
    ValGraph exact_graph = id_model.idGraph(IdMappingMode::EXACT);
    const DisjointSets<Val*>& val_sets = exact_graph.disjointValSets();

    // Common reduction iterDomains for reference TensorView
    std::unordered_set<IterDomain*> id_subset;
    std::copy_if(
        amax_reduction->getLogicalDomain().begin(),
        amax_reduction->getLogicalDomain().end(),
        std::inserter(id_subset, id_subset.begin()),
        [](IterDomain* id) { return id->isReduction(); });

    std::cout << amax_reduction->toString() << std::endl;
    std::cout << amax_reduction->getLogicalDomain().size() << std::endl;
    std::cout << "!!\t" << id_subset.size() << std::endl;

    for (TensorView* tv : dependency_tvs) {
      const std::vector<IterDomain*>& tv_logical_domain =
          tv->getLogicalDomain();

      // Collect reduction ids for this TensorView
      std::vector<IterDomain*> reduction_ids;
      std::copy_if(
          tv_logical_domain.begin(),
          tv_logical_domain.end(),
          std::back_inserter(reduction_ids),
          [](IterDomain* id) { return id->isReduction(); });

      // Get intersection from reference subset and this TensorView
      //  * Keep reference id if any of this TensorView's reduction ids are
      //    mapped via Exact IdGraph.
      std::unordered_set<IterDomain*> intersection;
      std::copy_if(
          id_subset.begin(),
          id_subset.end(),
          std::inserter(intersection, intersection.begin()),
          [&](IterDomain* subset_id) {
            return std::any_of(
                reduction_ids.begin(),
                reduction_ids.end(),
                [&](IterDomain* id) {
                  return val_sets.permissiveAreMapped(subset_id, id);
                });
          });

      // Update subsets if this TensorView has any common reduction axes
      if (!intersection.empty()) {
        id_subset.swap(intersection);
      }
    }

    std::cout << "!!\t" << id_subset.size() << std::endl;

    // Factor amax reduction into partial reductions.
    // Map common reduction iterDomains to integer axes

    // Get reduction indices to factor from current TensorView
    //  * Scan through reference ids
    //  * Find corresponding match for this TensorView
    //  * Return position for reduction id in this TensorView
    std::vector<int64_t> rfactor_indices;
    rfactor_indices.reserve(id_subset.size());
    std::transform(
        id_subset.begin(),
        id_subset.end(),
        std::back_inserter(rfactor_indices),
        [&](IterDomain* subset_id) {
          auto iter = std::find_if(
              amax_reduction->getLogicalDomain().begin(),
              amax_reduction->getLogicalDomain().end(),
              [&](IterDomain* id) {
                return val_sets.permissiveAreMapped(subset_id, id);
              });
          return std::distance(
              amax_reduction->getLogicalDomain().begin(), iter);
        });

    size_t num_reduction_ids = std::count_if(
        amax_reduction->getLogicalDomain().begin(),
        amax_reduction->getLogicalDomain().end(),
        [](IterDomain* id) { return id->isReduction(); });

    // Skip if all ids are used for this TensorView
    if (rfactor_indices.size() < num_reduction_ids) {
      std::cout << rfactor_indices << std::endl;
      factorReductionTensorView(amax_reduction, rfactor_indices);
      // Only apply partial rfactor once
      return;
    }
  }
}

// Gather all reduction TensorViews
// Find axes intersection common to all reductions
// Factor common axes for all reductions
void factorCommonReductionAxes(Fusion* fusion) {
  // Persistent schedule expects all reductions to have same axes.
  // Factor common reduction axes into separate reduction operations
  // to create better fusions.

  // Common reduction iterDomains for reference TensorView
  std::unordered_set<IterDomain*> id_subset;
  // All TensorViews with common reduction axes
  std::vector<TensorView*> tv_subset;

  std::vector<TensorView*> reduction_tvs =
      scheduler_utils::getReductionTvs(fusion);
  if (reduction_tvs.size() <= 1) {
    return;
  }

  FusionGuard fg(fusion);
  IdModel id_model(fusion, /*build_graphs=*/false, /*allow_self_mapping=*/true);
  id_model.buildExactGraph();
  ValGraph exact_graph = id_model.idGraph(IdMappingMode::EXACT);
  const DisjointSets<Val*>& val_sets = exact_graph.disjointValSets();

  for (TensorView* tv : reduction_tvs) {
    const std::vector<IterDomain*>& tv_logical_domain = tv->getLogicalDomain();

    // Initialize reference subset if empty
    if (tv_subset.empty()) {
      std::copy_if(
          tv_logical_domain.begin(),
          tv_logical_domain.end(),
          std::inserter(id_subset, id_subset.begin()),
          [](IterDomain* id) { return id->isReduction(); });
      tv_subset.push_back(tv);
      continue;
    }

    // Collect reduction ids for this TensorView
    std::vector<IterDomain*> reduction_ids;
    std::copy_if(
        tv_logical_domain.begin(),
        tv_logical_domain.end(),
        std::back_inserter(reduction_ids),
        [](IterDomain* id) { return id->isReduction(); });

    // Get intersection from reference subset and this TensorView
    //  * Keep reference id if any of this TensorView's reduction ids are
    //    mapped via Exact IdGraph.
    std::unordered_set<IterDomain*> intersection;
    std::copy_if(
        id_subset.begin(),
        id_subset.end(),
        std::inserter(intersection, intersection.begin()),
        [&](IterDomain* subset_id) {
          return std::any_of(
              reduction_ids.begin(), reduction_ids.end(), [&](IterDomain* id) {
                return val_sets.permissiveAreMapped(subset_id, id);
              });
        });

    // Update subsets if this TensorView has any common reduction axes
    if (!intersection.empty()) {
      id_subset.swap(intersection);
      tv_subset.push_back(tv);
    }
  }

  // All reduction iterDomains in last TensorView do not match any in id_subset
  // rfactor common subset of reduction axes for TensorView's in subset
  if (tv_subset.size() == 1) {
    return;
  }

  // Map common reduction iterDomains to integer axes
  for (TensorView* tv : tv_subset) {
    const std::vector<IterDomain*>& tv_logical_domain = tv->getLogicalDomain();

    // Get reduction indices to factor from current TensorView
    //  * Scan through reference ids
    //  * Find corresponding match for this TensorView
    //  * Return position for reduction id in this TensorView
    std::vector<int64_t> rfactor_indices;
    rfactor_indices.reserve(id_subset.size());
    std::transform(
        id_subset.begin(),
        id_subset.end(),
        std::back_inserter(rfactor_indices),
        [&](IterDomain* subset_id) {
          auto iter = std::find_if(
              tv_logical_domain.begin(),
              tv_logical_domain.end(),
              [&](IterDomain* id) {
                return val_sets.permissiveAreMapped(subset_id, id);
              });
          return std::distance(tv_logical_domain.begin(), iter);
        });

    size_t num_reduction_ids = std::count_if(
        tv_logical_domain.begin(), tv_logical_domain.end(), [](IterDomain* id) {
          return id->isReduction();
        });

    // Skip if all ids are used for this TensorView
    if (rfactor_indices.size() < num_reduction_ids) {
      // Separate common reduction axes
      factorReductionTensorView(tv, rfactor_indices);
    }
  }
}

} // namespace nvfuser::preseg_passes
