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
  NVF_CHECK(!axes.empty(), "No axes provided to rfactor replay.");

  const int64_t kNumDims = original_td->nDims();

  NVF_CHECK((int64_t)axes.size() < kNumDims);

  // Check that axes are valid
  std::for_each(axes.begin(), axes.end(), [kNumDims](int64_t i) {
    NVF_CHECK(
        i >= -kNumDims && i < kNumDims,
        "Rfactor received an axis outside the number of dims in the tensor.",
        " Acceptable inclusive range is ",
        -kNumDims,
        " to ",
        kNumDims - 1);
  });

  NVF_CHECK(
      std::all_of(
          axes.begin(),
          axes.end(),
          [original_td](int64_t i) {
            return original_td->axis(i)->isReduction();
          }),
      "Cannot rfactor axes that are not reduction axes.");

  std::vector<IterDomain*> original_td_root = original_td->root();

  // Put in a set to make searching easy
  std::unordered_set<IterDomain*> rfactor_root_axes;
  std::transform(
      axes.begin(),
      axes.end(),
      std::inserter(rfactor_root_axes, rfactor_root_axes.begin()),
      [&original_td_root](int64_t pos) { return original_td_root.at(pos); });

  // Generate a new TensorDomain and set up map from one root to this one.
  std::vector<IterDomain*> new_producer_root;
  new_producer_root.reserve(original_td_root.size());

  std::transform(
      original_td_root.begin(),
      original_td_root.end(),
      std::back_inserter(new_producer_root),
      [&](IterDomain* id) {
        // If this is an rfactor root, it will be a reduction in this stage
        if (rfactor_root_axes.find(id) != rfactor_root_axes.end()) {
          return IterDomainBuilder(id->start(), id->extent())
              .stop_offset(id->stopOffset())
              .iter_type(IterType::Reduction)
              .build();
          // If this is not an rfactor root, but a reduction root, it should be
          // turned into an iteration domain
        } else if (id->isReduction()) {
          return IterDomainBuilder(id->start(), id->extent())
              .stop_offset(id->stopOffset())
              .build();
        } else {
          return id->cloneWithoutRFactor();
        }
      });

  TensorDomain* producer_domain = IrBuilder::create<TensorDomain>(
      new_producer_root,
      TensorDomain::getContiguityFilledWith(new_producer_root, true));

  std::vector<IterDomain*> new_consumer_root;
  new_consumer_root.reserve(original_td_root.size() - axes.size());
  for (IterDomain* id : original_td_root) {
    // If this is an rfactor root, skip it at this stage
    if (rfactor_root_axes.find(id) != rfactor_root_axes.end()) {
      continue;
    }
    new_consumer_root.push_back(id->cloneWithoutRFactor());
  }

  TensorDomain* consumer_domain = IrBuilder::create<TensorDomain>(
      new_consumer_root,
      TensorDomain::getContiguityFilledWith(new_consumer_root, true));

  return std::make_pair(producer_domain, consumer_domain);
}

void applyReductionFactor(
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

void FactorReductionPass::runPass(Fusion* fusion) {
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
    const std::vector<IterDomain*>& tv_root_domain = tv->getRootDomain();

    // Initialize reference subset if empty
    if (tv_subset.empty()) {
      std::copy_if(
          tv_root_domain.begin(),
          tv_root_domain.end(),
          std::inserter(id_subset, id_subset.begin()),
          [](IterDomain* id) { return id->isReduction(); });
      tv_subset.push_back(tv);
      continue;
    }

    // Collect reduction ids for this TensorView
    std::vector<IterDomain*> reduction_ids;
    std::copy_if(
        tv_root_domain.begin(),
        tv_root_domain.end(),
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
    const std::vector<IterDomain*>& tv_root_domain = tv->getRootDomain();

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
              tv_root_domain.begin(),
              tv_root_domain.end(),
              [&](IterDomain* id) {
                return val_sets.permissiveAreMapped(subset_id, id);
              });
          return std::distance(tv_root_domain.begin(), iter);
        });

    size_t num_reduction_ids = std::count_if(
        tv_root_domain.begin(), tv_root_domain.end(), [](IterDomain* id) {
          return id->isReduction();
        });

    // Skip if all ids are used for this TensorView
    if (rfactor_indices.size() < num_reduction_ids) {
      // Separate common reduction axes
      applyReductionFactor(tv, rfactor_indices);
    }
  }
}

} // namespace nvfuser::preseg_passes
